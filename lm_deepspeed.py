import math
import os
import random

import click
import deepspeed
import numpy as np
import torch
from datasets import load_dataset
from deepspeed import get_accelerator
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils import logger
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import default_data_collator, get_scheduler

import wandb
from d3pm_runner import D3PM
from dit import DDiT_Llama


class WikiTextDataset(Dataset):
    def __init__(
        self, tokenizer=None, type_path="train", max_seq_length=512, debug=False
    ):

        if debug:
            self.dataset = load_dataset("wikitext", f"wikitext-2-raw-v1", split="test")
        else:
            self.dataset = load_dataset(
                "wikimedia/wikipedia", "20231101.en", split="train"
            )

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        # logger.info(text)
        if self.tokenizer is not None:
            inputs = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.squeeze()
        else:
            # use byte encoding
            seq = list(text.encode("utf-8"))
            if len(seq) < self.max_seq_length:
                seq += [0] * (self.max_seq_length - len(seq))
            else:
                seq = seq[: self.max_seq_length]
            input_ids = torch.tensor(seq, dtype=torch.long)

        return {"input_ids": input_ids}


def _z3_params_to_fetch(param_list):
    return [
        p
        for p in param_list
        if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = zero_stage == 3
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema, "module") else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, "ds_id"):
                with deepspeed.zero.GatheredParameters(
                    _z3_params_to_fetch([v]), enabled=zero_stage_3
                ):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@click.command()
@click.option("--local_rank", default=-1, help="Local rank")
@click.option("--max_seq_length", default=256, help="Max sequence length")
@click.option("--num_train_epochs", default=5, help="Number of training epochs")
@click.option("--learning_rate", default=1e-4, help="Learning rate")
@click.option("--offload", default=False, help="Offload")
@click.option("--train_batch_size", default=1024, help="Train batch size")
@click.option(
    "--per_device_train_batch_size", default=64, help="Per device train batch size"
)
@click.option("--zero_stage", default=2, help="Zero stage")
@click.option("--seed", default=42, help="Seed")
@click.option("--run_name", default=None, help="Run name")
def main(
    local_rank,
    max_seq_length=256,
    num_train_epochs=5,
    learning_rate=1e-4,
    offload=False,
    train_batch_size=512,
    per_device_train_batch_size=64,
    zero_stage=2,
    seed=42,
    run_name=None,
):
    # first, set the seed
    set_seed(seed)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    if run_name is None:
        run_name = f"LR:{learning_rate}_max_seq_length:{max_seq_length}_num_train_epochs:{num_train_epochs}_offload:{offload}"

    if local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(local_rank)
        device = torch.device(get_accelerator().device_name(), local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    offload_device = "cpu" if offload else "none"

    ds_config = {
        "train_micro_batch_size_per_gpu": per_device_train_batch_size,
        "train_batch_size": train_batch_size,
        "zero_optimization": {
            "stage": zero_stage,
            "offload_param": {"device": offload_device},
            "offload_optimizer": {"device": offload_device},
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e7,
            "stage3_prefetch_bucket_size": 3e7,
            "memory_efficient_linear": False,
        },
        "bfloat16": {"enabled": True},
        "gradient_clipping": 1.0,
    }

    torch.distributed.barrier()

    global_rank = torch.distributed.get_rank()

    ##### DEFINE model, dataset, sampler, dataloader, optim, schedular
    N = 256
    with deepspeed.zero.Init(enabled=(zero_stage == 3)):

        d3pm = D3PM(
            DDiT_Llama(N, dim=768, n_layers=8),
            1000,
            num_classes=N,
            hybrid_loss_coeff=0.0,
        ).cuda()

    total_params = sum(p.numel() for p in d3pm.parameters())
    size_in_bytes = total_params * 4
    size_in_gb = size_in_bytes / (1024**3)
    logger.info(
        f"Model Size: {size_in_bytes}, {size_in_gb} GB, Total Param Count: {total_params / 1e6} M"
    )

    dataset = WikiTextDataset(max_seq_length=max_seq_length, debug=False)

    train_sampler = (
        RandomSampler(dataset)
        if local_rank == -1
        else DistributedSampler(dataset, seed=seed)
    )

    dataloader = DataLoader(
        dataset,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=per_device_train_batch_size,
    )

    optimizer = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=learning_rate)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_train_epochs * math.ceil(len(dataloader)),
    )

    d3pm.train()

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=d3pm, config=ds_config, lr_scheduler=lr_scheduler, optimizer=optimizer
    )

    global_step = 0

    ##### actual training loop

    if global_rank == 0:
        wandb.init(
            project="d3pm_wiki",
            name=run_name,
            config={
                "N": N,
                "max_seq_length": max_seq_length,
                "num_train_epochs": num_train_epochs,
                "learning_rate": learning_rate,
                "offload": offload,
                "train_batch_size": train_batch_size,
                "per_device_train_batch_size": per_device_train_batch_size,
                "zero_stage": zero_stage,
                "seed": seed,
            },
        )

    for i in range(num_train_epochs):
        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:

            x = x["input_ids"].to(model_engine.device)

            # discritize x to N bins

            loss, info = model_engine(x)
            model_engine.backward(loss)
            model_engine.step()

            get_accelerator().empty_cache()
            norm = model_engine.get_global_grad_norm()

            if global_step % 10 == 0:
                if global_rank == 0:
                    wandb.log({"train_loss": loss, "train_grad_norm": norm})

            pbar.set_description(
                f"norm: {norm}, vb_loss: {info['vb_loss']:.4f}, ce_loss: {info['ce_loss']:.4f}"
            )

            global_step += 1

            if global_step % 600 == 1:
                d3pm.eval()

                with torch.no_grad():
                    init_noise = torch.randint(0, N, (16, max_seq_length)).cuda()

                    outputs = d3pm.sample_with_image_sequence(
                        init_noise, None, stride=40
                    )
                    gen_outputs = []
                    total = 0
                    # back to sentence, byte to utf-8
                    for _i in range(16):
                        sent = outputs[-1][_i].cpu().tolist()
                        correctly_parsed = True
                        try:
                            sent = b"".join([bytes([i]) for i in sent]).decode("utf-8")
                        except:
                            # if there is error, just unicodec
                            correctly_parsed = False
                            sent = "".join([chr(i) for i in sent])
                        sent = (
                            f"[{_i}] Sample Correctly parsed: "
                            + str(correctly_parsed)
                            + "\n"
                            + sent
                        )
                        total += 1 if correctly_parsed else 0

                        gen_outputs.append(sent)

                    print(sent)
                    model_engine.train()

                    # make a nice html to show the generated outputs
                    html_formatted = "<br>".join(gen_outputs)
                    # log text
                    if global_rank == 0:
                        wandb.log(
                            {
                                "generated_text": wandb.Html(html_formatted),
                                "correctly_parsed": total,
                            }
                        )
                    if global_step % 3000 == 1:
                        save_zero_three_model(
                            model_engine, global_rank, "./ckpt", zero_stage=zero_stage
                        )

                        print(f"Model saved at {global_step}")


if __name__ == "__main__":
    main()
