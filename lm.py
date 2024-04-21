import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_dataset


import wandb


from d3pm_runner import D3PM
from dit import DDiT_Llama


class WikiTextDataset(Dataset):
    def __init__(self, tokenizer=None, type_path="train", max_length=512, debug=False):
        if debug:
            vernum = 2
        else:
            vernum = 103
        self.vernum = vernum
        self.dataset = load_dataset(
            "wikitext", f"wikitext-{vernum}-raw-v1", split=type_path
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return (
            int(len(self.dataset) * 0.1) if (self.vernum == 103) else len(self.dataset)
        )

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        # logger.info(text)
        if self.tokenizer is not None:
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.squeeze()
        else:
            # use byte encoding
            seq = list(text.encode("utf-8"))
            if len(seq) < self.max_length:
                seq += [0] * (self.max_length - len(seq))
            else:
                seq = seq[: self.max_length]
            input_ids = torch.tensor(seq, dtype=torch.long)

        return {"input_ids": input_ids}


if __name__ == "__main__":

    wandb.init(project="d3pm_wiki")

    N = 256

    d3pm = D3PM(
        DDiT_Llama(N, dim=1024), 1000, num_classes=N, hybrid_loss_coeff=0.0
    ).cuda()

    print(f"Total Param Count: {sum([p.numel() for p in d3pm.x0_model.parameters()])}")
    dataset = WikiTextDataset(max_length=128, debug=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16)
    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=2e-5)
    d3pm.train()

    n_epoch = 4000
    device = "cuda"

    global_step = 0
    for i in range(n_epoch):

        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            x = x["input_ids"].to(device)

            # discritize x to N bins

            loss, info = d3pm(x)

            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 5.0)

            with torch.no_grad():
                param_norm = sum([torch.norm(p) for p in d3pm.x0_model.parameters()])

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()

            if global_step % 10 == 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "train_grad_norm": norm,
                        "train_param_norm": param_norm,
                    }
                )

            pbar.set_description(
                f"loss: {loss_ema:.4f}, norm: {norm:.4f}, param_norm: {param_norm:.4f}, vb_loss: {info['vb_loss']:.4f}, ce_loss: {info['ce_loss']:.4f}"
            )
            optim.step()
            global_step += 1

            if global_step % 600 == 1:
                d3pm.eval()

                with torch.no_grad():

                    init_noise = torch.randint(0, N, (16, 128)).cuda()

                    outputs = d3pm.sample_with_image_sequence(
                        init_noise, None, stride=40
                    )

                    last_tokens = outputs[-1][0].cpu().tolist()
                    print(last_tokens)
                    # back to sentence, byte to utf-8
                    try:
                        last_text = b"".join([bytes([i]) for i in last_tokens]).decode(
                            "utf-8"
                        )
                    except:
                        # if there is error, just unicode
                        last_text = "".join([chr(i) for i in last_tokens])

                    print(last_text)

                    # log text
                    wandb.log(
                        {
                            "generated_text": wandb.Html(last_text),
                        }
                    )

                d3pm.train()

                if global_step % 3000 == 1:
                    torch.save(d3pm.state_dict(), f"ckpt/d3pm_wiki_{global_step}.pth")

                    print(f"Model saved at {global_step}")
