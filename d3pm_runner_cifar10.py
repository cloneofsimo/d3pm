import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from d3pm_runner import D3PM
from dit import DiT_Llama

if __name__ == "__main__":

    wandb.init(project="d3pm_cifar10")

    N = 8  # number of classes for discretized state per pixel
    d3pm = D3PM(
        DiT_Llama(3, N, dim=1024), 1000, num_classes=N, hybrid_loss_coeff=0.0
    ).cuda()
    print(f"Total Param Count: {sum([p.numel() for p in d3pm.x0_model.parameters()])}")
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16)
    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=2e-5)
    d3pm.train()

    n_epoch = 4000
    device = "cuda"

    global_step = 0
    for i in range(n_epoch):

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, cond in pbar:
            optim.zero_grad()
            x = x.to(device)
            cond = cond.to(device)

            # discritize x to N bins
            x_cat = (x * (N - 1)).round().long().clamp(0, N - 1)
            loss, info = d3pm(x_cat, cond)

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
                    cond = torch.arange(0, 16).cuda() % 10
                    init_noise = torch.randint(0, N, (16, 3, 32, 32)).cuda()

                    images = d3pm.sample_with_image_sequence(
                        init_noise, cond, stride=40
                    )
                    # image sequences to gif
                    gif = []
                    for image in images:
                        x_from_dataloader = x_cat[:16].cpu() / (N - 1)
                        this_image = image.float().cpu() / (N - 1)
                        all_images = torch.cat([x_from_dataloader, this_image], dim=0)
                        x_as_image = make_grid(all_images, nrow=4)
                        img = x_as_image.permute(1, 2, 0).cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                        gif.append(Image.fromarray(img))

                    gif[0].save(
                        f"contents/sample_{global_step}.gif",
                        save_all=True,
                        append_images=gif[1:],
                        duration=100,
                        loop=0,
                    )

                    last_img = gif[-1]
                    last_img.save(f"contents/sample_{global_step}_last.png")

                    # log images
                    wandb.log(
                        {
                            "sample": wandb.Image(last_img),
                        }
                    )

                d3pm.train()
