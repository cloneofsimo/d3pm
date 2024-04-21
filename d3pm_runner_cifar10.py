import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb


class D3PM(nn.Module):
    def __init__(
        self,
        x0_model: nn.Module,
        n_T: int,
        num_classes: int = 10,
        forward_type="uniform",
        hybrid_loss_coeff=0.001,
    ) -> None:
        super(D3PM, self).__init__()
        self.x0_model = x0_model

        self.n_T = n_T
        self.hybrid_loss_coeff = hybrid_loss_coeff

        steps = torch.arange(n_T + 1, dtype=torch.float64) / n_T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999)
        
        #self.beta_t = [1 / (self.n_T - t + 1) for t in range(1, self.n_T + 1)]
        self.eps = 1e-6
        self.num_classses = num_classes
        q_onestep_mats = []
        q_mats = []  # these are cumulative

        for beta in self.beta_t:

            if forward_type == "uniform":
                mat = torch.ones(num_classes, num_classes) * beta / num_classes
                mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
                q_onestep_mats.append(mat)
            else:
                raise NotImplementedError
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)

        q_one_step_transposed = q_one_step_mats.transpose(
            1, 2
        )  # this will be used for q_posterior_logits

        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.logit_type = "logit"

        # register
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

        assert self.q_mats.shape == (
            self.n_T,
            num_classes,
            num_classes,
        ), self.q_mats.shape

    def _at(self, a, t, x):
        # t is 1-d, x is integer value of 0 to num_classes - 1
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        return a[t - 1, x, :]

    def q_posterior_logits(self, x_0, x_t, t):
        # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.
        # otherwise, we return the L_{t-1} loss.
        # Also, we never have t == 0.

        # if x_0 is integer, we convert it to one-hot.
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classses) + self.eps
            )
        else:
            x_0_logits = x_0.clone()

        assert x_0_logits.shape == x_t.shape + (self.num_classses,), print(
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
        )

        # Here, we caclulate equation (3) of the paper. Note that the x_0 Q_t x_t^T is a normalizing constant, so we don't deal with that.

        # fact1 is "guess of x_{t-1}" from x_t
        # fact2 is "guess of x_{t-1}" from x_0

        fact1 = self._at(self.q_one_step_transposed, t, x_t)

        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        qmats2 = self.q_mats[t - 2]  # bs, num_classes, num_classes
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))

        bc = torch.where(t_broadcast == 1, x_0_logits, out)

        return bc

    def vb(self, dist1, dist2):

        # flatten dist1 and dist2
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)

        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1)
            - torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        return out.sum(dim=-1).mean()

    def q_sample(self, x_0, t, noise):
        # forward process, x_0 is the clean input.
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def model_predict(self, x_0, t, cond):
        # this part exists because in general, manipulation of logits from model's logit
        # so they are in form of x_0's logit might be independent to model choice.
        # for example, you can convert 2 * N channel output of model output to logit via get_logits_from_logistic_pars
        # they introduce at appendix A.8.
        predicted_x0_logits = self.x0_model(x_0, t, cond)

        return predicted_x0_logits

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t from x_0, and tries to guess x_0 value from x_t using x0_model.
        x is one-hot of dim (bs, ...), with int values of 0 to num_classes - 1
        """
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        x_t = self.q_sample(
            x, t, torch.rand((*x.shape, self.num_classses), device=x.device)
        )
        # x_t is same shape as x
        assert x_t.shape == x.shape, print(
            f"x_t.shape: {x_t.shape}, x.shape: {x.shape}"
        )
        # we use hybrid loss.

        predicted_x0_logits = self.model_predict(x_t, t, cond)

        # based on this, we first do vb loss.
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t)

        vb_loss = self.vb(true_q_posterior_logits, pred_q_posterior_logits)

        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)

        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        return self.hybrid_loss_coeff * vb_loss +  ce_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def p_sample(self, x, t, cond, noise):

        predicted_x0_logits = self.model_predict(x, t, cond)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t)

        noise = torch.clip(noise, self.eps, 1.0)

        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))
    
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample

    def sample(self, x, cond):
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device)
            )

        return x

    def sample_with_image_sequence(self, x, cond, stride=10):
        steps = 0
        images = []
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device)
            )
            steps += 1
            if steps % stride == 0:
                images.append(x)

        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images

from dit import DiT_Llama

if __name__ == "__main__":

    wandb.init(project="d3pm_cifar10")

    N = 8  # number of classes for discretized state per pixel
    d3pm = D3PM(DiT_Llama(3, N, dim = 1024), 1000, num_classes=N, hybrid_loss_coeff=0.0).cuda()
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
                        x_from_dataloader = x_cat[:16].cpu() / (N-1)
                        this_image = image.float().cpu() / (N-1)
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