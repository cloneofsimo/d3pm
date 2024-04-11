import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from matplotlib import pyplot as plt

blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
)

blku = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(oc, oc, 2, stride=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
)


class DummyX0Model(nn.Module):

    def __init__(self, n_channel: int, N: int = 16) -> None:
        super(DummyX0Model, self).__init__()
        self.down1 = blk(n_channel, 16)
        self.down2 = blk(16, 32)
        self.down3 = blk(32, 64)
        self.down4 = blk(64, 512)
        self.down5 = blk(512, 512)
        self.up1 = blku(512, 512)
        self.up2 = blku(512 + 512, 64)
        self.up3 = blku(64 + 64, 32)
        self.up4 = blku(32 + 32, 16)
        self.convlast = blk(32, 16)
        self.final = nn.Conv2d(16, N * n_channel, 1, bias=False)

        self.tr1 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tr2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tr3 = nn.TransformerEncoderLayer(d_model=64, nhead=8)

        self.temb_1 = nn.Linear(32, 16)
        self.temb_2 = nn.Linear(32, 32)
        self.temb_3 = nn.Linear(32, 64)
        self.temb_4 = nn.Linear(32, 512)
        self.N = N

    def forward(self, x, t) -> torch.Tensor:
        x = (2 * x.float() / self.N) - 1.0
        t = t.float().reshape(-1, 1) / 1000
        t_as_sin = [torch.sin(t * 3.1415 * 2**i) for i in range(16)]
        t_as_cos = [torch.cos(t * 3.1415 * 2**i) for i in range(16)]
        tx = torch.cat(t_as_sin + t_as_cos, dim=1).to(x.device)

        t_emb_1 = self.temb_1(tx).reshape(x.shape[0], -1, 1, 1)
        t_emb_2 = self.temb_2(tx).reshape(x.shape[0], -1, 1, 1)
        t_emb_3 = self.temb_3(tx).reshape(x.shape[0], -1, 1, 1)
        t_emb_4 = self.temb_4(tx).reshape(x.shape[0], -1, 1, 1)

        x1 = self.down1(x) + t_emb_1
        x2 = self.down2(nn.functional.avg_pool2d(x1, 2)) + t_emb_2
        x3 = self.down3(nn.functional.avg_pool2d(x2, 2)) + t_emb_3
        x4 = self.down4(nn.functional.avg_pool2d(x3, 2)) + t_emb_4
        x5 = self.down5(nn.functional.avg_pool2d(x4, 2))

        x5 = (
            self.tr1(x5.reshape(x5.shape[0], x5.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(x5.shape)
        )

        y = self.up1(x5)

        y = (
            self.tr2(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(y.shape)
        )

        y = self.up2(torch.cat([x4, y], dim=1))

        y = (
            self.tr3(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(y.shape)
        )

        y = self.up3(torch.cat([x3, y], dim=1))

        y = self.up4(torch.cat([x2, y], dim=1))

        y = self.convlast(torch.cat([x1, y], dim=1))
        y = self.final(y)
        # reshape to B, C, H, W, N
        y = (
            y.reshape(y.shape[0], -1, self.N, *x.shape[2:])
            .transpose(2, -1)
            .contiguous()
        )

        return y


def get_logits_from_logistic_pars(loc, log_scale, num_classes=10):
    loc = loc.unsqueeze(-1)
    log_scale = log_scale.unsqueeze(-1)
    inv_scale = (-log_scale + 2.0).exp()

    bin_width = 2.0 / (num_classes - 1)
    bin_centers = torch.linspace(-1.0, 1.0, num_classes).to(loc.device)
    bin_centers = bin_centers.reshape([1] * (len(loc.shape) - 1) + [num_classes])
    bin_centers = bin_centers - loc
    log_cdf_min = -torch.log1p((-inv_scale * (bin_centers - 0.5 * bin_width)).exp())
    log_cdf_plus = -torch.log1p((-inv_scale * (bin_centers + 0.5 * bin_width)).exp())
    logits = log_minus_exp(log_cdf_plus, log_cdf_min)
    return logits


def log_minus_exp(a, b, epsilon=1e-6):
    return a + torch.log1p(-torch.exp(b - a) + epsilon)


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
        self.beta_t = [1 / (self.n_T - t + 1) for t in range(1, self.n_T + 1)]
        self.eps = 1e-8
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
        # fact2 = self._at_onehot(self.q_mats, t-1, )
        # x, a[t-1]
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        qmats2 = self.q_mats[t - 2]  # bs, num_classes, num_classes

        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        # print(f"Fact1Fact2", fact1.shape, fact2.shape)
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        # print(f"out: {out.shape}")
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        # print(t_broadcast.shape)
        bc = torch.where(t_broadcast == 1, x_0_logits, out)
        # print(f"bc: {bc.shape}")
        return bc

    def vb(self, dist1, dist2):
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

    def model_predict(self, x_0, t):
        if self.logit_type == "logit":
            predicted_x0_logits = self.x0_model(x_0, t)
        else:
            loc, log_scale = self.x0_model(x_0, t)
            predicted_x0_logits = get_logits_from_logistic_pars(
                loc, log_scale, self.num_classses
            )
            # for some reason this dont work that well.

        return predicted_x0_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        predicted_x0_logits = self.model_predict(x, t)

        # based on this, we first do vb loss.
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t)
        # print(f"predicted_x0_logits: {predicted_x0_logits.shape}, true_q_posterior_logits: {true_q_posterior_logits.shape}")
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t)

        vb_loss = self.vb(true_q_posterior_logits, pred_q_posterior_logits)

        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        # print(f"predicted_x0_logits: {predicted_x0_logits.shape}, x: {x.shape}")

        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        return vb_loss + ce_loss * self.hybrid_loss_coeff, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def p_sample(self, x, t, noise):

        predicted_x0_logits = self.model_predict(x, t)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t)

        noise = torch.clip(noise, self.eps, 1.0)

        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample

    def sample(self, x=None):
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t, torch.rand((*x.shape, self.num_classses), device=x.device)
            )

        return x


if __name__ == "__main__":

    x = torch.randint(0, 10, (2, 1, 32, 32))
    N = 16
    d3pm = D3PM(DummyX0Model(1, N), 1000, num_classes=N, hybrid_loss_coeff=0.0).cuda()
    print(f"Total Param Count: {sum([p.numel() for p in d3pm.x0_model.parameters()])}")
    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=32)
    # dataloader = DataLoader([dataset[0]] * 50000, batch_size=32, shuffle=True, num_workers=32)
    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=2e-3, betas=(0.95, 0.99))
    d3pm.train()

    n_epoch = 100
    device = "cuda"

    global_step = 0
    for i in range(n_epoch):
        d3pm.train()
        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            # print(x)
            optim.zero_grad()
            x = x.to(device)
            # discritize x to 10 bins
            x = (x * N).long().clamp(0, N - 1)
            loss, info = d3pm(x)

            if loss.item() > 1000 or torch.isnan(loss):
                print(f"loss is too high, skipping, {loss.item()}")
                loss.backward()
                optim.zero_grad()
                continue

            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 0.01)

            with torch.no_grad():
                param_norm = sum([torch.norm(p) for p in d3pm.x0_model.parameters()])
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()
            pbar.set_description(
                f"loss: {loss_ema:.4f}, norm: {norm:.4f}, param_norm: {param_norm:.4f}, vb_loss: {info['vb_loss']:.4f}, ce_loss: {info['ce_loss']:.4f}"
            )
            optim.step()
            global_step += 1

            if global_step % 300 == 1:
                d3pm.eval()

                with torch.no_grad():
                    x = d3pm.sample(torch.randint(0, N, (16, 1, 32, 32)).cuda())
                    x_as_image = make_grid(x.float() / N, nrow=4)
                    plt.figure()
                    plt.imshow(x_as_image.permute(1, 2, 0).cpu().numpy())
                    plt.show()
                    save_image(x_as_image, f"sample_{global_step}.png")

                d3pm.train()
