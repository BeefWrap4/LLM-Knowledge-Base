# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.5.2 DDPM 简化实现 - SinusoidalPositionEmbedding + SimpleUNet + DDPM
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch
# run: python 05_ddpm_diffusion.py
# expected_runtime: 5-15s
# expected_output: 前向加噪 + 训练 step + 反向采样
# ---
# See: ../tutorial/21_多模态大模型.md#21-5-2-ddpm-简化实现
# Interview hooks:
#   1. DDPM 训练目标"预测噪声"相比"预测 x0"有什么优势？
#   2. 为什么需要重参数化技巧直接采样 x_t？
#   3. 线性 / cosine 噪声调度对生成质量有什么影响？

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """时间步的正弦位置嵌入"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B]
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class SimpleUNet(nn.Module):
    """简化版 UNet，用于 DDPM 图像生成"""

    def __init__(self, in_channels=3, base_channels=32, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels * 4),
            nn.Linear(base_channels * 4, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # 编码器（下采样）
        self.enc1 = self._make_block(in_channels, base_channels)
        self.enc2 = self._make_block(base_channels, base_channels * 2)
        self.enc3 = self._make_block(base_channels * 2, base_channels * 4)
        # 中间层
        self.mid = self._make_block(base_channels * 4, base_channels * 4)
        # 时间嵌入投影
        self.time_proj = nn.Linear(time_emb_dim, base_channels * 4)
        # 解码器（上采样）
        self.dec3 = self._make_block(base_channels * 8, base_channels * 2)
        self.dec2 = self._make_block(base_channels * 4, base_channels)
        self.dec1 = self._make_block(base_channels * 2, base_channels)
        # 输出层
        self.final = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x, t):
        # x: [B, 3, H, W], t: [B]
        t_emb = self.time_mlp(t)
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        # 注入时间信息
        t_proj = self.time_proj(t_emb)[:, :, None, None]
        mid = self.mid(self.down(e3))
        mid = mid + t_proj * mid
        # 解码器（含跳跃连接）
        d3 = self.dec3(torch.cat([self.up(mid), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.final(d1)


class DDPM:
    """DDPM 训练与采样框架（CPU/GPU 兼容）"""

    def __init__(self, model, n_steps=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.model = model.to(device)
        self.n_steps = n_steps
        self.device = device
        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t):
        """前向加噪：x0 -> xt"""
        alpha_bar_t = self.alpha_bars[t][:, None, None, None]
        epsilon = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
        return xt, epsilon

    def training_step(self, x0):
        """单步训练"""
        B = x0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=self.device)
        xt, noise = self.add_noise(x0, t)
        pred_noise = self.model(xt, t)
        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, shape, n_steps=None):
        """DDPM 采样：从噪声生成图像（采样步数可减少以加速）"""
        n_steps = n_steps or self.n_steps
        # 等间隔子采样
        step_indices = torch.linspace(0, self.n_steps - 1, n_steps).long()
        self.model.eval()
        x = torch.randn(shape, device=self.device)
        for t in reversed(step_indices.tolist()):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t_batch)
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]
            noise = torch.randn_like(x) if t > 0 else 0
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
            ) + torch.sqrt(beta_t) * noise
        return x


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 极简版 UNet + DDPM 演示（n_steps=100 加速）
    unet = SimpleUNet(in_channels=3, base_channels=32, time_emb_dim=128)
    ddpm = DDPM(unet, n_steps=100, device=device)

    # 1) 训练 step
    x0 = torch.randn(2, 3, 32, 32, device=device)
    loss = ddpm.training_step(x0)
    print(f"Training loss: {loss.item():.4f}")
    loss.backward()
    print("Backward OK")

    # 2) 加噪演示
    t_demo = torch.tensor([10, 50], device=device)
    xt, noise = ddpm.add_noise(x0, t_demo)
    print(f"x_t shape: {tuple(xt.shape)}, t={t_demo.tolist()}")

    # 3) 采样（10 步加速演示）
    samples = ddpm.sample((2, 3, 32, 32), n_steps=10)
    print(f"Sampled shape: {tuple(samples.shape)}")


if __name__ == "__main__":
    main()
    print("OK")
