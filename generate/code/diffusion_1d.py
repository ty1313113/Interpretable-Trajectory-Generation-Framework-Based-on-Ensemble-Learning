# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, utils
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import numpy as np
# from torch.nn import functional as F

# # 1D Diffusion模型核心组件
# def conv_1d(*args, **kwargs):
#     return nn.Conv1d(*args, **kwargs)

# def zero_module(module):
#     for p in module.parameters():
#         p.detach().zero_()
#     return module

# class GroupNorm32(nn.GroupNorm):
#     def forward(self, x):
#         return super().forward(x.float()).type(x.dtype)

# def normalization(channels):
#     return GroupNorm32(32, channels)

# def timestep_embedding(timesteps, dim, max_period=10000):
#     half = dim // 2
#     freqs = torch.exp(
#         -torch.log(torch.tensor(max_period, dtype=torch.float32))
#         * torch.arange(start=0, end=half, dtype=torch.float32)
#         / half
#     ).to(device=timesteps.device)
#     args = timesteps[:, None].float() * freqs[None]
#     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#     if dim % 2:
#         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#     return embedding

# # 基础模块
# class TimestepBlock(nn.Module):
#     def forward(self, x, emb):
#         raise NotImplementedError

# class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
#     def forward(self, x, emb):
#         for layer in self:
#             if isinstance(layer, TimestepBlock):
#                 x = layer(x, emb)
#             else:
#                 x = layer(x)
#         return x

# class AttentionBlock(nn.Module):
#     def __init__(self, channels, num_heads=4):
#         super().__init__()
#         self.channels = channels
#         self.num_heads = num_heads
#         assert channels % num_heads == 0, "channels must be divisible by num_heads"
#         self.head_dim = channels // num_heads
        
#         self.norm = normalization(channels)
#         self.qkv = conv_1d(channels, channels * 3, 1)
#         self.proj_out = zero_module(conv_1d(channels, channels, 1))

#     def forward(self, x):
#         B, C, L = x.shape
#         x_in = x
        
#         # Normalize and project to qkv
#         x = self.norm(x)
#         qkv = self.qkv(x).chunk(3, dim=1)
#         q, k, v = [t.view(B, self.num_heads, self.head_dim, L) for t in qkv]
        
#         # Transpose for attention scores
#         q = q.transpose(-2, -1)  # [B, heads, L, head_dim]
#         k = k.transpose(-2, -1)  # [B, heads, L, head_dim]
#         v = v.transpose(-2, -1)  # [B, heads, L, head_dim]
        
#         # Compute attention scores
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         attn = torch.softmax(attn_scores, dim=-1)
        
#         # Apply attention to values
#         h = torch.matmul(attn, v)  # [B, heads, L, head_dim]
        
#         # Transpose back and reshape
#         h = h.transpose(-2, -1).reshape(B, C, L)
#         h = self.proj_out(h)
#         return x_in + h

# class ResBlock(TimestepBlock):
#     def __init__(self, channels, emb_channels, dropout, out_channels=None, use_scale_shift_norm=False):
#         super().__init__()
#         self.channels = channels
#         self.emb_channels = emb_channels
#         self.dropout = dropout
#         self.out_channels = out_channels or channels
#         self.use_scale_shift_norm = use_scale_shift_norm

#         self.in_layers = nn.Sequential(
#             normalization(channels),
#             nn.SiLU(),
#             conv_1d(channels, self.out_channels, 3, padding=1),
#         )
        
#         self.emb_layers = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
#         )
        
#         self.out_layers = nn.Sequential(
#             normalization(self.out_channels),
#             nn.SiLU(),
#             nn.Dropout(p=dropout),
#             zero_module(conv_1d(self.out_channels, self.out_channels, 3, padding=1)),
#         )

#         if self.out_channels == channels:
#             self.skip_connection = nn.Identity()
#         else:
#             self.skip_connection = conv_1d(channels, self.out_channels, 1)

#     def forward(self, x, emb):
#         h = self.in_layers(x)
#         emb_out = self.emb_layers(emb).type(h.dtype)
#         while len(emb_out.shape) < len(h.shape):
#             emb_out = emb_out[..., None]
        
#         if self.use_scale_shift_norm:
#             scale, shift = torch.chunk(emb_out, 2, dim=1)
#             h = h * (1 + scale) + shift
#             h = self.out_layers(h)
#         else:
#             h = h + emb_out
#             h = self.out_layers(h)
        
#         return self.skip_connection(x) + h

# class Downsample(nn.Module):
#     def __init__(self, channels, use_conv):
#         super().__init__()
#         self.channels = channels
#         self.use_conv = use_conv
#         if use_conv:
#             self.op = conv_1d(channels, channels, 3, stride=2, padding=1)
#         else:
#             self.op = nn.AvgPool1d(kernel_size=2, stride=2)

#     def forward(self, x):
#         return self.op(x)

# class Upsample(nn.Module):
#     def __init__(self, channels, use_conv):
#         super().__init__()
#         self.channels = channels
#         self.use_conv = use_conv
#         if use_conv:
#             self.conv = conv_1d(channels, channels, 3, padding=1)

#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=2, mode="nearest")
#         if self.use_conv:
#             x = self.conv(x)
#         return x

# # 1D UNet模型
# class UNet1D(nn.Module):
#     def __init__(
#         self,
#         seq_length=784,        # MNIST展平后长度
#         in_channels=1,         # 输入通道
#         model_channels=64,      # 模型基础通道
#         out_channels=1,         # 输出通道
#         num_res_blocks=2,       # 残差块数量
#         attention_resolutions=(392, 196, 98),  # 注意力分辨率(序列长度)
#         dropout=0.0,
#         channel_mult=(1, 2, 4, 8),  # 通道倍增因子
#         conv_resample=True,     # 使用卷积重采样
#         num_heads=4,            # 注意力头数
#         use_scale_shift_norm=False,
#     ):
#         super().__init__()
#         self.seq_length = seq_length
#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         self.out_channels = out_channels
#         self.num_heads = num_heads
        
#         # 时间步嵌入
#         time_embed_dim = model_channels * 4
#         self.time_embed = nn.Sequential(
#             nn.Linear(model_channels, time_embed_dim),
#             nn.SiLU(),
#             nn.Linear(time_embed_dim, time_embed_dim),
#         )
        
#         # 输入层
#         self.input_blocks = nn.ModuleList([
#             TimestepEmbedSequential(conv_1d(in_channels, model_channels, 3, padding=1))
#         ])
        
#         # 当前序列长度和下采样因子
#         current_length = seq_length
#         input_block_chans = [model_channels]
#         ch = model_channels
#         ds = 1  # 下采样因子
        
#         # 下采样路径
#         for level, mult in enumerate(channel_mult):
#             for _ in range(num_res_blocks):
#                 layers = [
#                     ResBlock(
#                         ch,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=mult * model_channels,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = mult * model_channels
#                 # 在指定分辨率添加注意力层
#                 if current_length in attention_resolutions:
#                     layers.append(AttentionBlock(ch, num_heads=num_heads))
#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 input_block_chans.append(ch)
            
#             # 非最后一层添加下采样
#             if level != len(channel_mult) - 1:
#                 self.input_blocks.append(
#                     TimestepEmbedSequential(Downsample(ch, conv_resample))
#                 )
#                 current_length //= 2
#                 ds *= 2
#                 input_block_chans.append(ch)
        
#         # 中间层
#         self.middle_block = TimestepEmbedSequential(
#             ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
#             AttentionBlock(ch, num_heads=num_heads),
#             ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
#         )
        
#         # 上采样路径
#         self.output_blocks = nn.ModuleList([])
#         for level, mult in list(enumerate(channel_mult))[::-1]:
#             for i in range(num_res_blocks + 1):
#                 ich = input_block_chans.pop()
#                 layers = [
#                     ResBlock(
#                         ch + ich,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=model_channels * mult,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = model_channels * mult
#                 # 在指定分辨率添加注意力层
#                 if current_length in attention_resolutions:
#                     layers.append(AttentionBlock(ch, num_heads=num_heads))
                
#                 # 添加上采样层
#                 if level != 0 and i == num_res_blocks:
#                     layers.append(Upsample(ch, conv_resample))
#                     current_length *= 2
#                     ds //= 2
                
#                 self.output_blocks.append(TimestepEmbedSequential(*layers))
        
#         # 输出层
#         self.out = nn.Sequential(
#             normalization(ch),
#             nn.SiLU(),
#             zero_module(conv_1d(model_channels, out_channels, 3, padding=1)),
#         )
    
#     def forward(self, x, timesteps):
#         # check_devices(x=x, t=timesteps, model_param=next(self.parameters()))
#         # 时间步嵌入
#         t_emb = timestep_embedding(timesteps, self.model_channels)
#         emb = self.time_embed(t_emb)
        
#         # 保存跳跃连接
#         hs = []
#         h = x
        
#         # 下采样路径
#         for module in self.input_blocks:
#             h = module(h, emb)
#             hs.append(h)
        
#         # 中间层
#         h = self.middle_block(h, emb)
        
#         # 上采样路径
#         for module in self.output_blocks:
#             cat_in = torch.cat([h, hs.pop()], dim=1)
#             h = module(cat_in, emb)
        
#         return self.out(h)

# # Diffusion模型
# class Diffusion1D:
#     def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
#         self.model = model
#         self.timesteps = timesteps
#         self.beta = torch.linspace(beta_start, beta_end, timesteps).to(next(model.parameters()).device)
#         self.alpha = 1. - self.beta
#         self.alpha_bar = torch.cumprod(self.alpha, dim=0)

#     def q_sample(self, x_start, t, noise=None):
#         # 前向扩散过程，添加噪声
#         if noise is None:
#             noise = torch.randn_like(x_start)
#         # 1D: [B, 1, L]
#         sqrt_alpha_bar = self.alpha_bar[t].sqrt().view(-1, 1, 1)
#         sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1)
#         return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

#     def p_sample(self, x, t):
#         # t: int or tensor
#         if isinstance(t, int):
#             t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
#         else:
#             t_tensor = t.to(x.device)
#         pred_noise = self.model(x, t_tensor)
#         beta_t = self.beta[t_tensor].view(-1, 1, 1)
#         alpha_t = self.alpha[t_tensor].view(-1, 1, 1)
#         alpha_bar_t = self.alpha_bar[t_tensor].view(-1, 1, 1)
#         sqrt_one_minus_alpha_bar = (1 - alpha_bar_t).sqrt()
#         coef1 = 1 / alpha_t.sqrt()
#         coef2 = beta_t / sqrt_one_minus_alpha_bar
#         mean = coef1 * (x - coef2 * pred_noise)
#         # 只判断第一个t是否大于0（全batch一致）
#         if (t_tensor[0] > 0):
#             noise = torch.randn_like(x)
#             sigma = beta_t.sqrt()
#             return mean + sigma * noise
#         else:
#             return mean

#     @torch.no_grad()
#     def sample(self, shape):
#         # 从高斯噪声生成1D序列
#         x = torch.randn(shape, device=next(self.model.parameters()).device)
#         for t in reversed(range(self.timesteps)):
#             x = self.p_sample(x, t)
#         return x

# class TTensordataset(torch.utils.data.Dataset):
#     def __init__(self, pt_path):
#         self.data = torch.load(pt_path)  # [N, L]
#         L = self.data.shape[1]
#         pad_len = (8 - (L % 8)) % 8
#         if pad_len > 0:
#             # 在最后一维pad
#             self.data = torch.nn.functional.pad(self.data, (0, pad_len), mode='constant', value=0)
#     def __len__(self):
#         return self.data.shape[0]
#     def __getitem__(self, idx):
#         return self.data[idx].unsqueeze(0).float(), 0

# # 数据处理
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: (x * 2) - 1),  # [-1, 1]
#     transforms.Lambda(lambda x: x.view(1, SEQ_LENGTH))  # 展平为1D序列
# ])


# # 训练配置
# BATCH_SIZE = 64

# # 数据集
# train_dataset = TTensordataset('t_tensor.pt')
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# EPOCHS = 500
# LR = 2e-4
# SEQ_LENGTH = train_dataset.data.shape[1]
# DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"


# # 模型初始化
# model = UNet1D(
#     seq_length=SEQ_LENGTH,
#     in_channels=1,
#     model_channels=64,
#     out_channels=1,
#     num_res_blocks=2,
#     attention_resolutions=[392, 196, 98],  # 序列长度/2, /4, /8
#     dropout=0.1,
#     channel_mult=[1, 2, 4, 8],
#     num_heads=4
# ).to(DEVICE)

# optimizer = optim.Adam(model.parameters(), lr=LR)
# diffusion = Diffusion1D(model)

# # 训练循环
# def save_1d_images(tensor, path, nrow=8):
#     # 将1D序列重塑为2D图像
#     images = tensor.view(-1, 1, 28, 28)
#     grid = utils.make_grid(images, nrow=nrow, normalize=True)
#     utils.save_image(grid, path)

# def check_devices(**tensors):
#     devices = {name: t.device for name, t in tensors.items() if hasattr(t, 'device')}
#     print("张量设备分布：", devices)
#     dev_set = set(devices.values())
#     if len(dev_set) > 1:
#         print("警告：以下张量不在同一设备上！")
#         for name, dev in devices.items():
#             print(f"{name}: {dev}")
#     else:
#         print("所有张量在同一设备上。")

# def log_shapes(model, input_shape, log_path="model_io.log"):
#     log = []
#     def hook(module, inp, out):
#         log.append(f"{module.__class__.__name__}: input {tuple(inp[0].shape)}, output {tuple(out.shape)}")
#     hooks = []
#     for layer in model.modules():
#         if not isinstance(layer, torch.nn.Sequential) and not isinstance(layer, torch.nn.ModuleList) and layer != model:
#             hooks.append(layer.register_forward_hook(hook))
#     dummy_x = torch.randn(*input_shape)
#     # 构造timesteps，假设batch维为input_shape[0]
#     dummy_t = torch.zeros(input_shape[0], dtype=torch.long)
#     model.eval()
#     with torch.no_grad():
#         model(dummy_x, dummy_t)
#     with open(log_path, "w") as f:
#         for line in log:
#             f.write(line + "\n")
#     for h in hooks:
#         h.remove()

# # 用法
# # log_shapes(model, (1, 1, 784))
# if __name__ == "__main__":
#     os.makedirs("results/t_tensor/sample", exist_ok=True)
#     os.makedirs("results/t_tensor/model_dict", exist_ok=True)

#     for epoch in range(1, EPOCHS + 1):
#         model.train()
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
#         for batch in pbar:
#             x, _ = batch
#             x = x.to(DEVICE)
            
#             # 扩散过程
#             t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=DEVICE)
#             noise = torch.randn_like(x)
#             x_noisy = diffusion.q_sample(x, t, noise)
            
#             # 预测噪声
#             pred_noise = model(x_noisy, t)
#             loss = nn.MSELoss()(pred_noise, noise)
            
#             # 优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             pbar.set_postfix(loss=loss.item())
        
#         # 每1轮保存样本
#         if epoch % 10 == 0:
#             model.eval()
#             with torch.no_grad():
#                 sample = diffusion.sample((16, 1, SEQ_LENGTH))
#                 # 保存到pt文件
#                 torch.save(sample, f"results/t_tensor/sample/sample_epoch_{epoch}.pt")
#                 torch.save(model.state_dict(), f"results/t_tensor/model_dict/model_epoch_{epoch}.pt")
    
#     print("训练完成!")
import torch
for n in range(10, 210, 10):
    # 读取t_tensor.pt文件
    num = n
    t_tensor_tosee=torch.load(f'results/t_tensor/sample/sample_epoch_{num}.pt')
    t_tensor_tosee = t_tensor_tosee.squeeze(1)  # 去掉通道维度
    # t_tensor_tosee = torch.load('t_tensor.pt')

    # 放到cpu并重新保存到当前路径
    t_tensor_tosee=t_tensor_tosee.cpu()
    print(t_tensor_tosee.shape)
    t_tensor_tosee = (t_tensor_tosee > 0.5).float()  # 将tensor二值化
    # torch.save(t_tensor_tosee, 't_tensor_30.pt')

    import matplotlib.pyplot as plt

    # 假设 x_gen_unpad 是 [B, L]，修改下面的代码，改为选择前16条sample的轨迹可视化，在一个4 * 4的方格里，修改现有的代码，目前显示为空！！
    num_samples = 16
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    print(axes.shape)
    for i in range(num_samples):
        axes[i].plot(t_tensor_tosee[i].cpu().numpy())
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
        # 打印每一个tensor的非零元素个数
        non_zero_count = (t_tensor_tosee[i] > 0).sum().item()
        axes[i].text(0.5, 0.9, f'Non-zero: {non_zero_count}', ha='center', va='center', transform=axes[i].transAxes)
    # 设置坐标轴标签
    for ax in axes:
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Amplitude')
    # 设置标题
    plt.suptitle('Generated 1D Samples', fontsize=16)
    # 调整子图间距
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
    # 保存图片
    plt.savefig(f't_tensor_{num}.png', bbox_inches='tight')
    # 显示图片
    plt.suptitle('Generated 1D Samples', fontsize=16)
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
    for ax in axes[num_samples:]:
        ax.axis('off')  # 隐藏多余的子图
    plt.show()
