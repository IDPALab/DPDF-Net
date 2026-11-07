import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from thop import profile

class DAC(nn.Module):
    def __init__(self, channels, r_theta=0.1, m_theta=0.4, e_theta=0.15, temperature=10.0):
        super().__init__()

        self.r_theta = r_theta
        self.m_theta = m_theta
        self.e_theta = e_theta
        self.temp = temperature

        self.k1 = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)
        self.k2 = nn.Conv3d(channels, channels, 5, padding=2, groups=channels)
        self.local_scanner = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, D, H, W = x.shape

        dx = torch.abs(x[..., 1:] - x[..., :-1])
        dy = torch.abs(x[..., 1:, :] - x[..., :-1, :])
        dd = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        grad_map = (F.pad(dx, (0, 1)) + F.pad(dy, (0, 0, 0, 1)) + F.pad(dd, (0, 0, 0, 0, 0, 1))) / 3.0
        grad_map = torch.mean(grad_map, dim=1, keepdim=True)
        m_map = torch.sigmoid(self.temp * (self.r_theta - grad_map))
        local_m_ratio = (self.local_scanner(m_map))
        select_weight = torch.sigmoid(self.temp * (local_m_ratio - self.m_theta))

        feat1 = self.k1(x)
        feat2 = self.k2(x)

        feat_adaptive = (select_weight * feat2 + (1 - select_weight) * feat1)
        diff = torch.pow(feat1 - feat_adaptive, 2)
        diff_dist = torch.sqrt(torch.mean(diff, dim=(1, 2, 3, 4), keepdim=True))
        valid_gate = torch.sigmoid(self.temp * (self.e_theta - diff_dist))
        return (valid_gate * feat_adaptive + (1 - valid_gate) * feat1)

def spike_activation(x, temp=3.0):
    out_s = torch.gt(x, 0.5).float()
    out_bp = torch.clamp(x, 0, 1)
    out_soft = (torch.tanh(temp * (out_bp - 0.5)) + np.tanh(temp * 0.5)) / (2 * np.tanh(temp * 0.5))
    return (out_s - out_soft).detach() + out_soft

class SparseLIF(nn.Module):
    def __init__(self, channels, group_size=4):
        super().__init__()

        self.decay = 0.25
        self.v_th = 1.0
        self.group_size = group_size

    def forward(self, x):
        B, T, C, H, W = x.shape

        g = self.group_size
        orig_T = T

        if T % g != 0:
            pad_t = g - (T % g)
            pad_tensor = torch.zeros(B, pad_t, C, H, W, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_tensor], dim=1)
            T = x.shape[1]

        num_groups = T // g
        x_group = (x.view(B, num_groups, g, C, H, W).mean(dim=2))
        u = torch.zeros_like(x_group[:, 0])

        spikes = []

        for t in range(num_groups):
            u = (u * self.decay + x_group[:, t])
            s = spike_activation(u / self.v_th)
            u = u * (1 - s)
            spikes.append(s)

        out = (torch.stack(spikes, dim=1).unsqueeze(2).repeat(1, 1, g, 1, 1, 1).view(B, T, C, H, W))

        return out[:, :orig_T]


class DRFS(nn.Module):
    def __init__(self, channels, freq_num=32, reduction=4, k_ratio=0.7):
        super(DRFS, self).__init__()

        self.k_ratio = k_ratio
        self.register_buffer('last_mask', torch.zeros(1))
        self.dct_conv = nn.Conv2d(channels, freq_num, kernel_size=3, padding=1, bias=False)
        self.freq_commander = nn.Sequential(

            nn.Conv2d(
                freq_num,
                freq_num // reduction,
                kernel_size=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                freq_num // reduction,
                3,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

        self.feedback_evaluator = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                channels,
                channels // 4,
                1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                channels // 4,
                1,
                1
            ),

            nn.Sigmoid()
        )

        self.continuous_conv = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                groups=channels
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                groups=channels
            ),

            nn.Conv2d(
                channels,
                channels,
                1
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

        self.sparse_lif = SparseLIF(channels)

        self.sparse_conv = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                1
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.ReLU(
                inplace=True
            )
        )


        self.register_buffer('sparse_nonzero_count', torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer('sparse_total_count', torch.tensor(0, dtype=torch.long), persistent=False)
        self.collect_sparsity = False


    def generate_mask(self, g_mask, B, H, W):
        flat_g = g_mask.view(B, -1)

        num_keep = int(H * W * self.k_ratio)

        threshold, _ = torch.kthvalue(flat_g, max(1, (H * W) - num_keep + 1), dim=1, keepdim=True)

        mask_hard = (g_mask >= threshold.view(B, 1, 1, 1)).float()

        return (mask_hard - g_mask).detach() + g_mask


    def reset_sparsity_stats(self):
        self.sparse_nonzero_count.zero_()

        self.sparse_total_count.zero_()


    def enable_sparsity_stats(self):
        self.collect_sparsity = True


    def disable_sparsity_stats(self):
        self.collect_sparsity = False


    def get_actual_sparsity(self):
        total = (self.sparse_total_count.item())

        if total == 0:
            return 0.0

        nonzero = (self.sparse_nonzero_count.item())

        sparsity = (1.0 - nonzero / total)

        return sparsity * 100.0

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)

        B, T, C, H, W = x.shape

        x_mean = torch.mean(x, dim=1)

        freq_map = self.dct_conv(x_mean)

        g_all = self.freq_commander(freq_map)
        g_cont = (g_all[:, 0:1].unsqueeze(1))
        g_sparse = (g_all[:, 1:2].unsqueeze(1))
        g_mask_init = (g_all[:, 2:3])
        mask_init = self.generate_mask(g_mask_init, B, H, W)

        self.last_mask = mask_init


        x_sparse_feat = self.sparse_lif(x * mask_init.unsqueeze(1))

        if (self.collect_sparsity and not self.training):
            with torch.no_grad():
                self.sparse_nonzero_count += (torch.count_nonzero(x_sparse_feat))

                self.sparse_total_count += (x_sparse_feat.numel())

        probing_error = (torch.abs(x - x_sparse_feat).mean(dim=1))

        confidence = (self.feedback_evaluator(probing_error).unsqueeze(1))

        x_cont = rearrange(
            self.continuous_conv(
                rearrange(
                    x,
                    'b t c h w -> (b t) c h w'
                )
            ),
            '(b t) c h w -> b t c h w',
            b=B
        )

        x_sparse = rearrange(
            self.sparse_conv(
                rearrange(
                    x_sparse_feat,
                    'b t c h w -> (b t) c h w'
                )
            ),
            '(b t) c h w -> b t c h w',
            b=B
        )

        refined_g_sparse = (g_sparse * confidence)
        gate_sum = (g_cont + refined_g_sparse + 1e-6)
        out = ((g_cont * x_cont + refined_g_sparse * x_sparse) / gate_sum + x)

        return out.permute(0, 2, 1, 3, 4)


class PCGDA(nn.Module):
    def __init__(self, in_channels, groups=4):
        super(PCGDA, self).__init__()

        mid = in_channels // 2
        self.register_buffer('base_grid_cache', torch.zeros(0), persistent=False)
        self.offset_net = nn.Sequential(

            nn.Conv3d(
                in_channels,
                in_channels // 4,
                kernel_size=1
            ),

            nn.BatchNorm3d(
                in_channels // 4
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv3d(
                in_channels // 4,
                in_channels // 4,
                kernel_size=3,
                padding=1,
                groups=in_channels // 4
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv3d(
                in_channels // 4,
                4,
                kernel_size=1
            )
        )

        self.pos_gate = nn.Sequential(nn.Conv3d(4, mid, kernel_size=1), nn.BatchNorm3d(mid), nn.Sigmoid())
        self.group_spec = nn.Sequential(

            nn.AdaptiveAvgPool3d(1),

            nn.Conv3d(
                in_channels,
                mid,
                kernel_size=1,
                groups=groups
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv3d(
                mid,
                in_channels,
                kernel_size=1,
                groups=groups
            ),

            nn.Sigmoid()
        )

        self.compress = nn.Conv3d(in_channels, mid, kernel_size=1)
        self.spatial = nn.Sequential(

            nn.Conv3d(
                mid,
                mid,
                (1, 3, 3),
                padding=(0, 1, 1),
                groups=mid
            ),

            nn.BatchNorm3d(
                mid
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv3d(
                mid,
                mid,
                (3, 1, 1),
                padding=(1, 0, 0),
                groups=mid
            ),

            nn.BatchNorm3d(
                mid
            ),

            nn.ReLU(
                inplace=True
            )
        )

        self.mapping = nn.Sequential(

            nn.Conv3d(
                in_channels + mid,
                in_channels,
                kernel_size=1
            ),

            nn.BatchNorm3d(
                in_channels
            )
        )

    def build_base_grid(self, b, d, h, w, device):
        grid_h = torch.linspace(-1, 1, h, device=device)

        grid_w = torch.linspace(-1, 1, w, device=device)

        mesh_h, mesh_w = torch.meshgrid(grid_h, grid_w, indexing='ij')

        base = torch.stack(
            [
                torch.sin(
                    math.pi * mesh_h
                ),

                torch.cos(
                    math.pi * mesh_h
                ),

                torch.sin(
                    math.pi * mesh_w
                ),

                torch.cos(
                    math.pi * mesh_w
                )
            ],
            dim=0
        )

        return (base.unsqueeze(0).unsqueeze(2).expand(1, -1, d, -1, -1))

    def forward(self, x):
        b, c, d, h, w = x.shape

        if (self.base_grid_cache.numel() == 0 or self.base_grid_cache.shape[2:] != (d, h, w)):
            self.base_grid_cache = (self.build_base_grid(1, d, h, w, x.device))

        base_loc = (self.base_grid_cache.expand(b, -1, -1, -1, -1))

        offset = (torch.tanh(self.offset_net(x)) * 0.1)

        loc = torch.clamp(base_loc + offset, -1.5, 1.5)

        p = (self.pos_gate(loc) * self.group_spec(x).mean(1, keepdim=True))

        out = self.mapping(torch.cat([x, self.spatial(self.compress(x) * p)], dim=1))

        return F.relu(out + x)


class MD_DSCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        ic = out_channels // 2

        self.p = nn.Sequential(

            nn.Conv3d(
                in_channels * 2,
                ic,
                3,
                padding=1
            ),

            nn.BatchNorm3d(
                ic
            ),

            nn.ReLU(
                inplace=True
            )
        )

        self.sim_conv = DAC(ic)

        self.att = nn.Sequential(

            nn.AdaptiveAvgPool3d(1),

            nn.Conv3d(
                out_channels,
                out_channels // 2,
                1
            ),

            nn.GELU(),

            nn.Conv3d(
                out_channels // 2,
                out_channels,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):
        diff = F.pad(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], (0, 0, 0, 0, 0, 1))

        x_cat = torch.cat([x, diff], 1)
        p_feat = self.p(x_cat)
        c_feat = self.sim_conv(p_feat)
        feat = torch.cat([p_feat, c_feat], 1)
        return (feat * self.att(feat))


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, k_ratio=0.7):
        super().__init__()

        self.c1 = nn.Sequential(nn.Conv3d(inplanes, planes, 1), nn.BatchNorm3d(planes), nn.ReLU(True))
        self.c2 = DRFS(planes, freq_num=planes, k_ratio=k_ratio)
        self.c3 = nn.Sequential(nn.Conv3d(planes, planes * 4, 1), nn.BatchNorm3d(planes * 4))
        self.ds = nn.Sequential(nn.Conv3d(inplanes, planes * 4, 1), nn.BatchNorm3d(planes * 4))

    def forward(self, x):
        return F.relu(self.c3(self.c2(self.c1(x))) + self.ds(x))

class DPDF_Net(nn.Module):
    def __init__(self, num_classes=11, k_ratio=0.7):
        super().__init__()

        self.md_dscm = MD_DSCM(1, 64)
        self.pcgda = PCGDA(64)
        self.layer = Bottleneck(64, 64, k_ratio=k_ratio)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer(self.pcgda(self.md_dscm(x)))
        x = F.adaptive_avg_pool3d(x, 1)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        return self.fc(x)


    def reset_sparsity_stats(self):
        self.layer.c2.reset_sparsity_stats()

    def enable_sparsity_stats(self):
        self.layer.c2.enable_sparsity_stats()

    def disable_sparsity_stats(self):
        self.layer.c2.disable_sparsity_stats()

    def get_actual_sparsity(self):
        return self.layer.c2.get_actual_sparsity()

# =============================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DPDF_Net(num_classes=11, k_ratio=0.7).to(device)
    model.eval()
    input_data = torch.randn([16, 1, 32, 7, 7]).to(device)
    model.reset_sparsity_stats()
    model.disable_sparsity_stats()

    with torch.no_grad():

        f_flops, f_params = profile(model.md_dscm, inputs=(input_data,), verbose=False)
        m1_out = model.md_dscm(input_data)

        d_flops, d_params = profile(model.pcgda, inputs=(m1_out,), verbose=False)
        m2_out = model.pcgda(m1_out)

        scsa_in = model.layer.c1(m2_out)
        s_flops, s_params = profile(model.layer.c2, inputs=(scsa_in,), verbose=False)

        t_flops, t_params = profile(model, inputs=(input_data,), verbose=False)

    model.reset_sparsity_stats()
    model.enable_sparsity_stats()

    with torch.no_grad():
        _ = model(input_data)

    model.disable_sparsity_stats()
    actual_sparsity = (model.get_actual_sparsity())

    print("\n" + "=" * 75)
    print(f"{'DPDF-Net':^70}")
    print("=" * 75)
    print(f"{'Module Name':<30} | " f"{'Params(M)':<15} | " f"{'FLOPs(M)':<15}")
    print("-" * 75)
    print(f"{'1. MD-DSCM':<30} | " f"{f_params / 1e6:<15.4f} | " f"{f_flops / 1e6:<15.4f}")
    print(f"{'2. PCGDA':<30} | " f"{d_params / 1e6:<15.4f} | " f"{d_flops / 1e6:<15.4f}")
    print(f"{'3. DRFS':<30} | " f"{s_params / 1e6:<15.4f} | " f"{s_flops / 1e6:<15.4f}")
    print("-" * 75)
    print(f"{'TOTAL MODEL (System)':<30} | " f"{t_params / 1e6:<15.4f} | " f"{t_flops / 1e6:<15.4f}")
    print("=" * 75)
    print(f"Target Keep Ratio : " f"{model.layer.c2.k_ratio:.2f}")
    print(f"Actual Sparsity   : " f"{actual_sparsity:.2f}%")
    print("=" * 75)
