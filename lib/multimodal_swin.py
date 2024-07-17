
import torch.nn.functional as F
from .backbone import MultiModalSwinTransformer
import torch.nn as nn
import numpy as np
import torch
import torch.utils.checkpoint as checkpoint

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


class MultiModalSwin(MultiModalSwinTransformer):
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.0
                 ):
        super().__init__(pretrain_img_size=pretrain_img_size,
                         patch_size=patch_size,
                         in_chans=in_chans,
                         embed_dim=embed_dim,
                         depths=depths,
                         num_heads=num_heads,
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                         qk_scale=qk_scale,
                         drop_rate=drop_rate,
                         attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate,
                         norm_layer=norm_layer,
                         ape=ape,
                         patch_norm=patch_norm,
                         out_indices=out_indices,
                         frozen_stages=frozen_stages,
                         use_checkpoint=use_checkpoint,
                         num_heads_fusion=num_heads_fusion,
                         fusion_drop=fusion_drop
                         )
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.use_checkpoint = False

    def forward_stem(self, x):
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        return x, Wh, Ww

    def forward_stage1(self, x, H, W):

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.layers[0].blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)  # output of a Block has shape (B, H*W, dim)

        return x


    def forward_stage2(self, x, H, W):

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.layers[1].blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)  # output of a Block has shape (B, H*W, dim)
        return x

    def forward_stage3(self, x, H, W):

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.layers[2].blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)  # output of a Block has shape (B, H*W, dim)
        return x

    def forward_stage4(self, x, H, W):

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.layers[3].blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)  # output of a Block has shape (B, H*W, dim)
        return x

    def forward_pwam1(self, x, H, W, l, l_mask):

        out = []

        x_reshape = x.permute(0,2,1).view(x.shape[0], x.shape[2], H, W)
        x_size = x_reshape.size()
        for i, p in enumerate(self.layers[0].psizes):
            px = self.layers[0].pyramids[i](x_reshape)
            px = px.flatten(2).permute(0,2,1)
            px_residual = self.layers[0].fusions[i](px, l, l_mask)
            px_residual = px_residual.permute(0,2,1).view(x.shape[0], self.layers[0].reduction_dim , p, p)

            out.append(F.interpolate(px_residual, x_size[2:], mode='bilinear', align_corners=True).flatten(2).permute(0,2,1))

        x_residual = self.layers[0].fusion(x, l, l_mask)
        out.append(x_residual)
        x = x + (self.layers[0].res_gate(x_residual) * x_residual)



        x_residual = self.layers[0].mixer(torch.cat(out, dim =2))
        x_residual = x_residual.view(-1, H, W, self.num_features[0]).permute(0, 3, 1, 2).contiguous()

        if self.layers[0].downsample is not None:
            x_down = self.layers[0].downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_residual, H, W, x_down, Wh, Ww
        else:
            return x_residual, H, W, x, H, W

    def forward_pwam2(self, x, H, W, l, l_mask):
        out = []

        x_reshape = x.permute(0,2,1).view(x.shape[0], x.shape[2], H, W)
        x_size = x_reshape.size()
        for i, p in enumerate(self.layers[1].psizes):
            px = self.layers[1].pyramids[i](x_reshape)
            px = px.flatten(2).permute(0,2,1)
            px_residual = self.layers[1].fusions[i](px, l, l_mask)
            px_residual = px_residual.permute(0,2,1).view(x.shape[0], self.layers[1].reduction_dim , p, p)
            out.append(F.interpolate(px_residual, x_size[2:], mode='bilinear', align_corners=True).flatten(2).permute(0,2,1))

        x_residual = self.layers[1].fusion(x, l, l_mask)
        out.append(x_residual)
        x = x + (self.layers[1].res_gate(x_residual) * x_residual)



        x_residual = self.layers[1].mixer(torch.cat(out, dim =2))
        x_residual = x_residual.view(-1, H, W, self.num_features[1]).permute(0, 3, 1, 2).contiguous()
        if self.layers[1].downsample is not None:
            x_down = self.layers[1].downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_residual, H, W, x_down, Wh, Ww
        else:
            return x_residual, H, W, x, H, W

    def forward_pwam3(self, x, H, W, l, l_mask):
        out = []

        x_reshape = x.permute(0,2,1).view(x.shape[0], x.shape[2], H, W)
        x_size = x_reshape.size()
        for i, p in enumerate(self.layers[2].psizes):
            px = self.layers[2].pyramids[i](x_reshape)
            px = px.flatten(2).permute(0,2,1)
            px_residual = self.layers[2].fusions[i](px, l, l_mask)
            px_residual = px_residual.permute(0,2,1).view(x.shape[0], self.layers[2].reduction_dim , p, p)
            out.append(F.interpolate(px_residual, x_size[2:], mode='bilinear', align_corners=True).flatten(2).permute(0,2,1))

        x_residual = self.layers[2].fusion(x, l, l_mask)
        out.append(x_residual)
        x = x + (self.layers[2].res_gate(x_residual) * x_residual)


        x_residual = self.layers[2].mixer(torch.cat(out, dim =2))
        x_residual = x_residual.view(-1, H, W, self.num_features[2]).permute(0, 3, 1, 2).contiguous()
        if self.layers[2].downsample is not None:
            x_down = self.layers[2].downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_residual, H, W, x_down, Wh, Ww
        else:
            return x_residual, H, W, x, H, W

    def forward_pwam4(self, x, H, W, l, l_mask):
        out = []

        x_reshape = x.permute(0,2,1).view(x.shape[0], x.shape[2], H, W)
        x_size = x_reshape.size()
        for i, p in enumerate(self.layers[3].psizes):
            px = self.layers[3].pyramids[i](x_reshape)
            px = px.flatten(2).permute(0,2,1)
            px_residual = self.layers[3].fusions[i](px, l, l_mask)
            px_residual = px_residual.permute(0,2,1).view(x.shape[0], self.layers[3].reduction_dim , p, p)
            out.append(F.interpolate(px_residual, x_size[2:], mode='bilinear', align_corners=True).flatten(2).permute(0,2,1))


        x_residual = self.layers[3].fusion(x, l, l_mask)
        out.append(x_residual)
        x = x + (self.layers[3].res_gate(x_residual) * x_residual)


        x_residual = self.layers[3].mixer(torch.cat(out, dim =2))
        x_residual = x_residual.view(-1, H, W, self.num_features[3]).permute(0, 3, 1, 2).contiguous()
        if self.layers[3].downsample is not None:
            x_down = self.layers[3].downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_residual, H, W, x_down, Wh, Ww
        else:
            return x_residual, H, W, x, H, W
