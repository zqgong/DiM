# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from mamba_ssm.modules.mamba_simple import Mamba
from functools import partial
from einops import rearrange


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                 Mamba Block                                   #
#################################################################################

class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.weight * x_normed).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g


class LLaMAMLP(nn.Module):
    def __init__(self, n_embd, intermediate_size, bias=True) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.fc_2 = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.proj = nn.Linear(intermediate_size, n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class HourGlass(nn.Module):
    def __init__(self, hidden_size, intermediate_size, out_features, down_scale=True, up_scale=True):
        super().__init__()
        self.down_scale = down_scale
        self.up_scale = up_scale

        if down_scale:
            self.down = nn.AvgPool1d(2, stride=2)
        
        if up_scale:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')

        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=intermediate_size, out_features=out_features, act_layer=approx_gelu, drop=0)
        self.mlp = nn.Sequential(
                nn.Linear(hidden_size, out_features, bias=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_scale:
            x = rearrange(x, "b l d -> b d l")
            x = self.down(x)
            x = rearrange(x, "b d l -> b l d")

        x = self.mlp(x)

        if self.up_scale:
            x = rearrange(x, "b l d -> b d l")
            x = self.up(x)
            x = rearrange(x, "b d l -> b l d")
        return x


class MambaBlock(nn.Module):
    def __init__(
        self, hidden_size, layer_idx, bidirectional, mlp_ratio=4.0, **block_kwargs 
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()

        self.bidirectional = bidirectional

        factory_kwargs = {"device": None, "dtype": None}
        self.mixer1 = partial(Mamba, layer_idx=layer_idx * 2, **factory_kwargs)(hidden_size)
        self.mixer2 = partial(Mamba, layer_idx=layer_idx * 2 + 1,  **factory_kwargs)(hidden_size)

        if self.bidirectional == "v1":
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            self.dense1 = HourGlass(hidden_size, mlp_hidden_dim, hidden_size, False, False) # True, True // down and up
            self.dense2 = HourGlass(hidden_size * 2, mlp_hidden_dim, hidden_size * 2, False, False)  # True, False // down 
            self.dense3 = HourGlass(hidden_size, mlp_hidden_dim, hidden_size * 2, False, False) # True, False // down
            self.dense4 = HourGlass(hidden_size * 2, mlp_hidden_dim, hidden_size, False, False) # False, True // up

            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 3 * hidden_size, bias=True)
            )
            
        
        elif self.bidirectional == "v2":
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else: 
            raise NotImplementedError("not implemented")

    def forward(
        self, x, c, inference_params=None
    ):

        if self.bidirectional == "v1":
            shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=1)

            x_ = modulate(self.norm1(x), shift, scale)

            x_ssm = self.dense1(x_)
            x_f = self.mixer1(x_ssm, inference_params=inference_params)
            x_b = self.mixer2(x_ssm.flip([1]), inference_params=inference_params)
            x_fb = torch.cat([x_f, x_b.flip([1])], dim=-1) 
            x_fb = self.dense2(x_fb)

            x_ = self.dense3(x_)
            x_ = x_ * x_fb
            x_o = self.dense4(x_)

            x = x + gate.unsqueeze(1) * x_o

        elif self.bidirectional == "v2":
            shift_m, scale_m, gate_m, shift_o, scale_o, gate_o = self.adaLN_modulation(c).chunk(6, dim=1)

            x = modulate(self.norm1(x), shift_m, scale_m)
            x_f = self.mixer1(x, inference_params=inference_params)
            x_b = self.mixer2(x.flip([1]), inference_params=inference_params)
            x_fb = (x_f + x_b.flip([1])) * 0.5

            x = x + gate_m.unsqueeze(1) * x_fb 
            x_o = self.mlp(modulate(self.norm2(x), shift_o, scale_o)) 

            x = x + gate_o.unsqueeze(1) * x_o
        
        else:            
            raise NotImplementedError("not implemented")

        return x

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 MOE DiT Model                                 #
#################################################################################

class DiTMoEBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, moe_num_expert=8, moe_top_k=2, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.experts = nn.ModuleList([Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
                                     for _ in range(moe_num_expert)])

        self.gate = NaiveGate(hidden_size, moe_num_expert, moe_top_k)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )

    def forward(self, x, c) -> torch.Tensor:

        shift_msa, scale_msa, gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        norm_x = self.norm2(x)
        # norm_x = modulate(self.norm2(x), shift_mlp, scale_mlp)  
        norm_x_squashed = norm_x.view(-1, norm_x.shape[-1])  

        gate_top_k_idx, gate_top_k_score = self.gate(norm_x_squashed, return_all_scores=False)  # N*SEQ, TOPK

        results = torch.zeros_like(norm_x_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(gate_top_k_idx == i)
            score = gate_top_k_score[batch_idx, nth_expert][..., None]
            pick = norm_x_squashed[batch_idx]
            results[batch_idx] += expert(pick) * score
        
        results = gate_mlp.unsqueeze(1) * results.view(x.shape)

        return x + results

class NaiveGate(nn.Module):

    def __init__(self, n_embd, moe_num_expert, moe_top_k):
        super().__init__()
        self.gate = nn.Linear(n_embd, moe_num_expert)
        self.top_k = moe_top_k

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  

        gate_top_k_score = nn.functional.softmax(gate_top_k_val, dim=1)

        if return_all_scores:
            gate_score = nn.functional.softmax(gate, dim=-1)

            return gate_top_k_idx, gate_top_k_score, gate, gate_score
        return gate_top_k_idx, gate_top_k_score


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x



class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        block_type="DiTBlock",
        bidirectional=None,
        moe_num_expert=8, 
        moe_top_k=2
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.block_type = block_type
        self.bidirectional = bidirectional

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        if self.block_type == "DiTBlock":
            self.blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
            ])
        elif self.block_type == "DiTMoEBlock":
            self.blocks = nn.ModuleList([
                DiTMoEBlock(hidden_size, num_heads, moe_num_expert=moe_num_expert, moe_top_k=moe_top_k, mlp_ratio=mlp_ratio) for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList(
                [   
                    MambaBlock(
                        hidden_size=hidden_size,
                        layer_idx=i, 
                        bidirectional=bidirectional, 
                    )
                    for i in range(depth)
                ]
            )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.block_type == "DiTBlock" or self.block_type == "DiTMoEBlock":
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        else:
            for idx_block, block in enumerate(self.blocks):
                if self.bidirectional == "v1":
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

                elif self.bidirectional == "v2":
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)   
        # for block in self.blocks:
        #     x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)       # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

# ================================ MOE ============================================
def DiT_XL_MOE_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, moe_num_expert=8, mlp_ratio=4.0, block_type="DiTMoEBlock", **kwargs)

def DiT_B_MOE_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, moe_num_expert=8, mlp_ratio=4.0, block_type="DiTMoEBlock", **kwargs)

# =============================== MAMBA ===========================================
def DiT_XL_BIMBv1_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, block_type="MambaBlock", bidirectional="v1", **kwargs)

def DiT_B_BIMBv1_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, block_type="MambaBlock", bidirectional="v1", **kwargs)

def DiT_XL_BIMBv2_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, block_type="MambaBlock", bidirectional="v2", **kwargs)

def DiT_B_BIMBv2_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, block_type="MambaBlock", bidirectional="v2", **kwargs)

# ================================= ORIGIN ===========================================
def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL-MOE/2": DiT_XL_MOE_2, "DiT-B-MOE/4": DiT_B_MOE_4,
    'DiT-XL-BIMBv2/2': DiT_XL_BIMBv2_2, 'DiT-XL-BIMBv1/2': DiT_XL_BIMBv1_2,
    'DiT-B-BIMBv2/4': DiT_B_BIMBv2_4,  'DiT-B-BIMBv1/4': DiT_B_BIMBv1_4,
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
