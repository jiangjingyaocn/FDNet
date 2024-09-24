import torch
from torch import nn
from .transformer_block import Block
from timm.models.layers import trunc_normal_
from einops import rearrange
import math

class TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                 Block(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                        for i in range(depth)])

        self.rgb_norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, rgb_fea):

        for block in self.blocks:
            rgb_fea = block(rgb_fea)

        rgb_fea = self.rgb_norm(rgb_fea)

        return rgb_fea


class token_TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(token_TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                 Block(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                        for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, fea):

        for block in self.blocks:
            fea = block(fea)

        fea = self.norm(fea)

        return fea

class group_token_TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(group_token_TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                 Block(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                        for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, fea):
        # fea [B, 1 + 1 + 1, 384]
        bs, _, _ = fea.shape

        fea = rearrange(fea, "b c dim -> (b c) dim").unsqueeze(0)

        for block in self.blocks:
            fea = block(fea)

        fea = self.norm(fea)

        fea = rearrange(fea.squeeze(), "(b c) dim -> b c dim", b=bs)


        return fea
class Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(Transformer, self).__init__()

        self.encoderlayer = TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)

    def forward(self, rgb_fea):

        rgb_memory = self.encoderlayer(rgb_fea)

        return rgb_memory


class saliency_token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        B, N, C = fea.shape
        x = self.norm(fea)
        T_s, F_s = x[:, 0, :].unsqueeze(1), x[:, 1:-2, :]
        # T_s [B, 1, 384]  F_s [B, 14*14, 384]

        q = self.q(F_s).reshape(B, N-3, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N-3, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)
        infer_fea = infer_fea + fea[:, 1:-2, :]
        _,_,number,_ = attn.shape
        return infer_fea,attn.reshape(B,  int(math.sqrt(number)), int(math.sqrt(number)))


class contour_token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        B, N, C = fea.shape
        x = self.norm(fea)
        T_s, F_s = x[:, -2, :].unsqueeze(1), x[:, 1:-2, :]
        # T_s [B, 1, 384]  F_s [B, 14*14, 384]

        q = self.q(F_s).reshape(B, N-3, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # attn = attn.softmax(dim=-1)
        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N-3, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + fea[:, 1:-2, :]
        # return infer_fea,attn.reshape(B, N-3, C)
        _,_,number,_ = attn.shape
        return infer_fea,attn.reshape(B,  int(math.sqrt(number)), int(math.sqrt(number)))

class background_token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        B, N, C = fea.shape
        x = self.norm(fea)
        T_s, F_s = x[:, -1, :].unsqueeze(1), x[:, 1:-2, :]
        # T_s [B, 1, 384]  F_s [B, 14*14, 384]

        q = self.q(F_s).reshape(B, N-3, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # attn = attn.softmax(dim=-1)
        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N-3, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + fea[:, 1:-2, :]
        # return infer_fea,attn.reshape(B, N-3, C)
        _,_,number,_ = attn.shape
        return infer_fea,attn.reshape(B,  int(math.sqrt(number)), int(math.sqrt(number)))


class token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        B, N, C = fea.shape
        x = self.norm(fea)
        T_s, F_s = x[:, 0, :].unsqueeze(1), x[:, 1:-2, :]
        # T_s [B, 1, 384]  F_s [B, 14*14, 384]

        q = self.q(F_s).reshape(B, N-3, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N-3, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + fea[:, 1:-2, :]
        return infer_fea

class token_Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_Transformer, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_s = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.saliency_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.contour_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.background_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.contour_saliency_background_token_trans = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        # self.saliency_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1)
        # self.contour_token_pre = contour_token_inference(dim=embed_dim, num_heads=1)
        # self.background_token_pre = background_token_inference(dim=embed_dim, num_heads=1)

        self.group_token_attention = group_token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)

        self.saliency_token_pre = token_inference(dim=embed_dim, num_heads=1)
        self.contour_token_pre = token_inference(dim=embed_dim, num_heads=1)
        self.background_token_pre = token_inference(dim=embed_dim, num_heads=1)

    def forward(self, rgb_fea):
        B, _, _ = rgb_fea.shape
        fea_1_16 = self.mlp_s(self.norm(rgb_fea))   # [B, 14*14, 384]

        saliency_tokens = self.saliency_token.expand(B, -1, -1)
        fea_1_16 = torch.cat((saliency_tokens, fea_1_16), dim=1)
        contour_tokens = self.contour_token.expand(B, -1, -1)
        fea_1_16 = torch.cat((fea_1_16, contour_tokens), dim=1)
        background_tokens = self.background_token.expand(B, -1, -1)
        fea_1_16 = torch.cat((fea_1_16, background_tokens), dim=1)
        # fea_1_16 [B, 1 + 14*14 + 1, 384]

        fea_1_16 = self.contour_saliency_background_token_trans(fea_1_16)
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        saliency_tokens = fea_1_16[:, 0, :].unsqueeze(1)
        contour_tokens = fea_1_16[:, -2, :].unsqueeze(1)
        background_tokens = fea_1_16[:, -1, :].unsqueeze(1)

        saliency_fea_1_16 = self.saliency_token_pre(fea_1_16)
        contour_fea_1_16 = self.contour_token_pre(fea_1_16)
        background_fea_1_16 = self.background_token_pre(fea_1_16)

        return saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, background_fea_1_16, background_tokens

class token_trans(nn.Module):
    def __init__(self, in_dim=64, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_trans, self).__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        # self.group_token_attention = group_token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.saliency_token_pre = token_inference(dim=embed_dim, num_heads=1)
        self.contour_token_pre = token_inference(dim=embed_dim, num_heads=1)
        self.background_token_pre = token_inference(dim=embed_dim, num_heads=1)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.norm2_c = nn.LayerNorm(embed_dim)
        self.mlp2_c = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.norm2_b = nn.LayerNorm(embed_dim)
        self.mlp2_b = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self, fea, saliency_tokens, contour_tokens, background_tokens):
        B, _, _ = fea.shape
        # fea [B, H*W, 64]
        # project to 384 dim
        fea = self.mlp(self.norm(fea))
        # fea [B, H*W, 384]

        fea = torch.cat((saliency_tokens, fea), dim=1)
        fea = torch.cat((fea, contour_tokens), dim=1)
        fea = torch.cat((fea, background_tokens), dim=1)
        # [B, 1 + H*W + 1, 384]
        # 此处修改一下，残差结构
        fea = self.encoderlayer(fea)
        # group_fea = self.group_token_attention(fea)
        # fea = fea + group_fea

        # fea [B, 1 + H*W + 1 + 1, 384]
        saliency_tokens = fea[:, 0, :].unsqueeze(1)
        contour_tokens = fea[:, -2, :].unsqueeze(1)
        background_tokens = fea[:, -1, :].unsqueeze(1)

        # saliency_fea [B, H*W, 384]
        saliency_fea = self.saliency_token_pre(fea)
        # saliency_fea [B, H*W, 384]
        contour_fea = self.contour_token_pre(fea)
        # contour_fea [B, H*W, 384]
        background_fea = self.background_token_pre(fea)

        # reproject back to 64 dim
        saliency_fea = self.mlp2(self.norm2(saliency_fea))
        contour_fea = self.mlp2_c(self.norm2_c(contour_fea))
        background_fea = self.mlp2_b(self.norm2_b(background_fea))

        return saliency_fea, contour_fea, background_fea, fea, saliency_tokens, contour_tokens, background_tokens
        # return saliency_fea, contour_fea, background_fea, fea, group_saliency_tokens, group_contour_tokens, background_tokens
