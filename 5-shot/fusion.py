import torch
import torch.nn as nn
import clip
import os
import math
from torch.nn import functional as F

class CLIPAlpha(nn.Module):
    def __init__(self, batch):
        super().__init__()
        self.clip_model, self.clip_preprocess = clip.load(device="cuda")
        self.mask_embedding = nn.Parameter(torch.randn(batch, 50, 512))
        self.Sattn = SAttention(512, 512, 16, 16, 4)
        self.SVattn = SVAttention(512, 512, 4096, 16, 16, 4, 512)
        self.Rattn = RelationAttention(512, 4)

        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, image_feature, text_O, text_A, text_R):
        text_O = clip.tokenize(text_O).cuda()
        text_A = clip.tokenize(text_A).cuda()
        text_R = clip.tokenize(text_R).cuda()

        text_features_o = self.clip_model.encode_text(text_O)
        text_features_o = text_features_o / text_features_o.norm(dim=1, keepdim=True).float()
        text_features_a = self.clip_model.encode_text(text_A)
        text_features_a = text_features_a / text_features_a.norm(dim=1, keepdim=True).float()
        text_features_r = self.clip_model.encode_text(text_R)
        text_features_r = text_features_r / text_features_r.norm(dim=1, keepdim=True).float()


        Zs, Atts = self.Sattn(text_features_a, text_features_o, self.mask_embedding)
        ZV = self.SVattn(text_features_a, Atts, Zs, image_feature)

        R_normalized = F.normalize(text_features_r, p=2, dim=2)
        cos_sim = torch.bmm(R_normalized, R_normalized.transpose(1, 2))
        Re = (cos_sim > 0.5).int()
        ZR = self.Rattn(ZV, Atts.permute(0, 2, 1), Re.float())

        return ZR


class MHCA(nn.Module):
    def __init__(self, d_q, d_kv, d_k, d_v, n_heads):
        super(MHCA, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = nn.Linear(d_q, n_heads * d_k)
        self.W_k = nn.Linear(d_kv, n_heads * d_k)
        self.W_v = nn.Linear(d_kv, n_heads * d_v)
        self.W_o = nn.Linear(n_heads * d_v, d_q)

    def forward(self, Q, K, V):
        B, N, _ = Q.shape
        _, M, _ = K.shape
        Q = self.W_q(Q).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, M, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, M, self.n_heads, self.d_v).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_probs, V)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)

        out = self.W_o(out)
        att_map = attn_probs.mean(dim=1)

        return out, att_map.permute(0, 2, 1)


class SAttention(nn.Module):
    def __init__(self, d_q, d_kv, d_k, d_v, n_heads):
        super(SAttention, self).__init__()
        self.mhca = MHCA(d_q, d_kv, d_k, d_v, n_heads)
        self.norm = nn.LayerNorm(d_q)

    def forward(self, A, O, Z):
        priors = A + O
        out, att_map = self.mhca(Z, priors, priors)
        out = self.norm(Z + out)
        return out, att_map

class SVAttention(nn.Module):
    def __init__(self, d_attr, d_mask, d_img, d_k, d_v, n_heads, n_scales):
        super().__init__()
        self.mhca = MHCA(d_mask, d_img, d_k, d_v, n_heads)
        self.norm = nn.LayerNorm(d_mask)
        self.mlp = nn.Sequential(
            nn.Linear(d_attr, d_attr),
            nn.ReLU(),
            nn.Linear(d_attr, n_scales)
        )

    def forward(self, A, AttS, ZS, V_list):
        _, N, Ds = ZS.shape
        AZ = torch.einsum("bmn,bmd->bnd", AttS, A)
        S = self.mlp(AZ)
        S = F.softmax(S, dim=-1)

        V = torch.flatten(V_list, start_dim=2)
        out, _ = self.mhca(ZS, V, V)
        out = self.norm(ZS + out)

        ZV_weighted = S * out

        # outs = []
        # for V in V_list:
        #     V = torch.flatten(V, start_dim=2)
        #     out, _ = self.mhca(ZS, V, V)
        #     out = self.norm(ZS + out)
        #     outs.append(out)
        #
        # ZV = torch.stack(outs, dim=1).permute(0, 2, 1, 3)
        # S_exp = S.unsqueeze(-1)
        # ZV_weighted = torch.sum(ZV * S_exp, dim=2)

        return ZV_weighted

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # projection matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, ZV, att_override=None):
        """
        ZV: [B, N, D]
        att_override: [B, N, N] (optional) pre-computed attention matrix (e.g., Att_tilde)
        """
        B, N, D = ZV.shape

        # project
        Q = self.W_q(ZV).view(B, N, self.num_heads, self.d_head).transpose(1, 2)  # [B, h, N, d_h]
        K = self.W_k(ZV).view(B, N, self.num_heads, self.d_head).transpose(1, 2)  # [B, h, N, d_h]
        V = self.W_v(ZV).view(B, N, self.num_heads, self.d_head).transpose(1, 2)  # [B, h, N, d_h]

        if att_override is None:
            # standard attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, h, N, N]
            attn = F.softmax(scores, dim=-1)
        else:
            # broadcast att_override across heads
            attn = att_override.unsqueeze(1).expand(B, self.num_heads, N, N)  # [B, h, N, N]

        out = torch.matmul(attn, V)  # [B, h, N, d_h]
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        return self.W_o(out)


class RelationAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads)
        self.linear = nn.Linear(1, 1, bias=True)  # simple linear for Att + Re^Z fusion
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, ZV, AttS, Re):
        """
        ZV: [B, N, D] mask embeddings
        AttS: [B, N, M] prior-mask map
        Re: [B, M, M] object relation map
        """
        B, N, D = ZV.shape
        _, _, M = AttS.shape

        # 1. Compute mask-level relation map Re^Z
        ReZ = torch.bmm(torch.bmm(AttS, Re), AttS.transpose(1, 2))  # [B, N, N]

        # 2. Compute original attention Att
        Att = torch.matmul(ZV, ZV.transpose(1, 2)) / math.sqrt(D)  # [B, N, N]

        # 3. Fuse Att + ReZ, then apply linear + softmax
        fused = Att + ReZ

        fused = self.linear(fused.unsqueeze(-1)).squeeze(-1)  # [B, N, N]
        Att_tilde = F.softmax(fused, dim=-1)

        # 4. MHSA using Att_tilde
        mhsa_out = self.mhsa(ZV, att_override=Att_tilde)

        # 5. Add & Norm & FFN
        out = self.norm(ZV + mhsa_out)
        out = out + self.ffn(out)

        return out  # final Z^R [B, N, D]