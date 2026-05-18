import torch.nn as nn
from sam_ori.modeling.image_encoder import ImageEncoderViT
import torch
from torch.nn import functional as F
from Trifuse import *
from einops import rearrange
import clip
import math

class Image_encoder(nn.Module):
    def __init__(self, image_encoder: ImageEncoderViT):
        super(Image_encoder, self).__init__()
        self.image_encoder = image_encoder

    def forward(self, x):
        x_out = self.image_encoder(x)
        return x_out

class Sammus(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model, self.clip_preprocess = clip.load(device="cuda")
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.image_encoder = Image_encoder(image_encoder=ImageEncoderViT(embed_dim=768, depth=12, num_heads=12, global_attn_indexes=[2, 5, 8, 11]))
        self.image_encoder.load_state_dict(torch.load('/home/baoliuxin/find/sam_vit_b_01ec64.pth'), strict=False)
        for name, param in self.image_encoder.named_parameters():
            if "post_pos_embed" not in name:
                param.requires_grad = False

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.Com_conv = nn.Sequential(
            ConvModule(768, 256, 3, bn=True, relu=True),
            ConvModule(256, 50, 1, bn=False, relu=False))
        self.conv = nn.Conv2d(768*2, 768, 1)
        self.linear = nn.Linear(512, 768)
        self.attn = CrossAttention(768, 768)
        self.attn_image = TGCrossFusion()

        self.aspp1 = ASPP(in_channels=768, atrous_rates=[3, 6, 12, 18], out_channels=768)
        self.conv_try = nn.Conv2d(768, 50, 1)

    def forward(self, input, input_th, input_d, s_input, s_input_th, s_input_d, s_mask, text, name):

        Q_embedding_v, QFeats = self.get_visual_embs(input)
        Q_embedding_t, QFeats_th = self.get_visual_embs(input_th)
        Q_embedding_d, QFeats_d = self.get_visual_embs(input_d)
        _, SFeats = self.get_visual_embs(s_input)
        _, SFeats_th = self.get_visual_embs(s_input_th)
        _, SFeats_d = self.get_visual_embs(s_input_d)

        text = clip.tokenize(text).cuda()
        text_features = self.clip_model.encode_text(text).float()
        text_features = text_features / text_features.norm(dim=1, keepdim=True).float()

        feats = []
        for idx, (feat, feat_d, feat_t) in enumerate(zip(QFeats, QFeats_d, QFeats_th)):

            f = self.attn_image(text_features, feat, feat_t, feat_d)
            f = self.aspp1(f)
            feats.append(f)

            # b, c, h, w = feat.shape

            # feat = rearrange(feat, 'b c h w -> b (h w) c')
            # feats.append(self.attn_image(text_features, feat, feat))
            # # query_feat = self.attn_image(text_features, feat, feat)
            #
            # feat_d = rearrange(feat_d, 'b c h w -> b (h w) c')
            # feats_d.append(feat_d)
            # # query_feat_d = self.attn_image(text_features, feat_d, feat_d)
            #
            # feat_t = rearrange(feat_t, 'b c h w -> b (h w) c')
            # feats_t.append(feat_t)
            # query_feat_t = self.attn_image(text_features, feat_t, feat_t)
            # print(query_feat.size())

        support_feature = [a + b + c for a, b, c in zip(SFeats, SFeats_th, SFeats_d)]
        S_feat = self.make_feature(support_feature, s_mask.clone())

        image_feats = []
        for idx, (q, s) in enumerate(zip(feats, S_feat)):
            b, c, h, w = q.shape
            q_r = rearrange(q, 'b c h w -> b (h w) c')
            s_r = rearrange(s, 'b c h w -> b (h w) c')

            feat = self.attn(s_r, q_r, q_r)
            feat = self.conv(torch.cat((feat, q), dim=1))
            feat = self.Com_conv(feat)
            image_feats.append(feat)

        return feats, image_feats

    def make_feature(self, features, masks):
        for idx, feature in enumerate(features):
            mask = F.interpolate(masks, feature.size()[2:], mode='bilinear',
                                 align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        image_embeddings_list = []
        Feats = []
        for i in range(pixel_values.shape[0]):
            image_embeddings, Feat = self.image_encoder(
                pixel_values[i].unsqueeze(0)
            )
            image_embeddings_list.append(image_embeddings)
            Feats.append(Feat)

        PFeats = []
        for ib in range(len(Feats[0])):
            Cfeat = Feats[0][ib].permute(0, 3, 1, 2)
            for j in range(1, len(Feats)):
                Cfeat = torch.cat((Cfeat, Feats[j][ib].permute(0, 3, 1, 2)), 0)
            PFeats.append(Cfeat)

        image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings, PFeats

class CrossAttention(nn.Module):
    def __init__(self, in_channels, emb_dim):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, Q, K, V, pad_mask=None):
        b, hw, c = Q.shape
        h = w = int(math.sqrt(hw))

        att_weights = torch.einsum('bid,bjd->bij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            mask = pad_mask[:, None, :].expand(-1, hw, -1)
            att_weights = att_weights.masked_fill(mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bij,bjd->bid', att_weights, V)

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        out = self.proj_out(out)

        return out

class TGCrossFusion(nn.Module):
    def __init__(self, text_dim=512, feat_dim=768, heads=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.heads = heads

        self.text_q_proj = nn.Linear(text_dim, feat_dim)

        self.k_proj = nn.ModuleList([nn.Conv2d(feat_dim, feat_dim, 1) for i in range(3)])
        self.v_proj = nn.ModuleList([nn.Conv2d(feat_dim, feat_dim, 1) for i in range(3)])

        self.fusion_w = nn.Linear(feat_dim * 3, 3)

    def cross_attend(self, q, feat, k_proj, v_proj):

        B, C, H, W = feat.shape

        K = k_proj(feat).flatten(2).transpose(1, 2)  # [B, HW, C]
        V = v_proj(feat).flatten(2).transpose(1, 2)  # [B, HW, C]

        attn = torch.softmax(q @ K.transpose(-1, -2) / (C ** 0.5), dim=-1)
        ctx = attn @ V
        ctx = ctx.mean(dim=1).view(B, C, 1, 1)

        out = feat * torch.sigmoid(ctx)
        # out = feat + ctx

        return out

    def forward(self, text, F_rgb, F_t, F_d):
        Q = self.text_q_proj(text)

        out_rgb = self.cross_attend(Q, F_rgb, self.k_proj[0], self.v_proj[0])
        out_t   = self.cross_attend(Q, F_t,   self.k_proj[1], self.v_proj[1])
        out_d   = self.cross_attend(Q, F_d,   self.k_proj[2], self.v_proj[2])

        concat = torch.cat([out_rgb, out_t, out_d], dim=1)

        w = self.fusion_w(concat.permute(0, 2, 3, 1))
        w = torch.softmax(w, dim=-1)

        w_rgb, w_t, w_d = w[..., 0:1], w[..., 1:2], w[..., 2:3]
        w_rgb = w_rgb.permute(0, 3, 1, 2)
        w_t   = w_t.permute(0, 3, 1, 2)
        w_d   = w_d.permute(0, 3, 1, 2)

        F_fused = w_rgb * out_rgb + w_t * out_t + w_d * out_d
        return F_fused

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bn=True, relu=True, bias=True):
        super(ConvModule, self).__init__()
        self.inp_dim = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = LayerNorm2d(out_channels)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            LayerNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            LayerNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            LayerNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            LayerNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            # print(x.size())
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)