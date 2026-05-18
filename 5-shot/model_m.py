import torch
import torch.nn as nn
from fusion import CLIPAlpha
from sam_ori.network.sammus import Sammus
from sam_ori.modeling.common import LayerNorm2d
from torch.nn import functional as F

class Mnet(nn.Module):
    def __init__(self, batch, shot):
        super().__init__()
        self.shot = shot
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.clip = CLIPAlpha(batch).cuda()
        self.sam = Sammus().cuda()
        self.mlp1 = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.final1 = nn.Sequential(
            ConvModule(32, 16, 3, bn=True, relu=True),
            ConvModule(16, 2, 1, bn=False, relu=False))
        self.final2 = nn.Sequential(
            ConvModule(32, 16, 3, bn=True, relu=True),
            ConvModule(16, 2, 1, bn=False, relu=False))
        self.final3 = nn.Sequential(
            ConvModule(32, 16, 3, bn=True, relu=True),
            ConvModule(16, 2, 1, bn=False, relu=False))
        self.final4 = nn.Sequential(
            ConvModule(32, 16, 3, bn=True, relu=True),
            ConvModule(16, 2, 1, bn=False, relu=False))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )
        self.up8 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, bias=True),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, bias=True),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, bias=True),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0, bias=True),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.Com_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=32, kernel_size=1, padding=0, bias=True),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.Com_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=32, kernel_size=1, padding=0, bias=True),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.Com_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=32, kernel_size=1, padding=0, bias=True),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.Com_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=32, kernel_size=1, padding=0, bias=True),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv_image = nn.Sequential(
            nn.Conv2d(in_channels=4096, out_channels=512, kernel_size=1, padding=0, bias=True),
            LayerNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            LayerNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=50, kernel_size=1, padding=0, bias=True),
            LayerNorm2d(50),
            nn.ReLU(inplace=True),
        )
        self.conv_com = nn.Conv2d(in_channels=100, out_channels=50, kernel_size=1, padding=0, bias=True)

    def forward(self, input, input_th, input_d, s_inputs, s_input_ths, s_input_ds, s_masks, text_O, text_A, text_R, text, name):
        x1, x2, x3, x4 = [], [], [], []
        image1, image2, image3, image4 = [], [], [], []
        text1, text2, text3, text4 = [], [], [], []

        for i in range(self.shot):
            s_input = s_inputs[:, i, :, :, :]
            s_input_th = s_input_ths[:, i, :, :, :]
            s_input_d = s_input_ds[:, i, :, :, :]
            s_mask = s_masks[:, i, :, :]

            query_feat, image_feat = self.sam(input, input_th, input_d, s_input, s_input_th, s_input_d, s_mask, text, name)
            f_xs = []
            xs = []

            for idx, (q_feat, i_feat) in enumerate(zip(query_feat, image_feat)):
                mask_embedding = self.clip(q_feat, text_O, text_A, text_R)
                VL = self.mlp1(query_feat[3].permute(0, 2, 3, 1))
                ZR = self.mlp2(mask_embedding)
                x = torch.einsum('BHWD,BND->BNHW', VL, ZR)
                fx = self.conv_com(torch.cat((x, i_feat), dim=1))
                f_xs.append(fx)
                xs.append(x)

            x1.append(f_xs[0])
            x2.append(f_xs[1])
            x3.append(f_xs[2])
            x4.append(f_xs[3])

            image1.append(image_feat[0])
            image2.append(image_feat[1])
            image3.append(image_feat[2])
            image4.append(image_feat[3])

            text1.append(xs[0])
            text2.append(xs[1])
            text3.append(xs[2])
            text4.append(xs[3])

        x1 = torch.cat(x1, dim=0).mean(dim=0, keepdim=True)
        x2 = torch.cat(x2, dim=0).mean(dim=0, keepdim=True)
        x3 = torch.cat(x3, dim=0).mean(dim=0, keepdim=True)
        x4 = torch.cat(x4, dim=0).mean(dim=0, keepdim=True)

        image1 = torch.cat(image1, dim=0).mean(dim=0, keepdim=True)
        image2 = torch.cat(image2, dim=0).mean(dim=0, keepdim=True)
        image3 = torch.cat(image3, dim=0).mean(dim=0, keepdim=True)
        image4 = torch.cat(image4, dim=0).mean(dim=0, keepdim=True)

        text1 = torch.cat(text1, dim=0).mean(dim=0, keepdim=True)
        text2 = torch.cat(text2, dim=0).mean(dim=0, keepdim=True)
        text3 = torch.cat(text3, dim=0).mean(dim=0, keepdim=True)
        text4 = torch.cat(text4, dim=0).mean(dim=0, keepdim=True)

        x1 = self.Com_conv1(x1)
        x2 = self.Com_conv2(x2)
        x3 = self.Com_conv3(x3)
        x4 = self.Com_conv4(x4)

        x4e = self.conv4(x4)
        x4e_pred = self.final4(x4e)
        x4e_pred = self.up4(x4e_pred)

        x3 = self.up2(x3)
        x3e = self.conv3(torch.cat((x4e, x3), dim=1))
        x3e_pred = self.final3(x3e)
        x3e_pred = self.up2(x3e_pred)

        x2 = self.up4(x2)
        x2e = self.conv2(torch.cat((x3e, x2), dim=1))
        x2e_pred = self.final2(x2e)

        x1 = self.up8(x1)
        x1e = self.conv1(torch.cat((x2e, x1), dim=1))
        x1e_pred = self.final1(x1e)


        image1 = self.Com_conv1(image1)
        image2 = self.Com_conv2(image2)
        image3 = self.Com_conv3(image3)
        image4 = self.Com_conv4(image4)

        image4e = self.conv4(image4)
        image4e_pred = self.final4(image4e)
        image4e_pred = self.up4(image4e_pred)

        image3 = self.up2(image3)
        image3e = self.conv3(torch.cat((image4e, image3), dim=1))
        image3e_pred = self.final3(image3e)
        image3e_pred = self.up2(image3e_pred)

        image2 = self.up4(image2)
        image2e = self.conv2(torch.cat((image3e, image2), dim=1))
        image2e_pred = self.final2(image2e)

        image1 = self.up8(image1)
        image1e = self.conv1(torch.cat((image2e, image1), dim=1))
        image1e_pred = self.final1(image1e)


        text1 = self.Com_conv1(text1)
        text2 = self.Com_conv2(text2)
        text3 = self.Com_conv3(text3)
        text4 = self.Com_conv4(text4)

        text4e = self.conv4(text4)
        text4e_pred = self.final4(text4e)
        text4e_pred = self.up4(text4e_pred)

        text3 = self.up2(text3)
        text3e = self.conv3(torch.cat((text4e, text3), dim=1))
        text3e_pred = self.final3(text3e)
        text3e_pred = self.up2(text3e_pred)

        text2 = self.up4(text2)
        text2e = self.conv2(torch.cat((text3e, text2), dim=1))
        text2e_pred = self.final2(text2e)

        text1 = self.up8(text1)
        text1e = self.conv1(torch.cat((text2e, text1), dim=1))
        text1e_pred = self.final1(text1e)

        return x1e_pred, x2e_pred, x3e_pred, x4e_pred, image1e_pred, image2e_pred, image3e_pred, image4e_pred, text1e_pred, text2e_pred, text3e_pred, text4e_pred

    def compute_objective(self, logit_masks, gt_mask):
        if isinstance(logit_masks, list):
            loss = 0
            for logit_mask in logit_masks:
                bsz = logit_mask.size(0)
                logit_mask = logit_mask.view(bsz, 2, -1)
                gt_mask = gt_mask.view(bsz, -1).long()
                loss = loss + self.cross_entropy_loss(logit_mask, gt_mask)

        else:
            logit_mask = logit_masks
            bsz = logit_mask.size(0)
            logit_mask = logit_mask.view(bsz, 2, -1)
            gt_mask = gt_mask.view(bsz, -1).long()
            loss = self.cross_entropy_loss(logit_mask, gt_mask)

        return loss



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


