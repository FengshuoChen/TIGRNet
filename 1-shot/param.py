import torch
from model import Mnet
import os
from thop import profile
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def profile_mnet():
    bsz = 1
    img_size = 512

    model = Mnet(bsz)
    model.cuda()

    input_img = torch.randn(bsz, 3, img_size, img_size).cuda()
    input_th = torch.randn(bsz, 3, img_size, img_size).cuda()
    input_d = torch.randn(bsz, 3, img_size, img_size).cuda()

    s_input = torch.randn(bsz, 3, img_size, img_size).cuda()
    s_input_th = torch.randn(bsz, 3, img_size, img_size).cuda()
    s_input_d = torch.randn(bsz, 3, img_size, img_size).cuda()
    s_mask = torch.randn(bsz, 1, img_size, img_size).cuda()

    text_O, text_A, text_R, text = ["text"] * bsz, ["text"] * bsz, ["text"] * bsz, ["text"] * bsz
    name = ["name"] * bsz

    inputs = (input_img, input_th, input_d, s_input, s_input_th, s_input_d, s_mask,
              text_O, text_A, text_R, text, name)

    # 4. 开始分析
    print("--- 正在计算参数量与 FLOPs ---")
    flops, params = profile(model, inputs=inputs)
    # from fvcore.nn import FlopCountAnalysis, flop_count_table
    #
    # # 1. 创建分析器
    # flops_analyzer = FlopCountAnalysis(model, inputs)
    #
    # # 2. 打印不同层级的详细表格
    # # max_depth 控制显示的深度，1 只显示最外层，3-4 可以看到 SAM 内部的 encoder 等
    # print(flop_count_table(flops_analyzer, max_depth=3))
    #
    print(f"Total Params: {params / 1000000:.2f} M")
    print(f"Total FLOPs: {flops / 1000000000:.2f} G")
    #
    #
    # summary(model, inputs=inputs)


if __name__ == "__main__":
    profile_mnet()