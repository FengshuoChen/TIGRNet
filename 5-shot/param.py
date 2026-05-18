import torch
from model import Mnet
import os
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def profile_mnet():
    bsz = 1
    shot = 5
    img_size = 512

    model = Mnet(bsz, shot)
    model.cuda()

    input_img = torch.randn(bsz, 3, img_size, img_size).cuda()
    input_th = torch.randn(bsz, 3, img_size, img_size).cuda()
    input_d = torch.randn(bsz, 3, img_size, img_size).cuda()

    s_input = torch.randn(bsz, shot, 3, img_size, img_size).cuda()
    s_input_th = torch.randn(bsz, shot, 3, img_size, img_size).cuda()
    s_input_d = torch.randn(bsz, shot, 3, img_size, img_size).cuda()
    s_mask = torch.randn(bsz, shot, 1, img_size, img_size).cuda()

    text_O, text_A, text_R, text = ["text"] * bsz, ["text"] * bsz, ["text"] * bsz, ["text"] * bsz
    name = ["name"] * bsz

    inputs = (input_img, input_th, input_d, s_input, s_input_th, s_input_d, s_mask,
              text_O, text_A, text_R, text, name)

    # 4. 开始分析
    print("--- 正在计算参数量与 FLOPs ---")
    flops, params = profile(model, inputs=inputs)

    print(f"Total Params: {params / 1000000:.2f} M")
    print(f"Total FLOPs: {flops / 1000000000:.2f} G")

    # 5. 手动二次确认参数量 (PyTorch 原生方式)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Native PyTorch Total Params: {total_params / 1e6:.2f} M")
    print(f"Native PyTorch Trainable Params: {trainable_params / 1e6:.2f} M")

    def count_parameters(model):
        # numel() 返回张量中的元素总数
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params / 1e6  # 转换为 M (百万)

    print(f"Total Trainable Params: {count_parameters(model):.2f} M")

    # 1. 计算 FLOPs (注意：fvcore 统计的是每秒浮点运算，通常被称为 Flops)
    flops = FlopCountAnalysis(model, inputs)
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} G")  # 转换为 G

    # 2. 打印精美的参数/计算量表格（可以看到具体哪个模块大）
    print(parameter_count_table(model))



if __name__ == "__main__":
    profile_mnet()