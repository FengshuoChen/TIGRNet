r""" Hypercorrelation Squeeze testing code """
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2
from model import *
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from common import dataset_mask_train, dataset_mask_val
import os
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def visual_depth(img):
    img = img.astype(np.float64)
    min_val = np.min(img)
    max_val = np.max(img)
    img_norm = (img - min_val) / (max_val - min_val)

    img_norm = (img_norm * 255).astype(np.uint8)

    return img_norm


def apply_mask(image, mask, color, alpha=0.5):
    r""" Apply mask to the given image. """
    for c in range(image.shape[2]):
        image[:, :, c] = np.where(mask > 0.5 * 255,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def visualize(ori_query_rgb, ori_query_d, ori_query_th, q_mask,
              ori_support_rgb, ori_support_d, ori_support_th, s_mask,
              ori_name, q_pred, i):

    colors_dict = {'red': (255, 50, 50), 'blue': (102, 140, 255)}

    # ---- 转换维度 + 类型 ----
    def to_numpy_img(x):
        x = x.squeeze().cpu().numpy()
        if x.ndim == 3 and x.shape[0] == 3:   # (3,H,W) → (H,W,3)
            x = np.transpose(x, (1, 2, 0))
        # 归一化到0-255范围
        if x.dtype != np.uint8:
            x = (x - x.min()) / (x.max() - x.min() + 1e-8) * 255
            x = x.astype(np.uint8)
        return x

    ori_query_rgb = to_numpy_img(ori_query_rgb)
    ori_query_d = to_numpy_img(ori_query_d)
    ori_query_th = to_numpy_img(ori_query_th)
    q_mask = q_mask.squeeze().cpu().numpy()

    ori_support_rgb = to_numpy_img(ori_support_rgb)
    ori_support_d = to_numpy_img(ori_support_d)
    ori_support_th = to_numpy_img(ori_support_th)
    s_mask = s_mask.squeeze().cpu().numpy()

    q_h_ori, q_w_ori = q_mask.shape

    q_pred = F.interpolate(q_pred.float().unsqueeze(1), [q_h_ori, q_w_ori],
                           mode='bilinear', align_corners=True)
    q_pred = q_pred.cpu().squeeze().numpy() * 255

    path1 = "./visual_maps"
    if not os.path.exists(path1):
        os.mkdir(path1)

    # ---- 应用 mask 并保存 ----
    def safe_save(np_img, path):
        """确保 np_img 是 uint8 格式后保存"""
        if np_img.dtype != np.uint8:
            np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        Image.fromarray(np_img).save(path)

    ori_name = ori_name[0]

    query_gt = apply_mask(ori_query_rgb.copy(), q_mask, colors_dict['red'])
    safe_save(query_gt, os.path.join(path1, ori_name + '_query_GT.png'))

    query_pred = apply_mask(ori_query_rgb.copy(), q_pred, colors_dict['blue'])
    safe_save(query_pred, os.path.join(path1, ori_name + '_query_pred.png'))

    support_gt = apply_mask(ori_support_rgb.copy(), s_mask, colors_dict['red'])
    safe_save(support_gt, os.path.join(path1, ori_name + '_support_GT.png'))

    # ---- 保存原图 ----
    safe_save(ori_query_rgb, os.path.join(path1, ori_name + '_query_rgb.png'))
    safe_save(visual_depth(ori_query_d), os.path.join(path1, ori_name + '_query_d.png'))
    safe_save(ori_query_th, os.path.join(path1, ori_name + '_query_th.png'))
    safe_save(ori_support_rgb, os.path.join(path1, ori_name + '_support_rgb.png'))
    safe_save(visual_depth(ori_support_d), os.path.join(path1, ori_name + '_support_d.png'))
    safe_save(ori_support_th, os.path.join(path1, ori_name + '_support_th.png'))



def test(model, dataloader, sub_list, nshot):
    r""" Test HSNet """

    average_meter = AverageMeter(sub_list)
    for idx, (
        query_rgb, query_th, query_d, query_mask,
        support_rgb, support_th, support_d, support_mask,
        subcls, text_O, text_A, text_R, text, name
    ) in enumerate(tqdm(dataloader)):

        support_mask = support_mask.float()
        query_mask = query_mask.float()
        support_rgb = support_rgb.cuda()
        support_d = support_d.cuda()
        support_th = support_th.cuda()
        support_mask = support_mask.cuda()
        query_rgb = query_rgb.cuda()
        query_d = query_d.cuda()
        query_th = query_th.cuda()
        query_mask = query_mask.cuda()
        subcls = subcls.cuda()

        logit_mask, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = model(
            query_rgb, query_th, query_d,
            support_rgb, support_th, support_d,
            support_mask, text_O, text_A, text_R, text, name
        )

        if isinstance(logit_mask, list):
            logit_mask = logit_mask[-1]

        pred_mask = logit_mask.argmax(dim=1)
        loss = model.compute_objective(logit_mask, query_mask)

        # 2. Evaluate prediction
        query_mask = query_mask.squeeze(1)
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, query_mask)
        average_meter.update(area_inter, area_union, subcls, loss.detach().clone())

        # ✅ 调用 visualize()
        # 限制只保存前几个样本的可视化结果（避免过多文件）
        # if idx < 5:
        #     visualize(
        #         ori_query_rgb=query_rgb.cpu(),
        #         ori_query_d=query_d.cpu(),
        #         ori_query_th=query_th.cpu(),
        #         q_mask=query_mask.cpu(),
        #         ori_support_rgb=support_rgb.cpu(),
        #         ori_support_d=support_d.cpu(),
        #         ori_support_th=support_th.cpu(),
        #         s_mask=support_mask.cpu(),
        #         ori_name=name[0] if isinstance(name, (list, tuple)) else str(name),
        #         q_pred=pred_mask.cpu(),
        #         i=idx
        #     )

        # Write evaluation results
        average_meter.write_result('Test', 0)
        miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


def main(model):
    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)

    # Model initialization
    if args.fold == 3:
        sub_list = list(range(0, 15))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        sub_val_list = list(range(15, 20))  # [16,17,18,19,20]
    elif args.fold == 2:
        sub_list = list(range(0, 10)) + list(range(15, 20))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        sub_val_list = list(range(10, 15))  # [6,7,8,9,10]
    elif args.fold == 1:
        sub_list = list(range(0, 5)) + list(range(10, 20))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        sub_val_list = list(range(5, 10))
    elif args.fold == 0:
        sub_list = list(range(5, 20))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        sub_val_list = list(range(0, 5))
    # Device setup

    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    # model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    weights = torch.load(os.path.join(args.load,"best_model.pt"))

    if 'module' in list(weights.keys())[0]:
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
    model.load_state_dict(weights)

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    # Visualizer.initialize(args.visualize)

    # Dataset initialization

    dataloader_val = dataset_mask_val.Tri_Dataset(data_dir=args.datapath, fold=args.fold,
                                                       normalize_mean=[0.3884923, 0.361114, 0.3357993],
                                                       normalize_std=[0.14982404, 0.1512635, 0.16091296],
                                                       normalize_mean_d=[0.9863242, 0.9863242, 0.9863242],
                                                       normalize_std_d=[0.05647239, 0.05647239, 0.05647239],
                                                       normalize_mean_th=[0.40243158, 0.40243158, 0.40243158],
                                                       normalize_std_th=[0.09522554, 0.09522554, 0.09522554],
                                                       mode=args.mode)

    val_sampler = None
    val_loader = torch.utils.data.DataLoader(dataloader_val, batch_size=1, shuffle=False,
                                             num_workers=args.nworker, pin_memory=True, sampler=val_sampler)
    print(len(val_loader))
    # Test HSNet
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, val_loader, sub_val_list, args.nshot)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='./VDT-2048-5i')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=1)
    parser.add_argument('--load', type=str, default='./83.85/')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model', type=str, default='Qwen')
    args = parser.parse_args()

    if (args.model == 'Qwen'):
        Logger.initialize(args, training=True, modelname="Qwen")
        model = Mnet(args.bsz)

    else:
        raise KeyError(f"model version {args.model} not defined")

    # Model initialization

    main(model)
