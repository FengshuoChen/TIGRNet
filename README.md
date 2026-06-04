# Text-Image Guided Retrieval Network for Triple-modal Images Few-Shot Semantic Segmentation

This repository provides the official implementation of **TIGRNet** for triple-modal few-shot semantic segmentation.

**Text-Image Guided Retrieval Network for Triple-modal Images Few-Shot Semantic Segmentation**  
Fengshuo Chen, Liuxin Bao, Juting Miao, Chao Tao, Xiaofei Zhou

TIGRNet leverages text-image guidance for visible-depth-thermal few-shot semantic segmentation. It introduces a Text Guided Fusion (TGF) module and a dual-branch object retrieval strategy with Image-Guided Retrieval (IGR) and Text-Guided Retrieval (TGR) to improve semantic alignment and cross-modal feature fusion.

## News

- The trained weights are available in `weight.zip`: [Baidu Netdisk](https://pan.baidu.com/s/1-JsxLC8YeGF8Gl0r6EZ6iw) (code: `TIGR`).
- The 1-shot prediction maps are available in `1-shot_predictions.zip`: [Baidu Netdisk](https://pan.baidu.com/s/1r9-v7f6iHMEDqEWKQlvMmQ?pwd=TIGR) (code: `TIGR`).

## Prerequisites

The code is implemented with PyTorch. Please install the common dependencies used by the training and testing scripts:

```bash
pip install torch torchvision opencv-python pillow numpy tqdm tensorboardX thop torchinfo fvcore
```

This project also uses CLIP and a SAM image encoder. Please make sure the required CLIP/SAM dependencies and checkpoints are available before training or testing.

The SAM checkpoint path is loaded in `1-shot/sam_ori/network/sammus.py` and `5-shot/sam_ori/network/sammus.py`. Please update the path to your local `sam_vit_b_01ec64.pth` before running the scripts.

## Dataset

The experiments are conducted on the `VDT-2048-5i` dataset. The expected dataset directory contains triple-modal images, binary masks, split files, and text descriptions:

```text
VDT-2048-5i/
  seperated_images/
    *_rgb.png
    *_th.png
    *_d.png
  Binary_map/
    split0.txt
    split1.txt
    split2.txt
    split3.txt
    1/
    2/
    ...
    20/
  label_ps/
    O_*.txt
    A_*.txt
    R_*.txt
  text1/
    *.txt
```

`text1/` stores the VLM-generated semantic descriptions, while `label_ps/` stores the object, attribute, and relation prior text files used by the text-guided branch.

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/FengshuoChen/TIGRNet.git
cd TIGRNet
```

### 2. Prepare weights

Download the trained model file:

- `weight.zip`: [Baidu Netdisk](https://pan.baidu.com/s/1-JsxLC8YeGF8Gl0r6EZ6iw) (code: `TIGR`)

Extract the archive and place each checkpoint directory where it can be passed to `--load`. The testing scripts expect a `best_model.pt` file under the load directory.

Example:

```text
TIGRNet/
  weights/
    1-shot/
      fold0/
        best_model.pt
    5-shot/
      fold0/
        best_model.pt
```

### 3. Training

Train the 1-shot model:

```bash
cd 1-shot
python train.py --datapath /path/to/VDT-2048-5i --fold 0 --bsz 1 --lr 1e-4 --epoch 150
```

Train the 5-shot model:

```bash
cd 5-shot
python train.py --datapath /path/to/VDT-2048-5i --fold 0 --shot 5 --bsz 1 --lr 1e-4 --epoch 50
```

Training logs and checkpoints are saved under `logs/`.

### 4. Testing

Evaluate the 1-shot model:

```bash
cd 1-shot
python test.py --datapath /path/to/VDT-2048-5i --fold 0 --nshot 1 --load ../weights/1-shot/fold0
```

Evaluate the 5-shot model:

```bash
cd 5-shot
python test.py --datapath /path/to/VDT-2048-5i --fold 0 --shot 5 --load ../weights/5-shot/fold0
```

The scripts report mIoU and FB-IoU for the selected fold. Use `--fold 0`, `--fold 1`, `--fold 2`, and `--fold 3` for four-fold evaluation.

### 5. Prediction Maps

We provide the 1-shot prediction maps:

- `1-shot_predictions.zip`: [Baidu Netdisk](https://pan.baidu.com/s/1r9-v7f6iHMEDqEWKQlvMmQ?pwd=TIGR) (code: `TIGR`)


## Contact

If you have any questions, please contact:

- Fengshuo Chen: `23061614@hdu.edu.cn`
- Liuxin Bao: `lxbao@hdu.edu.cn`
