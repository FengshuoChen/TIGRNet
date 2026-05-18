import os
import random
from PIL import Image

class Tri_Dataset(object):
    def __init__(self, data_dir, fold, input_size=[512, 512], prob=0.7):
        # ------------------- load data list -------------------
        self.data_dir = data_dir
        self.new_exist_class_list = self.get_new_exist_class_dict(fold)
        self.binary_pair_list = self.get_binary_pair_list()
        self.input_size = input_size
        self.prob = prob
        self.split = fold

        # 定义子集划分
        if self.split == 3:
            self.sub_list = list(range(1, 16))
        elif self.split == 2:
            self.sub_list = list(range(1, 11)) + list(range(16, 21))
        elif self.split == 1:
            self.sub_list = list(range(1, 6)) + list(range(11, 21))
        elif self.split == 0:
            self.sub_list = list(range(6, 21))

    def get_new_exist_class_dict(self, fold):
        """读取除当前 fold 外的 split 列表"""
        new_exist_class_list = []
        fold_list = [0, 1, 2, 3]
        # fold_list.remove(fold)
        for f in fold_list:
            file_path = os.path.join(self.data_dir, 'Binary_map', f'split{f}.txt')
            with open(file_path) as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("_")
                    img_name = parts[0]
                    cat = int(parts[1])
                    new_exist_class_list.append([img_name, cat])
        return new_exist_class_list

    def get_binary_pair_list(self):
        """读取每个类别的图像列表"""
        binary_pair_list = {}
        for cls in range(1, 21):
            txt_path = os.path.join(self.data_dir, 'Binary_map', f'{cls}.txt')
            with open(txt_path) as f:
                binary_pair_list[cls] = [line.split()[0] for line in f if line.strip()]
        return binary_pair_list

    def __getitem__(self, index):
        """保持原采样逻辑，只输出 name"""
        # 1️⃣ 选 query 图像与类别
        query_name = self.new_exist_class_list[index][0]
        sample_class = self.new_exist_class_list[index][1]

        # 2️⃣ 随机选 support 图像（不同于 query）
        support_img_list = self.binary_pair_list[sample_class]
        while True:
            support_name = random.choice(support_img_list)
            if support_name != query_name:
                break

        # 3️⃣ 构造路径
        support_name_rgb = support_name.replace('.png', '_rgb.png')
        support_name_th = support_name.replace('.png', '_th.png')
        support_name_d = support_name.replace('.png', '_d.png')

        query_name_rgb = query_name.replace('.png', '_rgb.png')
        query_name_th = query_name.replace('.png', '_th.png')
        query_name_d = query_name.replace('.png', '_d.png')

        # 4️⃣ 模拟读取（仅检查存在，不进行任何图像处理）
        Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_rgb))
        Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_th))
        Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_d))
        Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_rgb))
        Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_d))

        # 5️⃣ 仅输出 name
        name = [support_name, query_name, sample_class]
        return name

    def __len__(self):
        return len(self.new_exist_class_list)
