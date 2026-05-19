from openai import OpenAI
import base64
from common import dataset_text
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def base_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_text(support_name, query_name, sample_class):
    support_rgb = base_image("./VDT-2048-5i/seperated_images/" + support_name.replace('.png', '_rgb.png'))
    support_th = base_image("./VDT-2048-5i/seperated_images/" + support_name.replace('.png', '_th.png'))
    support_d = base_image("./VDT-2048-5i/seperated_images/" + support_name.replace('.png', '_d.png'))
    support = base_image("./VDT-2048-5i/Binary_map/" + str(sample_class.item()) + '/' + support_name)
    query_rgb = base_image("./VDT-2048-5i/seperated_images/" + query_name.replace('.png', '_rgb.png'))
    query_th = base_image("./VDT-2048-5i/seperated_images/" + query_name.replace('.png', '_th.png'))
    query_d = base_image("./VDT-2048-5i/seperated_images/" + query_name.replace('.png', '_d.png'))

    client = OpenAI(
        api_key="",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "You are a useful assistant who can understand pictures."}]},
            {"role": "user", "content": [

                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{support_rgb}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{support_th}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{support_d}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{support}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{query_rgb}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{query_th}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{query_d}"}},

                {"type": "text",
                 "text": "The few-shot saliency object detection task first provides a set of support sets, including RGB, thermal and depth images, which can complement each other in information and have saliency object masks to reflect the saliency objects in this set of images. Then, it is necessary to find the same salient targets as those in the support set in the RGB, thermal and depth images of the query set. Note that the support image has the same saliency goal as the query image.Now, given a set of support sets, including RGB images, thermal images, depth images and their corresponding masks (the first image, the second image, the third image and the fourth image respectively), Please identify the features of the salient objects that are most similar to those in the known support set in the RGB, thermal and depth images of the query set (the fourth image, the fifth image and the sixth image respectively).It is possible to combine RGB images, thermal imaging images and depth images in three complementary ways. Only describe the location, category and features of the salient objects in the query set, without adding any other content or explanations."},
            ],
             }
        ],
    )

    return completion.choices[0].message.content

def generate_O(text):
    client = OpenAI(
        api_key="",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "text",
                 "text": "Extracts the nouns in the given text, outputs only the nouns from the text, and cannot output anything else, given the text:" + text},
            ],
             }
        ],
    )

    return completion.choices[0].message.content

def generate_A(text):
    client = OpenAI(
        api_key="",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "text",
                 "text": "Extracts the adjectives in the given text, outputs only the adjectives from the text, and cannot output anything else, given the text:" + text},
            ],
             }
        ],
    )

    return completion.choices[0].message.content

def generate_R(text):
    client = OpenAI(
        api_key="",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "text",
                 "text": "Extracts prepositions from the given text, outputs only the prepositions from the text, and cannot output anything else, given the text:" + text},
            ],
             }
        ],
    )

    return completion.choices[0].message.content

def text_descripe(support_name, query_name, sample_class):
    # text = generate_text(support_name, query_name, sample_class)
    datapath = '/home/baoliuxin/find/VDT-2048-5i/text1'
    for filename in os.listdir(datapath):
        if filename.endswith('.txt'):  # 仅处理 .txt 后缀的文件
            filepath = os.path.join(datapath, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                print(text)
                O = generate_O(text)
                A = generate_A(text)
                R = generate_R(text)

                query_name = filename.replace('.png.txt', '.txt')
                print(query_name)
                save_to_file('O_', O, query_name)
                save_to_file('A_', A, query_name)
                save_to_file('R_', R, query_name)

    return "yes"


def save_to_file(prefix, content, query_name):
    root = '/home/baoliuxin/find/VDT-2048-5i/label'
    os.makedirs(root, exist_ok=True)

    filename = f"{prefix}{query_name}"
    filepath = os.path.join(root, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved {prefix} result to {filepath}")

if __name__ == '__main__':
    datapath = '/home/baoliuxin/find/VDT-2048-5i'
    batch_size = 1
    fold = 0

    if fold == 3:
        sub_list = list(range(0, 15))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        sub_val_list = list(range(15, 20))  # [16,17,18,19,20]
    elif fold == 2:
        sub_list = list(range(0, 10)) + list(range(15, 20))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        sub_val_list = list(range(10, 15))  # [6,7,8,9,10]
    elif fold == 1:
        sub_list = list(range(0, 5)) + list(range(10, 20))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        sub_val_list = list(range(5, 10))
    elif fold == 0:
        sub_list = list(range(5, 20))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        sub_val_list = list(range(0, 5))

    dataloader_trn = dataset_text.Tri_Dataset(data_dir=datapath, fold=fold)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(dataloader_trn, batch_size=batch_size, shuffle=(train_sampler is None),
                                               num_workers=8, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    for idx, (name) in enumerate(
                train_loader):
        support_name, query_name, sample_class = name[0][0], name[1][0], name[2][0]

        O, A, R = text_descripe(support_name, query_name, sample_class)
        print(support_name,O,A,R)
