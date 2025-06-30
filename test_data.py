from mydataset import DADA
import torch
import numpy as np
from map_ex import compute_iou_between_boxes
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import re

def get_map_boxes(attention_map):
    # 假设 attention_map 是你的(h, w, c)的注意力图，取其中一个通道，例如 attention_map[:,:,0]
    attention_map = attention_map[:,:,0]  # 如果有多个通道，这里选择第一个通道
    orig_h=attention_map.shape[0]
    orig_w=attention_map.shape[1]
    non_zero_indices = np.argwhere(attention_map > 0)
    # 如果有非零点，获取最小和最大的 x, y 坐标
    if non_zero_indices.size > 0:
        y_min, x_min = non_zero_indices.min(axis=0)
        y_max, x_max = non_zero_indices.max(axis=0)
    else:
        raise ValueError("No non-zero regions found in the attention map")
    # 计算 box 的宽高以及中心点
    w = x_max - x_min
    h = y_max - y_min
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    # 得到 x_center, y_center, w, h 的 bounding box
    boxes =torch.tensor([x_center/orig_w, y_center/orig_h, w/orig_w, h/orig_h])



    return boxes
#
# 1. 将map_boxes从特征图的尺度转换到image的尺度
def convert_to_image_scale(box, map_size, image_size):
    x_center, y_center, width, height = box
    scale_x=image_size[0]/map_size[0]
    scale_y=image_size[1]/map_size[1]
    # 按比例缩放 x 和 y 坐标，以及宽高
    x_center_new = x_center *  scale_x
    y_center_new = y_center * scale_y
    width_new = width * (image_size[0] / map_size[0])
    height_new = height * (image_size[1] / map_size[1])

    return torch.tensor([x_center_new, y_center_new, width_new, height_new])

def fix_hyphen_spacing(text):
    return re.sub(r'\s*-\s*','-',text)

split_marker='resulting in'

def process_text(text):
    if split_marker in text:
        return text.split(split_marker)[-1].strip()
    return text

if __name__=="__main__":
    total_phrases = 0
    correct_phrases = 0
    r_iou=0
    model = load_model("./groundingdino/config/GroundingDINO_SwinT_OGC.py",
                       "./groundingdino_swint_ogc.pth")
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    image_size = (800, 800)  # (width, height)
    map_size = (224, 224)  # (width, height)
    val_dataset = DADA(root_path=r"./generated videos",map_path=r"./DADA2000")
    # device = torch.device("cuda", 0)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        pin_memory=True, num_workers=4, drop_last=True)
    clip_scores = []
    for idx, batch in enumerate(val_dataloader):
        # device = torch.device("cuda", 0)
        prompts = batch["prompt"]
        g_v = batch["g_pixel_values"]
        m_r=batch["pixel_map_values"]
        prompt_c=prompts[0]
        prompt_c = process_text(prompt_c)
        # prompt_c = prompt_c.split(" ")[-1]
        for i in range(16):
            gg_v=g_v[i][0]
            m_v_path=m_r[i][0]
            image_source, image = load_image(gg_v)
            map_source, map = load_image(m_v_path)
            map_source = cv2.resize(map_source, (224, 224), interpolation=cv2.INTER_LINEAR)
            map_boxess = get_map_boxes(map_source)
            map_boxess = convert_to_image_scale(map_boxess, map_size, image_size)
            map_boxes = map_boxess.unsqueeze(0).to(dtype=torch.float32)
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=prompt_c,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
            phrases = [fix_hyphen_spacing(text) for text in phrases]
            if len(boxes)>0:
                for box in boxes:
                    iou = compute_iou_between_boxes(tuple(box.squeeze(0).tolist()),
                                                tuple(map_boxes.squeeze(0).tolist()))
                    if iou>0:
                        iou_found=True
                        r_iou += 1
                        break
            result = any(word in prompt_c for word in phrases)
            total_phrases += 1
            if result == True:
                correct_phrases += 1
    print(correct_phrases)
    print(total_phrases)
    if total_phrases > 0:
        accuracy = correct_phrases / total_phrases
        affordance=r_iou / total_phrases
    else:
        accuracy = 0
        affordance=0
    print(f"accuracy:{accuracy:.2%}")
    print(f"accuracy:{affordance:.2%}")


