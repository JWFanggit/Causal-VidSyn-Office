import cv2
import numpy as np

def save_frames_with_boxes(video_frames, boxes, output_dir):
    idx=0
    for frame, frame_boxes in zip(video_frames,boxes):
        frame = frame.transpose(1, 2, 0)
        img = frame.copy()

        for box in frame_boxes:
            xmin, ymin, xmax, ymax = box
            if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
                # No bounding box coordinates for this object in this frame, so skip it.
                continue
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # 在图像上绘制边界框

        frame_filename = f"{output_dir}/frame_{idx:04d}.png"  # 创建文件名，例如 frame_0000.png
        cv2.imwrite(frame_filename, img)  # 保存当前帧为图像文件
        idx+=1
    print("Frames with bounding boxes saved successfully.")

# 示例用法
output_dir = "./output_frames"  # 创建一个目录来保存帧
video_frames = np.random.randint(0, 255, (16, 3, 224, 224), dtype=np.uint8)  # Example video frames.
bounding_boxes = np.random.randint(0, 224, (16, 4, 4))  # Example bounding box coordinates.

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_frames_with_boxes(video_frames, bounding_boxes, output_dir)