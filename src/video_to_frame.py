import cv2
import os
from math import ceil
import random
#
def split_video_to_frames(video_path, output_dir,idx):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的总帧数和帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算每个输出视频的帧数
    # frames_per_output = ceil(total_frames / target_frames)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # # 遍历视频,并保存每个输出视频帧
    # for i in range(target_frames):
    #     start_frame = i * frames_per_output
    #     end_frame = min((i + 1) * frames_per_output, total_frames)
    #
    #     # 设置视频的当前帧位置
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if (total_frames-16)-180 > 0:
        start_frame = random.randint(180, total_frames - 16)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # 创建输出目录
        output_subdir = os.path.join(output_dir, f"{idx}")
        os.makedirs(output_subdir, exist_ok=True)
        print(start_frame)
    # 保存每个帧为图像文件
        for frame_idx in range(start_frame,start_frame+16):
            ret, frame = cap.read()
            if ret:
                output_path = os.path.join(output_subdir, f"frame_{frame_idx - start_frame + 1}.png")
                cv2.imwrite(output_path, frame)

    # 释放视频捕获对象
        cap.release()
#
# split_video_to_frames(r"/media/work/TOSHIBA EXT/OpenDV/DriveAGI-main/opendv/OpenDV-YouTube/videos/4K_DRIVE/-0qvF0wddpo.webm", r"/media/work/TOSHIBA EXT/save_opendv",idx=1)
output_dir=r"/media/work/TOSHIBA EXT/save_opendv"
root_path=r"/media/work/TOSHIBA EXT/OpenDV/DriveAGI-main/opendv/OpenDV-YouTube/videos"
idx=0
for i in range(0,3):
    for path in os.listdir(root_path):
        video_path=os.path.join(root_path,path)
        for video_name in os.listdir(video_path):
            video=os.path.join(video_path,video_name)
            idx+=1
            try:
                split_video_to_frames(video,output_dir,idx)
            except Exception as e:
                print(f"error path:{video}")
                pass


