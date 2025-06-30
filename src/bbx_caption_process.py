# from mapping_caption import mapping_caption,recaculate_bbx
# from Fourier import PositionNet
# #(n,4)--(N,4)--(stack)
#
# import torch
#
# # 定义 recaculate_bbx 函数
# def recaculate_bbx(normalized_bbx, original_image_size, target_image_size):
#     # 你的 recaculate_bbx 函数的实现
#     # ...
# # 假设你有一个形状为 (B, N, 4) 的 boxes 张量，其中 B 是批量大小，N 是当前的 bbox 数目
# # self.max_N 是目标的 bbox 数目
# B = 3  # 例如，批量大小为 3
# N = 5  # 例如，当前的 bbox 数目为 5
# max_N = 7  # 目标的 bbox 数目为 7
# original_image_size = (1560, 880)
# target_image_size = (224, 224)
#
# # 创建示例的 boxes 张量
# boxes = torch.rand(B, N, 4)  # 示例的随机 bbox 信息
#
# # Step 1: 使用 "match max_N 模块" 填充或截断 boxes
# if N < max_N:
#     pad_objects = torch.zeros(B, max_N - N, 4).to(boxes.device)
#     boxes = torch.cat([boxes, pad_objects], dim=1)
# elif N > max_N:
#     boxes = boxes[:, :max_N, :]
#
# # Step 2: 对每一行的 boxes 信息应用 recaculate_bbx 函数
# new_boxes = []
# for b in range(B):
#     row_boxes = boxes[b]  # 获取当前批次的 boxes
#     recalculated_row_boxes = torch.stack([recaculate_bbx(bbx, original_image_size, target_image_size) for bbx in row_boxes], dim=0)
#     new_boxes.append(recalculated_row_boxes)
# # 将 new_boxes 转换为张量
# new_boxes = torch.stack(new_boxes, dim=0)
# # 输出 new_boxes，这将包含填充或截断后以及重新计算大小后的 bbox 信息
# print(new_boxes)
#
#
#
#
#
#
#
#
#
#
#
#
#
