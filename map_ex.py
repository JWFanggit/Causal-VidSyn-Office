


def convert_to_absolute(box, img_size, target_img_size):
    """将归一化的box坐标转换为目标图像尺寸下的绝对像素坐标"""
    x_center, y_center, h, w = box
    img_h, img_w = img_size
    target_h, target_w = target_img_size

    # 计算在目标图像尺寸下的中心坐标和宽高
    abs_x_center = x_center * target_w
    abs_y_center = y_center * target_h
    abs_h = h * target_h
    abs_w = w * target_w

    # 左上角和右下角的坐标
    x1 = abs_x_center - abs_w / 2
    y1 = abs_y_center - abs_h / 2
    x2 = abs_x_center + abs_w / 2
    y2 = abs_y_center + abs_h / 2

    return x1, y1, x2, y2



def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 计算交集矩形框的左上角和右下角坐标
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    # 计算交集的宽度和高度
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    # 交集面积
    inter_area = inter_w * inter_h

    # box1和box2的面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 并集面积
    union_area = area1 + area2 - inter_area

    # 计算IoU
    if union_area == 0:
        return 0
    iou = inter_area / union_area

    return iou


# def calculate_iou(pred_box, gt_box):
#     # 计算IoU
#     x_min_pred, y_min_pred, x_max_pred, y_max_pred = pred_box
#     x_min_gt, y_min_gt, x_max_gt, y_max_gt = gt_box
#
#     # 打印框的坐标，检查是否有重叠
#     print(f"Prediction box: {pred_box}")
#     print(f"Ground truth box: {gt_box}")
#
#     # 计算交集宽度和高度
#     inter_width = max(0, min(x_max_pred, x_max_gt) - max(x_min_pred, x_min_gt))
#     inter_height = max(0, min(y_max_pred, y_max_gt) - max(y_min_pred, y_min_gt))
#
#     # 打印交集的宽度和高度
#     print(f"Intersection width: {inter_width}, Intersection height: {inter_height}")
#
#     # 交集面积
#     inter_area = inter_width * inter_height
#
#     # 预测框和真实框的面积
#     pred_area = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
#     gt_area = (x_max_gt - x_min_gt) * (y_max_gt - y_min_gt)
#
#     # 并集面积
#     union_area = pred_area + gt_area - inter_area
#
#     # 计算IoU
#     iou = inter_area / union_area if union_area != 0 else 0
#     return iou
#


def normalize_to_absolute_coords(norm_coords, image_width, image_height):
    # 归一化坐标转绝对坐标
    x_center, y_center, width, height = norm_coords

    print("norm_coords",norm_coords)
    x_center_abs = x_center * image_width
    y_center_abs = y_center * image_height
    width_abs = width * image_width
    height_abs = height * image_height
    # 计算四个角坐标，并确保坐标在图像范围内
    x_min = max(0, x_center_abs - width_abs / 2)
    y_min = max(0, y_center_abs - height_abs / 2)
    x_max = min(image_width, x_center_abs + width_abs / 2)
    y_max = min(image_height, y_center_abs + height_abs / 2)
    return x_min, y_min, x_max, y_max

#
# def calculate_max_iou(abs_box1, box2_list):
#     max_iou = 0
#     for box2 in box2_list:
#         iou = calculate_iou(abs_box1, box2)
#         max_iou = max(max_iou, iou)
#     return max_iou


def compute_iou_between_boxes(box1, box2):
    # """计算两个不同图像尺度下的box的IoU"""
    # 图像1的尺寸
    img_size1 = (800, 800)
    # 图像2的尺寸
    img_size2 = (224, 224)

    # 将两个框都转换到目标图像尺寸 (800, 800)
    target_size = (800, 800)

    # 转换box1到800x800的绝对坐标
    abs_box1 = convert_to_absolute(box1, img_size1, target_size)
    #
    # # 转换box2到800x800的绝对坐标
    abs_box2 = convert_to_absolute(box2, img_size2, target_size)
    # abs_box1=normalize_to_absolute_coords(box1,800,800)
    # print("abs_box1",abs_box1)
    # print("box2",box2)
    # 计算IoU
    # iou = calculate_iou(abs_box1, abs_box2)
    # visualize_boxes(800, 800, abs_box1, box2)
    iou=calculate_iou(abs_box1,abs_box2 )

    if iou > 0:
        return iou
    else:
        return False




# # 归一化的boxes坐标（x_center, y_center, h, w）
# box1 = (0.5, 0.5, 0.2, 0.2)  # 在(800, 800)图像上的box
# box2 = (0.5, 0.5, 0.3, 0.3)  # 在(224, 224)图像上的box
#
# iou = compute_iou_between_boxes(box1, box2)
# print(iou)  # 输出IoU值或者False