import torch


def mapping_caption(category_number):
    category_mapping={'motorcycle':1,'truck':2,'bus':3,'traffic light':4,
                      'person':5,'bicycle':6,'car':7}
    if category_number in category_mapping.values():
        caption=next(key for key,value in category_mapping.items() if value==category_number)
        # print(caption)
        return caption


def recaculate_bbx(normalized_bbx,original_image_size,target_image_size):
    # original_image_size=(1560,880)
    # target_image_size=(224,224)
    # normalized_bbx=(x_center,y_center,h,w)
    x_center_orig=normalized_bbx[0]*original_image_size[0]
    y_center_orig=normalized_bbx[1]*original_image_size[1]
    h_orig=normalized_bbx[2]*original_image_size[0]
    w_orig=normalized_bbx[3]*original_image_size[1]
    xmin_orig=x_center_orig-w_orig/2
    ymin_orig=y_center_orig-h_orig/2
    xmax_orig=x_center_orig+w_orig/2
    ymax_orig=y_center_orig+h_orig/2
    # 这里需要看是否需要归一化
    xmin_target=int(xmin_orig*target_image_size[0]/original_image_size[0])
    ymin_target=int(ymin_orig*target_image_size[1]/original_image_size[1])
    xmax_target=int(xmax_orig*target_image_size[0]/original_image_size[0])
    ymax_target=int(ymax_orig*target_image_size[1]/original_image_size[1])
    # print(bbx)
    return torch.tensor([xmin_target,ymin_target,xmax_target,ymax_target])






original_image_size=(1280,720)
target_image_size=(224,224)
normalized_bbx=(0.507422,0.564583,0.011719,0.045833)
out=recaculate_bbx(normalized_bbx,original_image_size,target_image_size)
print(out)


