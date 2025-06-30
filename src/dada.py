import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from einops import rearrange


class DADA2KS3(Dataset):
    def __init__(self, root_path, interval, phase,
                 data_aug=False):
        self.root_path = root_path
        self.interval = interval
        # self.transforms = transforms
        self.data_aug = data_aug
        self.fps = 30
        self.phase = phase
        self.data_list, self.tai, self.tco, self.text= self.get_data_list()
        self.bbx_path = r".../labels_30fps_diffdet_inference/Cap_label"
        self.qa_path=r".../all_data.pth"
        self.map_path=r".../DADA-CAP-fixation"
        self.qa_data=self.load_qa()
    def load_qa(self):
        loaded_data = torch.load(self.qa_path)
        return loaded_data

    def get_data_list(self):
        if self.phase == "train":
            list_file = os.path.join(self.root_path + "/" + 'training_sg2.txt')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs,tais, tcos,texts= [], [], [], []
            with open(list_file, 'r', encoding='utf-8') as f:
                for ids, line in enumerate(f.readlines()):
                    parts = line.strip().split(',[')
                    ID,label,tai,tco=parts[0].split(' ')
                    fileIDs.append(ID)
                    tais.append(tai)
                    tcos.append(tco)
                    texts.append(parts[1])
            return fileIDs, tais, tcos,texts
        if self.phase == "val":
            list_file = os.path.join(self.root_path + "/" + 'test.txt')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs, tais, tcos, texts = [], [], [], []
            with open(list_file, 'r', encoding='utf-8') as f:
                for ids, line in enumerate(f.readlines()):
                    parts = line.strip().split(',[')
                    ID, label, tai, tco = parts[0].split(' ')
                    fileIDs.append(ID)
                    tais.append(tai)
                    tcos.append(tco)
                    texts.append(parts[1])
            return fileIDs, tais, tcos, texts

    def __len__(self):
        return len(self.data_list)

    def pross_video_data(self, video):
        video_datas = []
        for fid in range(len(video)):
            video_data = video[fid]
            video_data = Image.open(video_data)
            video_data = video_data.resize((224, 224))
            video_data = np.asarray(video_data, np.float32)
            if len(video_data.shape) <3:
                video_data = np.stack((video_data,video_data,video_data),-1)
            video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
        video_data = rearrange(video_data, 'f w h c -> f c w h')
        return video_data

    def read_nomarl_rgbvideo(self, video_file):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data

        video_data = self.pross_video_data(video_file)

        return video_data


    def pdbbx(self, bbx, max_N):
        N = bbx.shape[0]
        if N < max_N:
            pad_objects = torch.zeros(max_N - N, 4)
            bbx = torch.cat([bbx, pad_objects], dim=0)
        elif N > max_N:
            bbx = bbx[:max_N, :]

        return bbx

    def to_valid(self,x0, y0, x1, y1, image_size, min_box_size):
        valid = True

        if x0 > image_size or y0 > image_size or x1 < 0 or y1 < 0:
            valid = False  # no way to make this box vide, it is completely cropped out
            return valid, (None, None, None, None)

        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, image_size)
        y1 = min(y1, image_size)

        if (x1 - x0) * (y1 - y0) / (image_size * image_size) < min_box_size:
            valid = False
            return valid, (None, None, None, None)

        return valid, (x0, y0, x1, y1)

    def recalculate_box_and_verify_if_valid(self, normalized_bbx, original_image_size, target_image_size,image_size):
        # normalized_bbx = (x1, y1, x2, y2)
        x1_orig, y1_orig, x2_orig, y2_orig = normalized_bbx
        # Scale coordinates from original image size to target image size
        x1_target = x1_orig * (target_image_size[0] / original_image_size[0])
        y1_target = y1_orig * (target_image_size[1] / original_image_size[1])
        x2_target = x2_orig * (target_image_size[0] / original_image_size[0])
        y2_target = y2_orig * (target_image_size[1] / original_image_size[1])
        valid, (x0, y0, x1, y1) = self.to_valid(x1_target,y1_target,x2_target,y2_target,image_size, min_box_size=0.01)

        # if valid:
        #     # we also perform random flip.
        #     # Here boxes are valid, and are based on image_size
        #     # if trans_info["performed_flip"]:
        #         x0, x1 = image_size - x1, image_size - x0

        return valid, (x0, y0, x1, y1)

    def mapping_caption(self, category_number):
        category_mapping = {'{}': 0, 'motorcycle': 1, 'truck': 2, 'bus': 3, 'traffic light': 4,
                            'person': 5, 'bicycle': 6, 'car': 7}
        if category_number in category_mapping.values():
            caption = next(key for key, value in category_mapping.items() if value == category_number)
            return caption

    def extract_masks(self, video_frames, bounding_boxes):
        masks = []
        for i in range(video_frames.shape[0]):
            frame = video_frames[i]
            frame_masks = np.zeros_like(frame)  # Create a blank mask with the same dimensions as the frame

            for j in range(bounding_boxes.shape[1]):
                bbx = bounding_boxes[i, j].int()  # Convert the bounding box coordinates to integers
                x1, y1, x2, y2 = bbx
                frame_masks[..., y1:y2, x1:x2] = frame[..., y1:y2, x1:x2]  # 复制 bounding box 区域内的像素到 mask
            masks.append(frame_masks)
        return np.array(masks)

    def bbx_caption_process(self, bbx_info, original_image_size, target_image_size, max_N,image_size):
        caption = bbx_info[:, 0]  # Add a new dimension
        caption_text = [self.mapping_caption(category_number.item()) for category_number in caption]
        caption_text = ", ".join(caption_text)
        # # Extract the second to fifth elements of each row and keep them as (N, 4) tensor
        bbx = bbx_info[:, 1:]
        areas=[]
        all_boxes=[]
        all_boxes_ = []
        all_masks=[]
        for i in range(bbx.shape[0]):
            row_bbx = bbx[i]
            valid, (x0, y0, x1, y1) = self.recalculate_box_and_verify_if_valid(row_bbx, original_image_size, target_image_size,image_size)
            if valid:
                areas.append((x1 - x0) * (y1 - y0))
                all_boxes.append(torch.tensor([x0, y0, x1, y1]) / image_size)  # scale to 0-1
                all_boxes_.append(torch.tensor([x0, y0, x1, y1]))
                all_masks.append(1)
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[0:max_N]
        new_boxes = torch.zeros(max_N, 4)
        new_boxes_ = torch.zeros(max_N, 4)
        masks = torch.zeros(max_N)
        for i, idx in enumerate(wanted_idxs):
            new_boxes[i] = all_boxes[idx]
            new_boxes_[i]=all_boxes_[idx]
            masks[i] = all_masks[idx]
        new_bbx = self.pdbbx(new_boxes, max_N)
        new_bbx_=self.pdbbx(new_boxes_, max_N)
        image_masks = masks
        text_masks = masks
        return new_bbx,caption_text,image_masks,text_masks,new_bbx_

    def gather_info(self, index):
        # accident_id = int(self.data_list[index].split('/')[0])
        accident_id = self.data_list[index]
        video_id = int(self.data_list[index].split('/')[1])
        catagroy=int(self.data_list[index].split('/')[0])
        text = self.text[index]
        return accident_id, video_id, text,catagroy
    def __getitem__(self, index):
        # read RGB video (trimmed)
        tco = int(self.tco[index])
        video_path = os.path.join(self.root_path+"/",self.data_list[index]+"/"+"images")
        maps_path = os.path.join(self.map_path + "/", self.data_list[index] + "/" + "maps")
        start_frame =tco - 16
        v_r = [video_path + "/" + f'{i:06d}' + ".jpg" for i in range(start_frame, start_frame + 16)]
        m_r= [maps_path + "/" + f'{i:06d}' + ".png" for i in range(start_frame-1, start_frame + 15)]
        accident_id, video_id,text,catagroy= self.gather_info(index)
        option_token=self.qa_data[video_id]['option_token']
        question_token=self.qa_data[video_id]['answer_token']
        answer_id=self.qa_data[video_id]['answer_id']
        all_ids = list(range(5))
        gt_option =option_token[answer_id,:]
        all_ids.remove(answer_id)
        non_gt_option_token = option_token[all_ids,:]
        vr = self.read_nomarl_rgbvideo(v_r)
        mr = self.read_nomarl_rgbvideo(m_r)
        bbx_info_list=[]
        bbx_info_list_ = []
        image_mask=[]
        text_mask=[]
        bbx_path=os.path.join(self.bbx_path+"/",self.data_list[index])
        bbx_path = [bbx_path + "/" + f'{i:06d}' + ".json" for i in range(tco-16,tco)]
        for bbx_file in bbx_path:
            with open(bbx_file) as json_file:
                lines = json.load(json_file)
                if not lines or len(lines) == 0 or all(line.isspace() for line in lines):
                    filtered_datas = torch.zeros(1, 5, dtype=torch.float32)
                else:
                    bbx_info = lines["bboxes"]
                    scores = lines["scores"]
                    label = lines["labels"]
                    filtered_data = [[lbl, *info] for info, scr, lbl in zip(bbx_info, scores, label) if scr > 0.2]
                    if not filtered_data or len(filtered_data) == 0:
                        filtered_datas = torch.stack(
                            [torch.zeros(1, 5, dtype=torch.float32) for _ in range(16)]).squeeze(1)
                    else:
                        filtered_datas = torch.stack(
                            [torch.tensor(list(map(float, line)), dtype=torch.float32) for line in filtered_data])


                new_bbx,caption_text,image_masks,text_masks,new_bbx_ = self.bbx_caption_process(filtered_datas, original_image_size=(1560, 660),
                                                        target_image_size=(224, 224), max_N=10,image_size=224)

                bbx_info_list.append(new_bbx)
                bbx_info_list_.append(new_bbx_)
                image_mask.append(image_masks)
                text_mask.append(text_masks)
        boxes = torch.stack(bbx_info_list)
        boxes_=torch.stack(bbx_info_list_)
        mask=self.extract_masks(vr,boxes_)
        example = {
        "pixel_values": vr / 127.5 - 1.0,
        "pixel_map_values": mr/ 255,
        "prompt": text,
        "bbx": boxes,
        "mask":mask / 127.5 - 1.0,
        "answer_id": answer_id,
        "question":question_token,
        "option":gt_option,
        "accident_id":video_id,
        "caption": caption_text,
        "cat":catagroy,
        "non_option_token":non_gt_option_token,
        }
        return example








class DADA2KS3_double(Dataset):
    def __init__(self, root_path, interval, phase,
                 data_aug=False):
        self.root_path = root_path
        self.interval = interval
        # self.transforms = transforms
        self.data_aug = data_aug
        self.fps = 30
        self.phase = phase
        self.data_list, self.tai, self.tco, self.text,self.p_text= self.get_data_list()
        self.bbx_path = r".../labels_30fps_diffdet_inference/Cap_label"
        self.qa_path=r".../all_data.pth"
        self.map_path=r"../Drive-Gaze"
        self.qa_data=self.load_qa()

    def load_qa(self):
        loaded_data = torch.load(self.qa_path)
        return loaded_data


    def get_data_list(self):
        if self.phase == "train":
            # list_file = os.path.join(self.root_path + "/" + 'OOD_train.txt')
            list_file = os.path.join(self.root_path + "/" + 'train_RPFD.txt')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs,tais, tcos,texts,p_texts= [], [], [], [],[]
            with open(list_file, 'r', encoding='utf-8') as f:
                for ids, line in enumerate(f.readlines()):
                    parts = line.strip().split(',[')
                    ID,label,tai,tco=parts[0].split(' ')
                    fileIDs.append(ID)
                    tais.append(tai)
                    tcos.append(tco)
                    texts.append(parts[1].split('[')[0])
                    p_texts.append(parts[1].split('[')[1])
            return fileIDs, tais, tcos,texts,p_texts
        if self.phase == "val":
            list_file = os.path.join(self.root_path + "/" + 'test.txt')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs, tais, tcos, texts = [], [], [], []
            with open(list_file, 'r', encoding='utf-8') as f:
                for ids, line in enumerate(f.readlines()):
                    parts = line.strip().split(',[')
                    ID, label, tai, tco = parts[0].split(' ')
                    fileIDs.append(ID)
                    tais.append(tai)
                    tcos.append(tco)
                    texts.append(parts[1])
            return fileIDs, tais, tcos, texts

    def __len__(self):
        return len(self.data_list)
    def pross_video_data(self, video):
        video_datas = []
        for fid in range(len(video)):
            video_data = video[fid]
            video_data = Image.open(video_data)
            video_data = video_data.resize((224, 224))
            video_data = np.asarray(video_data, np.float32)
            if len(video_data.shape) <3:
                video_data = np.stack((video_data,video_data,video_data),-1)
            video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
        video_data = rearrange(video_data, 'f w h c -> f c w h')
        return video_data

    def read_nomarl_rgbvideo(self, video_file):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data
        video_data = self.pross_video_data(video_file)
        return video_data

    def pdbbx(self, bbx, max_N):
        N = bbx.shape[0]
        if N < max_N:
            pad_objects = torch.zeros(max_N - N, 4)
            bbx = torch.cat([bbx, pad_objects], dim=0)
        elif N > max_N:
            bbx = bbx[:max_N, :]

        return bbx

    def to_valid(self,x0, y0, x1, y1, image_size, min_box_size):
        valid = True

        if x0 > image_size or y0 > image_size or x1 < 0 or y1 < 0:
            valid = False  # no way to make this box vide, it is completely cropped out
            return valid, (None, None, None, None)

        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, image_size)
        y1 = min(y1, image_size)

        if (x1 - x0) * (y1 - y0) / (image_size * image_size) < min_box_size:
            valid = False
            return valid, (None, None, None, None)

        return valid, (x0, y0, x1, y1)

    def recalculate_box_and_verify_if_valid(self, normalized_bbx, original_image_size, target_image_size,image_size):
        # normalized_bbx = (x1, y1, x2, y2)
        x1_orig, y1_orig, x2_orig, y2_orig = normalized_bbx
        # Scale coordinates from original image size to target image size
        x1_target = x1_orig * (target_image_size[0] / original_image_size[0])
        y1_target = y1_orig * (target_image_size[1] / original_image_size[1])
        x2_target = x2_orig * (target_image_size[0] / original_image_size[0])
        y2_target = y2_orig * (target_image_size[1] / original_image_size[1])
        valid, (x0, y0, x1, y1) = self.to_valid(x1_target,y1_target,x2_target,y2_target,image_size, min_box_size=0.01)

        return valid, (x0, y0, x1, y1)












    def mapping_caption(self, category_number):
        category_mapping = {'{}': 0, 'motorcycle': 1, 'truck': 2, 'bus': 3, 'traffic light': 4,
                            'person': 5, 'bicycle': 6, 'car': 7}
        if category_number in category_mapping.values():
            caption = next(key for key, value in category_mapping.items() if value == category_number)
            # print(caption)
            return caption
    #
    #
    def extract_masks(self, video_frames, bounding_boxes):
        masks = []
        for i in range(video_frames.shape[0]):
            frame = video_frames[i]
            frame_masks = np.zeros_like(frame)  # Create a blank mask with the same dimensions as the frame

            for j in range(bounding_boxes.shape[1]):
                bbx = bounding_boxes[i, j].int()  # Convert the bounding box coordinates to integers
                x1, y1, x2, y2 = bbx
                frame_masks[..., y1:y2, x1:x2] = frame[..., y1:y2, x1:x2]  # 复制 bounding box 区域内的像素到 mask

            masks.append(frame_masks)
        return np.array(masks)
    #
    def bbx_caption_process(self, bbx_info, original_image_size, target_image_size, max_N,image_size):
        caption = bbx_info[:, 0]  # Add a new dimension
        caption_text = [self.mapping_caption(category_number.item()) for category_number in caption]
        caption_text = ", ".join(caption_text)
        # # Extract the second to fifth elements of each row and keep them as (N, 4) tensor
        bbx = bbx_info[:, 1:]
        # bbx = bbx_info
        # new_bbx = torch.zeros_like(bbx)
        areas=[]
        all_boxes=[]
        all_boxes_ = []
        all_masks=[]
        for i in range(bbx.shape[0]):
            row_bbx = bbx[i]
            valid, (x0, y0, x1, y1) = self.recalculate_box_and_verify_if_valid(row_bbx, original_image_size, target_image_size,image_size)
            if valid:
                areas.append((x1 - x0) * (y1 - y0))
                all_boxes.append(torch.tensor([x0, y0, x1, y1]) / image_size)  # scale to 0-1
                all_boxes_.append(torch.tensor([x0, y0, x1, y1]))
                all_masks.append(1)
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[0:max_N]
        new_boxes = torch.zeros(max_N, 4)
        new_boxes_ = torch.zeros(max_N, 4)
        masks = torch.zeros(max_N)

        for i, idx in enumerate(wanted_idxs):
            new_boxes[i] = all_boxes[idx]
            new_boxes_[i]=all_boxes_[idx]
            masks[i] = all_masks[idx]
        # processed_row_bbx[processed_row_bbx < 0] = 0
        # new_bbx[i] = processed_row_bbx
        new_bbx = self.pdbbx(new_boxes, max_N)
        new_bbx_=self.pdbbx(new_boxes_, max_N)
        image_masks = masks
        text_masks = masks

        return new_bbx,caption_text,image_masks,text_masks,new_bbx_

    def gather_info(self, index):
        # accident_id = int(self.data_list[index].split('/')[0])
        accident_id = self.data_list[index]
        video_id = int(self.data_list[index].split('/')[1])
        catagroy=int(self.data_list[index].split('/')[0])
        text = self.text[index]
        p_text=self.p_text[index]
        return accident_id, video_id, text,catagroy,p_text


    def __getitem__(self, index):
        # read RGB video (trimmed)

        tai = int(self.tai[index])
        tco = int(self.tco[index])
        # print(tco)
        video_path = os.path.join(self.root_path+"/",self.data_list[index]+"/"+"images")
        maps_path = os.path.join(self.map_path + "/", self.data_list[index] + "/" + "maps")
        # start_frame = random.randint(tco - 27, tco - 16)
        start_frame =tco - 16
        v_r = [video_path + "/" + f'{i:06d}' + ".jpg" for i in range(start_frame, start_frame + 16)]
        m_r= [maps_path + "/" + f'{i:06d}' + ".png" for i in range(start_frame-1, start_frame + 15)]
        accident_id, video_id,text,catagroy,p_text= self.gather_info(index)
        option_token=self.qa_data[video_id]['option_token']
        question_token=self.qa_data[video_id]['answer_token']
        answer_id=self.qa_data[video_id]['answer_id']
        all_ids = list(range(5))
        gt_option =option_token[answer_id,:]
        all_ids.remove(answer_id)
        non_gt_option_token = option_token[all_ids,:]
        vr = self.read_nomarl_rgbvideo(v_r)
        flip_vr=vr[::-1]
        mr = self.read_nomarl_rgbvideo(m_r)
        flip_mr=mr[::-1]
        bbx_info_list=[]
        bbx_info_list_ = []
        image_mask=[]
        text_mask=[]
        bbx_path=os.path.join(self.bbx_path+"/",self.data_list[index])
        bbx_path = [bbx_path + "/" + f'{i:06d}' + ".json" for i in range(tco-16,tco)]
        for bbx_file in bbx_path:
            with open(bbx_file) as json_file:
                lines = json.load(json_file)
                if not lines or len(lines) == 0 or all(line.isspace() for line in lines):

                    filtered_datas = torch.zeros(1, 5, dtype=torch.float32)
                else:

                    bbx_info = lines["bboxes"]
                    scores = lines["scores"]
                    label = lines["labels"]
                    # filtered_bbx_info = [info for info, s in zip(bbx_info,scores) if s > 0.3]
                    filtered_data = [[lbl, *info] for info, scr, lbl in zip(bbx_info, scores, label) if scr > 0.2]
                    if not filtered_data or len(filtered_data) == 0:
                        filtered_datas = torch.stack(
                            [torch.zeros(1, 5, dtype=torch.float32) for _ in range(16)]).squeeze(1)
                    else:
                        filtered_datas = torch.stack(
                            [torch.tensor(list(map(float, line)), dtype=torch.float32) for line in filtered_data])
                new_bbx,caption_text,image_masks,text_masks,new_bbx_ = self.bbx_caption_process(filtered_datas, original_image_size=(1560, 660),
                                                        target_image_size=(224, 224), max_N=10,image_size=224)
                bbx_info_list.append(new_bbx)
                bbx_info_list_.append(new_bbx_)
                image_mask.append(image_masks)
                text_mask.append(text_masks)
        boxes = torch.stack(bbx_info_list)
        boxes_=torch.stack(bbx_info_list_)
        mask=self.extract_masks(vr,boxes_)
        example = {
        "pixel_values": vr / 127.5 - 1.0,
        "flip_pixel_values":flip_vr/ 127.5 - 1.0,
        "pixel_map_values": mr/ 255,
        "flip_pixel_map_values":flip_mr/255,
        "prompt": text,
        "p_prompt":p_text,
        "bbx": boxes,
        "mask":mask / 127.5 - 1.0,
        "answer_id": answer_id,
        "question":question_token,
        "option":gt_option,
        "accident_id":video_id,
        "caption": caption_text,
        "cat":catagroy,
        "non_option_token":non_gt_option_token,
        }
        return example

