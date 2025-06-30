import os
import random
# import h5py
import numpy as np
import cv2
import torch
# from fid import calculate_fid
from torchvision import models, transforms
# from scripts.compute_fvd import caculate_fvd
# from clip_score import caculate_clip
# import clip
import json
from torchvision import transforms
from einops import rearrange, repeat, reduce
import glob
from PIL import Image
# import decord
# decord.bridge.set_bridge('torch')
# from norm import norm,norm1

from torch.utils.data import Dataset,DataLoader
from einops import rearrange


class InceptionV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = torch.nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = torch.nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = torch.nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = torch.nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


# with h5py.File(r'/media/ubuntu/Seagate Expansion Drive/h5/full-origin', 'r') as f:
#     dataset = f['full_origin_dataset']
# class RAADataset(Dataset):
#     def __init__(self,origin_h5_file,normal_h5_file,abnormal_h5_file):
#         self.origin_h5_file=h5py.File(origin_h5_file,'r')
#         self.normal_h5_file=h5py.File(normal_h5_file,'r')
#         self.abnormal_h5_file=h5py.File(abnormal_h5_file,'r')
#         self.origin_video_names=['ov_{}'.format(i) for i in range(1,5001)]
#         self.normal_video_names=['nv_{}'.format(i) for i in range(1,5001)]
#         self.normal_start_id=['start_id_{}'.format(i) for i in range(1,5001)]
#         self.abnormal_video_names=['av_{}'.format(i) for i in range(1,5001)]
#         self.abnormal_start_id=['start_id_{}'.format(i) for i in range(1,5001)]
#         self.normal_prompt=['normal_prompt_{}'.format(i) for i in range(1,5001)]
#         self.abnormal_prompt = ['abnormal_prompt_{}'.format(i) for i in range(1, 5001)]
#     def insert_converted_frames(self,original_video_batch, converted_video_batch, start_frames, normal):
#         # for i in range(original_video_batch.shape[0]):
#         original_video_batch[:, start_frames[0]:start_frames[0]+converted_video_batch.shape[1], :, :]=converted_video_batch[:,:,:,:]
#         # converted_video_batch[i]
#         if normal:
#             label = [(torch.tensor([1, 0])) for _ in range(original_video_batch.shape[0])]
#             tai = [(torch.tensor(-1)) for _ in range(original_video_batch.shape[0])]
#         else:
#             label = [(torch.tensor([0, 1])) for _ in range(original_video_batch.shape[0])]
#             tai = start_frames
#         original_video_batch = rearrange(original_video_batch, 'c f h w  -> f c h w')
#         return original_video_batch, label, tai
#
#
#
#
#
#
#     def __len__(self):
#         return len(self.origin_video_names)
#
#     def __getitem__(self,index):
#         # positive_index=random.randint(0,1799)
#         # negtive_index=random.randint(0,1799)
#         origin_video_name= self.origin_video_names[index]
#         normal_video_name=self.normal_video_names[index]
#         normal_start_name=self.normal_start_id[index]
#         abnormal_video_name = self.abnormal_video_names[index]
#         abnormal_start_name = self.abnormal_start_id[index]
#         # origin_p_video_name=self.origin_video_names[positive_index]
#         # origin_n_video_name=self.origin_video_names[negtive_index]
#         # positive_video_name=self.normal_video_names[positive_index]
#         # negtive_video_name=self.abnormal_video_names[positive_index]
#         # positive_start_id = self.normal_start_id[index]
#         # negtive_start_id =self.abnormal_start_id[negtive_index]
#         normal_start_id = self.normal_h5_file['min_normal_dataset'][normal_start_name][:]
#         abnormal_start_id = self.abnormal_h5_file['min_abnormal_dataset'][abnormal_start_name ][:]
#         origin_video_data=self.origin_h5_file['min_origin_dataset'][origin_video_name][:]
#         # origin_video_data=  origin_video_data /  127.5 - 1.0
#         normal_video_data=self.normal_h5_file['min_normal_dataset'][normal_video_name][:]
#         abnormal_video_data = self.abnormal_h5_file['min_abnormal_dataset'][abnormal_video_name][:]
#         normal_prompt_name=self.normal_prompt[index]
#         normal_prompt=self.normal_h5_file['min_normal_dataset'][normal_prompt_name][:]
#         normal_prompt=normal_prompt.tostring().decode('utf-8').replace("\x00",'')
#         abnormal_prompt_name = self.abnormal_prompt[index]
#         abnormal_prompt = self.abnormal_h5_file['min_abnormal_dataset'][abnormal_prompt_name][:]
#         abnormal_prompt = abnormal_prompt.tostring().decode('utf-8').replace("\x00", '')
#
#
#
#         # origin_video_daata = norm(origin_video_data)
#         # origin_video_data = torch.from_numpy(origin_video_data) / 127.5 - 1.0
#         # origin_video_data = rearrange(origin_video_data, 'c f h w  -> f c h w')
#         # origin_p_video_data = self.origin_h5_file['full_origin_dataset'][origin_p_video_name][:]
#         # origin_p_video_data  = torch.from_numpy(origin_p_video_data ) / 127.5 - 1.0
#         # origin_n_video_data=self.origin_h5_file['full_origin_dataset'][origin_n_video_name][:]
#         # origin_n_video_data = torch.from_numpy(origin_n_video_data) / 127.5 - 1.0
#         # positive_video_data=self.normal_h5_file['full_normal_dataset'][positive_video_name][:]
#         # positive_video_data=torch.from_numpy(positive_video_data)
#         # normal_video_data, label_nv, tai_nv = self.insert_converted_frames(origin_p_video_data, positive_video_data,                                           normal_start_id, normal=True)
#         # normal_video_data=norm1(normal_video_data)
#         # negtive_video_data = self.abnormal_h5_file['full_abnormal_dataset'][negtive_video_name][:]
#         # negtive_video_data = torch.from_numpy( negtive_video_data)
#         # abnormal_video_data, label_av, tai_av = self.insert_converted_frames(origin_n_video_data,  negtive_video_data,
#         #                                                                      abnormal_start_id, normal=False)
#         # abnormal_video_data=norm1(abnormal_video_data)
#         data={ "ov": origin_video_data,
#                "nv": normal_video_data,
#                "n_p": normal_prompt,
#                # "label_nv":label_nv,
#                # "label_av":label_av,
#                "start_id": normal_start_id ,
#                "tai":abnormal_start_id,
#                "av":abnormal_video_data,
#                "a_p":abnormal_prompt
#                # "tai_av": tai_av,
#         }
#         return data


class BDDA_T(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.video_folders=sorted(os.listdir(root_path))
        self.fps = 30
        self.g_root_path=r"/data/Mamba_T2V/WT_OAVD_CVPR/Test/BDDA_DOWN/train_2024-11-18T14-45-28"
    def __len__(self):
        return len(self.video_folders)

    def pross_video_data(self,video):
         video_datas=[]
         for fid in range(len(video)):
             video_data=video[fid]
             video_data=Image.open(video_data)
             video_data = video_data.resize((224, 224))
             video_data= np.asarray(video_data, np.float32)
             video_datas.append(video_data)

         video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
         video_data = rearrange(video_data, 'f w h c -> c f w h')
         return video_data

    def read_nomarl_rgbvideo(self, video_file):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data
        # cond_video_data=video_file[start-16:start-10]
        # cond_video_data=self.pross_video_data(cond_video_data)
        tran_video_data=video_file[0:16]
        tran_video_data=self.pross_video_data(tran_video_data)
        return tran_video_data

    def read_abnomarl_rgbvideo(self, video_file, start, end):
        tran_video_data = video_file[end-16:end]
        tran_video_data = self.pross_video_data(tran_video_data)
        return tran_video_data


    def __getitem__(self, index):
        video_path = os.path.join(self.root_path,self.video_folders[index])
        g_video_path=os.path.join(self.g_root_path,self.video_folders[index])
        video_path=glob.glob(video_path+'/'+"*.[jp][pn]g")
        video_path= sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        g_video_path = glob.glob(g_video_path + '/' + "*.[jp][pn]g")
        g_video_path = sorted(g_video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        train_vdata= self.read_nomarl_rgbvideo(video_path)
        g_vdata=self.read_nomarl_rgbvideo(g_video_path)
        accident_id = self.video_folders[index]
        example = {
        "pixel_values": train_vdata/ 127.5 -1.0,
        "g_pixel_values":g_vdata/127.5 -1.0,
        "accident_id":accident_id}
        return example







class DADA(Dataset):
    def __init__(self, root_path,map_path):
        self.fps = 30
        # self.data_list, self.tar, self.tai, self.tco, self.NC_text, self.R_text, self.P_text, self.C_text = self.get_data_list()
        self.root_path = root_path
        self.map_path=map_path
        self.data_list, self.tai, self.tco, self.text= self.get_data_list()

    def get_data_list(self):
        list_file = os.path.join(self.map_path + "/" + '.txt')
        assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
        fileIDs,tais, tcos,texts= [], [], [], []
        # samples_visited, visit_rows = [], []
        with open(list_file, 'r', encoding='utf-8') as f:
            # for ids, line in enumerate(f.readlines()):
            for ids, line in enumerate(f.readlines()):
                parts = line.strip().split(',')
                ID,label,tai,tco=parts[0].split(' ')
                fileIDs.append(ID)
                tais.append(tai)
                tcos.append(tco)
                texts.append(parts[1])
        return fileIDs, tais, tcos,texts

    def __len__(self):
        return len(self.data_list)
    def pross_video_data(self, video):
        video_datas = []
        for fid in range(len(video)):
            video_data = video[fid]
            video_data = Image.open(video_data)
            video_data = video_data.resize((224,224))
            video_data = np.asarray(video_data, np.float32)
            if len(video_data.shape) <3:
                video_data = np.stack((video_data,video_data,video_data),-1)
            video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
        video_data = rearrange(video_data, 'f w h c -> c f w h')
        return video_data

    def read_nomarl_rgbvideo(self, video_file):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data

        video_data = self.pross_video_data(video_file)

        return video_data


    def gather_info(self, index):
        # accident_id = int(self.data_list[index].split('/')[0])
        accident_id = self.data_list[index]
        video_id = int(self.data_list[index].split('/')[1])
        catagroy=int(self.data_list[index].split('/')[0])
        text = self.text[index]
        return accident_id, video_id, text,catagroy

    def convert_path(self,path):
        # 找到最后一个 / 的位置
        last_slash_index = path.rfind('/')

        # 如果找到最后一个 /，则替换为 _
        if last_slash_index != -1:
            new_path = path[:last_slash_index] + "_" + path[last_slash_index + 1:]
            return new_path
        else:
            # 如果没有 /，则直接返回原路径
            return path

    def remove_last_part(self,path):
        # 从右侧分割一次，得到两部分
        parts = path.rsplit('/', 1)

        # 如果成功分割，则返回第一部分
        if len(parts) == 2:
            return parts[0]
        else:
            # 如果没有 /，则直接返回原路径
            return path

    def __getitem__(self, index):
        # read RGB video (trimmed)
        tco=int(self.tco[index])
        accident_id=self.data_list[index]
        video_path = os.path.join(self.root_path+"/",self.data_list[index])
        map_list=self.remove_last_part(self.data_list[index])
        maps_path = os.path.join(self.map_path + "/", map_list + "/" + "maps")
        start_frame =tco-16
        v_r = [video_path + "/" + f'{i:04d}' + ".png" for i in range(start_frame, start_frame + 16)]
        m_r= [maps_path + "/" + f'{i:04d}' + ".png" for i in range(start_frame, start_frame + 16)]
        accident_id, video_id,text,catagroy= self.gather_info(index)
      
        self.g_list=self.convert_path(self.data_list[index])
        g_video_path = os.path.join(self.root_path + "/",self.g_list)
        g_video_path = glob.glob(g_video_path + '/' + "*.[jp][pn]g")
        g_video_path = sorted(g_video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        example = {
        "pixel_map_values": m_r,
        "prompt": text,
        "accident_id":accident_id,
        "g_pixel_values":g_video_path
        }
        return example

















































































#
# class DADA1(Dataset):
#     def __init__(self, root_path):
#         self.fps = 30
#         # self.data_list, self.tar, self.tai, self.tco, self.NC_text, self.R_text, self.P_text, self.C_text = self.get_data_list()
#         self.root_path = root_path
#         self.data_list, self.tai, self.tco, self.text= self.get_data_list()
#         # self.bbx_path = r"/media/ubuntu/Seagate Expansion Drive/NEW_OOD/padding_outputs"
#         self.json_file= r"/data/dada.json"
#         self.json_key_list, self.json_values_list=self.read_dada_json()
#         self.r_tai,self.r_tco=self.match_keys_and_ids()
#         self.g_root_path=r"/data/Mamba_T2V/WT_LAMP/TEST/DADA_SX"
#
#
#         # self.qa_path=r"/media/work/TOSHIBA EXT/toki_nize/all_data.pth"
#         self.map_path=root_path
#         # self.qa_data=self.load_qa()
#
#     # def load_qa(self):
#     #     loaded_data = torch.load(self.qa_path)
#     #     return loaded_data
#     # "
#     def read_dada_json(self):
#         with open(self.json_file, 'r', encoding='utf-8') as file:
#             data =json.load(file)
#         key_list = []
#         values_list = []
#         for key, value in data.items():
#             key_list.append(key)
#             values_list.append(value)
#         return key_list,values_list
#
#
#
#     def get_data_list(self):
#         # list_file = os.path.join(self.root_path + "/" + 'OOD_train.txt')
#         list_file = os.path.join(self.root_path + "/" + 'nips_train.txt')
#         # ff=open(os.path.join(self.root_path, self.phase + '\word.txt'),encoding='utf-8')
#         assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
#         fileIDs,tais, tcos,texts= [], [], [], []
#         # samples_visited, visit_rows = [], []
#         with open(list_file, 'r', encoding='utf-8') as f:
#             # for ids, line in enumerate(f.readlines()):
#             for ids, line in enumerate(f.readlines()):
#                 parts = line.strip().split(',')
#                 ID,label,tai,tco=parts[0].split(' ')
#                 fileIDs.append(ID)
#                 tais.append(tai)
#                 tcos.append(tco)
#                 texts.append(parts[1])
#         # file_counts=Counter(file.split('/')[0] for file in fileIDs)
#         # sorted_files=sorted(fileIDs,key=lambda x:file_counts[x.split('/')[0]],reverse=True)
#         # print( sorted_files)
#         return fileIDs, tais, tcos,texts
#
#     def match_keys_and_ids(self):
#         key_list=self.json_key_list
#         match_tais=[]
#         match_tcos=[]
#         for key in key_list:
#             if key in self.data_list:
#                 index=self.data_list.index(key)
#                 match_tais.append(self.tai[index])
#                 match_tcos.append(self.tco[index])
#         return match_tais,match_tcos
#
#
#
#     def __len__(self):
#         return len(self.json_key_list)
#
#     def pross_video_data(self, video):
#         video_datas = []
#         for fid in range(len(video)):
#             video_data = video[fid]
#             video_data = Image.open(video_data)
#             video_data = video_data.resize((224,224))
#             video_data = np.asarray(video_data, np.float32)
#             if len(video_data.shape) <3:
#                 video_data = np.stack((video_data,video_data,video_data),-1)
#             video_datas.append(video_data)
#
#         # guide_image=video_datas[0]
#         # guide_image = rearrange(guide_image, 'w h c -> c w h')
#         video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
#         video_data = rearrange(video_data, 'f w h c -> c f w h')
#         return video_data
#
#     def read_nomarl_rgbvideo(self, video_file):
#         """Read video frames
#         """
#         # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
#         # get the video data
#
#         video_data = self.pross_video_data(video_file)
#
#         return video_data
#
#
#     def gather_info(self, index):
#         # accident_id = int(self.data_list[index].split('/')[0])
#         accident_id = self.data_list[index]
#         video_id = int(self.data_list[index].split('/')[1])
#         catagroy=int(self.data_list[index].split('/')[0])
#         text = self.text[index]
#         return accident_id, video_id, text,catagroy
#
#     # def read_qa_data(self,):
#     #     for video_name, entry in loaded_data.items():
#     #         a.append(video_name)
#     #         print(f"Video Name: {entry['video_name']}")
#     #         print(f"Option Token Shape: {entry['option_token'].shape}")  # 仍然是 (5, 77)
#     #         print(f"Answer ID: {entry['answer_id']}")
#     #         print(f"Answer Token: {entry['answer_token'].shape}")  # 仍然是 tensor
#     #         b.append(entry['answer_token'])
#
#     def __getitem__(self, index):
#         # read RGB video (trimmed)
#
#         # tai = int(self.tai[index])
#         # tco = int(self.tco[index])
#         tco=int(self.r_tco[index])
#         accident_id=self.json_key_list[index]
#         video_path = os.path.join(self.root_path+"/",self.json_key_list[index]+"/"+"images")
#         maps_path = os.path.join(self.map_path + "/",self.json_key_list[index] + "/" + "maps")
#         start_frame =tco-16
#         v_r = [video_path + "/" + f'{i:04d}' + ".png" for i in range(start_frame, start_frame + 16)]
#         m_r= [maps_path + "/" + f'{i:04d}' + ".png" for i in range(start_frame, start_frame + 16)]
#
#         # accident_id, video_id,text,catagroy= self.gather_info(index)
#         text=self.json_values_list[index]
#         # option_token=self.qa_data[video_id]['option_token']
#         # question_token=self.qa_data[video_id]['answer_token']
#         # answer_id=self.qa_data[video_id]['answer_id']
#         # all_ids = list(range(5))
#         # gt_option =option_token[answer_id,:]
#         # all_ids.remove(answer_id)
#         # non_gt_option_token = option_token[all_ids,:]
#         # print(self.qa_data[video_id])
#         # texts,y,accident_id,cause= self.gather_info(index)
#         # vr = self.read_nomarl_rgbvideo(v_r)
#         # mr = self.read_nomarl_rgbvideo(m_r)
#         # bbx_paths = os.path.join(self.bbx_path + "/", str(accident_id))
#         # bbx_path = [bbx_paths + "/" + f'{i:04d}' + ".txt" for i in range(start_frame, start_frame + 16)]
#         # bbx_path = [bbx_path for bbx_path in os.listdir(bbx_paths)]
#         # bbx_path = sorted(bbx_path, key=lambda x: int(x.split('.')[0]))
#         # bbx_info_list = []
#         # caption_list = []
#         # for bbx_file in bbx_path:
#         #     bbx_file = os.path.join(bbx_paths, bbx_file)
#         #     with open(bbx_file, 'r') as file:
#         #         lines = file.readlines()
#         #         if not lines or len(lines) == 0 or all(line.isspace() for line in lines):
#         #             filtered_datas = torch.zeros(1, 4, dtype=torch.float32)
#         #             captions = ['The frame detection result is None,']
#         #         else:
#         #             filtered_data = [list(map(int, line.split(' ')[:4])) for line in lines]
#         #             captions = [line.split(' ')[-2] for line in lines]
#         #             if not filtered_data or len(filtered_data) == 0:
#         #                 filtered_datas = torch.stack(
#         #                     [torch.zeros(1, 4, dtype=torch.float32) for _ in range(16)]).squeeze(1)
#         #             else:
#         #                 filtered_datas = torch.stack(
#         #                     [torch.tensor(list(map(float, line)), dtype=torch.float32) for line in filtered_data])
#         #         bbx = self.bbx_caption_process(filtered_datas, original_image_size=(640, 640),
#         #                                        target_image_size=(224, 224),max_N=10)
#         #         merged_caption = ", ".join(captions)
#         #         caption_list.append(merged_caption)
#         #         bbx_info_list.append(bbx)
#         # boxes = torch.stack(bbx_info_list)
#         # image_mask = torch.stack(image_mask)
#         # text_mask = torch.stack(text_mask)
#         # captions=torch.stack(caption_list)
#         # mask=self.extract_masks(vr,boxes_)
#         g_video=[]
#         for i in range (len(text)):
#             g_video_path = os.path.join(self.g_root_path + "/", self.json_key_list[index] + f"_{i}")
#             g_video_path = glob.glob(g_video_path + '/' + "*.[jp][pn]g")
#             g_video_path = sorted(g_video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
#             # g_r = self.read_nomarl_rgbvideo(g_video_path)
#             g_video.append(g_video_path)
#         example = {
#         "pixel_values": v_r,
#         "pixel_map_values": m_r,
#         "prompt": text,
#         # "bbx": boxes,
#         # "mask":mask / 127.5 - 1.0,
#         # "answer_id": answer_id,
#         # "question":question_token,
#         # "option":gt_option,
#         # "video_id":video_id,
#         "accident_id":accident_id,
#         "g_pixel_values":g_video
#         # "caption": " ".join(caption_list),
#
#         # "cat":catagroy,
#         # "non_option_token":non_gt_option_token,
#         }
#         return example
#
#



class T2V(Dataset):
    def __init__(self, g_root_path):
        # self.root_path = root_path
        self.video_folders=sorted(os.listdir(g_root_path))
        self.fps = 30
        self.g_root_path=g_root_path

    def __len__(self):
        return len(self.video_folders)

    def pross_video_data(self,video):
         video_datas=[]
         for fid in range(len(video)):
             video_data=video[fid]
             video_data=Image.open(video_data)
             video_data = video_data.resize((224, 224))
             video_data= np.asarray(video_data, np.float32)
             video_datas.append(video_data)

         video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
         video_data = rearrange(video_data, 'f w h c -> c f w h')
         return video_data

    def read_nomarl_rgbvideo(self, video_file):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data
        # cond_video_data=video_file[start-16:start-10]
        # cond_video_data=self.pross_video_data(cond_video_data)
        tran_video_data=video_file[0:16]
        tran_video_data=self.pross_video_data(tran_video_data)
        return tran_video_data

    def read_abnomarl_rgbvideo(self, video_file, start, end):
        tran_video_data = video_file[end-16:end]
        tran_video_data = self.pross_video_data(tran_video_data)
        return tran_video_data


    def __getitem__(self, index):
        # video_path = os.path.join(self.root_path,self.video_folders[index])
        g_video_path=os.path.join(self.g_root_path,self.video_folders[index])
        # video_path=glob.glob(video_path+'/'+"*.[jp][pn]g")
        # video_path= sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        g_video_path = glob.glob(g_video_path + '/' + "*.[jp][pn]g")
        g_video_path = sorted(g_video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        # train_vdata= self.read_nomarl_rgbvideo(video_path)
        # g_vdata=self.read_nomarl_rgbvideo(g_video_path)
        accident_id = self.video_folders[index]
        example = {
        # "pixel_values": train_vdata/ 127.5 -1.0,
        # "g_pixel_values":g_vdata/127.5 -1.0,
        "g_pixel_values":g_video_path,
        "accident_id":accident_id}
        return example




if __name__=="__main__":
    import itertools
    # val_dataset = BDDA_D(root_path=r"/media/work/Elements SE/LLL/C_OAVD/BDDA")
    # val_dataset = BDDA_T(root_path=r"/data/LLL/BDDA_Y")
    # val_dataset = BDDA_T(root_path=r"/data/LLL/HEVI")
    val_dataset = DADA(root_path=r"/media/work/My Passport/DADA2000")
    # val_dataset=T2V(g_root_path=r"/data/Mamba_T2V/WT_OAVD_CVPR/Test/UP/T2V_UP/train_2024-11-18T15-23-00")
    # Preprocessing the dataset
    # train_dataset.prompt_ids = tokenizer(
    #     train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    # ).input_ids[0]
    device = torch.device("cuda",0)
    # # DataLoaders creation:
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        pin_memory=True,num_workers=8, drop_last=True)

    clip_scores = []
    model=InceptionV3()
    # device = torch.device("cuda", 1)
    models, preprocess = clip.load("ViT-B/32", device=device)
    # with open(r"/data/LLL/cate_balanced.txt", 'r') as file:
    #     txt_data = file.readlines()
    # txt_data_cycle = itertools.cycle(txt_data)

    # with open(r'H:\NIPS\Latte\opendk_cl.txt', 'a') as f:
    for idx, batch in enumerate(val_dataloader):
        device = torch.device("cuda", 0)
        # text_batch=list(itertools.islice(txt_data_cycle,4))
        # txt_data = next(txt_data_cycle)
        # promptss=[]
        # for text_data in text_batch:
        #     prompts = txt_data.split('/')[1]
        #     promptss.append(prompts)
        prompts = batch["prompt"]
        # o_v = batch["pixel_values"].to(device)
        g_v=batch["g_pixel_values"]
        g_v=torch.cat(g_v,dim=0).to(device)
        clip_score=caculate_clip(g_v,prompts,models,preprocess )
        clip_scores.append(clip_score)
        # f.write(f'{idx}: {clip_score}\n')  # 每个 score 占一行
        # 可以在此处打印或处理平均值
        print(f'Clip score for batch {idx}: {clip_score}')


    averge_clip_score=sum(clip_scores)/len(clip_scores)
    print(averge_clip_score)
        # f.close()

    # device = torch.device("cuda", 0)
    # # DataLoaders creation:
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=1, shuffle=False,
    #     pin_memory=True, num_workers=8, drop_last=True)
    #
    # clip_scores = []
    # model = InceptionV3()
    # # device = torch.device("cuda", 1)
    # models, preprocess = clip.load("ViT-B/32", device=device)
    #
    # # with open(r'H:\NIPS\Latte\opendk_cl.txt', 'a') as f:
    # for idx, batch in enumerate(val_dataloader):
    #     device = torch.device("cuda", 0)
    #     prompts = batch["prompt"]
    #     # o_v = batch["pixel_values"].to(device)
    #     g_v = batch["g_pixel_values"]
    #     g_v=torch.cat(g_v,dim=0).to(device)
    #     clip_score = caculate_clip(g_v,prompts, models, preprocess)
    #     clip_scores.append(clip_score)
    #     # f.write(f'{idx}: {clip_score}\n')  # 每个 score 占一行
    #     # 可以在此处打印或处理平均值
    #     print(f'Clip score for batch {idx}: {clip_score}')
    #
    # averge_clip_score = sum(clip_scores) / len(clip_scores)
    # print(averge_clip_score)


#
# def cnt_params(model):
#     return sum(param.numel() for param in model.parameters())
#     logging.info(f'nnet has {cnt_params(nnet)} parameters')