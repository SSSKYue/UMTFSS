import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random

from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)

class SemData(Dataset):
    def __init__(self, split=3, shot=1, data_root=None, data_list=None, transform=None, mode='train', use_coco=False, use_split_coco=False):
        assert mode in ['train', 'val', 'test']
        
        self.mode = mode 
        self.split = split  
        self.shot = shot
        self.data_root = data_root
        self.use_coco = use_coco

        if use_coco:
            self.nclass = 80
            self.img_path = os.path.join(data_root, '{}2014'.format(self.mode))
            self.ann_path = os.path.join(data_root, 'annotations/{}2014'.format(self.mode))
            self.pseudo_path = os.path.join(data_root, 'coco_moco_pseudo')
        else:
            self.nclass = 20
            self.img_path = os.path.join(data_root, 'JPEGImages')
            self.ann_path = os.path.join(data_root, 'SegmentationClassAug')
            self.pseudo_path = os.path.join(data_root, 'pascal_moco_pseudo') 

        if not use_coco:
            self.class_list = list(range(1, 21)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3: 
                self.sub_list = list(range(1, 16)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21)) #[16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21)) #[1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16)) #[11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21)) #[1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11)) #[6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21)) #[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6)) #[1,2,3,4,5]
            elif self.split == 5: 
                self.sub_val_list = list(range(1, 21))
                self.sub_list = []

        else:
            if use_split_coco:
                print('INFO: using SPLIT COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 5: 
                    self.sub_val_list = [1,2,3,4,5,6,7,9,15,16,17,18,19,20,40,57,58,59,61,66]
                    self.sub_list = []
            else:
                print('INFO: using COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81)) 
                    self.sub_val_list = list(range(1, 21))    

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)   

        dataset = 'coco' if use_coco else 'pascal'

        fss_list_root = './lists/'+ dataset +'/{}_fold_{}'.format(data_list.split('/')[-1][:-4], self.split)
        print('fss_list_root', fss_list_root)
        fss_data_list_path = os.path.join(fss_list_root, 'list_fold_{}.txt'.format(self.split))
        fss_sub_class_file_list_path = os.path.join(fss_list_root, 'clus_class_file_list_{}.txt'.format(self.split))

        if os.path.exists(fss_list_root):   
             # # Read FSS Data
            with open(fss_data_list_path, 'r') as f:
                f_str = f.readlines()
            self.data_list = []
            for line in f_str:
                img = line.strip()
                self.data_list.append(img)
            with open(fss_sub_class_file_list_path, 'r') as f:
                f_str = f.read()
            self.sub_class_file_list = eval(f_str)

        else:
            if self.mode == 'train':
                self.data_list, self.sub_class_file_list = self.make_dataset(split, data_root, data_list, self.sub_list)
                assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
            elif self.mode == 'val':
                self.data_list, self.sub_class_file_list = self.make_dataset(split, data_root, data_list, self.sub_val_list)
                print('data_list:', len(self.data_list))
                assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list) 

            os.mkdir(fss_list_root)
            # Write FSS Data
            with open(fss_data_list_path, 'w') as f:
                for query_name in self.data_list:
                    f.write(query_name + '\n')
            with open(fss_sub_class_file_list_path, 'w') as f:
                f.write(str(self.sub_class_file_list))

        self.transform = transform


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        name = self.data_list[index]
        q_name = name
        image_path = os.path.join(self.img_path, name+'.jpg')
        label_path = os.path.join(self.ann_path, name+'.png')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  

        padding_mask = np.zeros_like(label)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))          
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255) 
        new_label_class = []       
        for c in label_class:
            if c in self.sub_val_list:
                if self.mode == 'val' or self.mode == 'test':
                    new_label_class.append(c)
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class    
        assert len(label_class) > 0


        cls_id = random.randint(1,len(label_class))-1
        class_chosen = label_class[cls_id]
        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)
        label[:,:] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0],target_pix[1]] = 1 
        label[ignore_pix[0],ignore_pix[1]] = 255  

        for cl in label_class:
            assert len(self.sub_class_file_list[cl]) > 5, 'class {} has less then 5 samples!!'.format(cl)

        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1,num_file)-1
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1,num_file)-1
                support_name = file_class_chosen[support_idx]
                support_image_path = os.path.join(self.img_path, support_name+'.jpg')
                support_label_path = os.path.join(self.ann_path, support_name+'.png')    
            s_name = support_name                 
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []
        support_padding_list = []
        subcls_list = []
        for k in range(self.shot):  
            if self.mode == 'train':
                subcls_list.append(self.sub_list.index(class_chosen))
            else:
                subcls_list.append(self.sub_val_list.index(class_chosen))
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k] 
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)      
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:,:] = 0
            support_label[target_pix[0],target_pix[1]] = 1 
            support_label[ignore_pix[0],ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))     

            if not self.use_coco:
                support_padding_label = np.zeros_like(support_label)
                support_padding_label[support_label==255] = 255
            else:
                support_padding_label = np.zeros_like(support_label)
            support_image_list.append(support_image)
            support_label_list.append(support_label)
            support_padding_list.append(support_padding_label)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot                    
        
        raw_label = label.copy()
        if self.transform is not None:
            image, label, padding_mask = self.transform(image, label, padding_mask)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k], support_padding_list[k] = self.transform(support_image_list[k], support_label_list[k], support_padding_list[k])

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        if support_padding_list is not None:
            s_eys = support_padding_list
            s_ey = s_eys[0].unsqueeze(0)
            for i in range(1, self.shot):
                s_ey = torch.cat([s_eys[i].unsqueeze(0), s_ey], 0)   

        if self.mode == 'train':
            return image, label, s_x, s_y, padding_mask, s_ey
        else:
            return image, label, s_x, s_y, padding_mask, s_ey, subcls_list, raw_label
            # return image, label, s_x, s_y, padding_mask, s_ey, subcls_list, raw_label, q_name, s_name

    def make_dataset(self, split=0, data_root=None, data_list=None, sub_list=None):    
        assert split in [0, 1, 2, 3, 5, 10, 11, 999]
        if not os.path.isfile(data_list):
            raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

        image_label_list = []  
        list_read = open(data_list).readlines()
        print("Processing data...")
        sub_class_file_list = {}
        for sub_c in sub_list:
            sub_class_file_list[sub_c] = []

        for l_idx in tqdm(range(len(list_read))):
            line = list_read[l_idx]
            line = line.strip()
            line_split = line.split(' ')
            name = line_split[0].split('/')[-1][:-4]
            label_name = os.path.join(self.ann_path, name+'.png')
            label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
            label_class = np.unique(label).tolist()

            if 0 in label_class:
                label_class.remove(0)
            if 255 in label_class:
                label_class.remove(255)

            new_label_class = []       
            for c in label_class:
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0],target_pix[1]] = 1 
                    if tmp_label.sum() >= 2 * 32 * 32:      
                        new_label_class.append(c)

            label_class = new_label_class    

            if len(label_class) > 0:
                image_label_list.append(name)
                for c in label_class:
                    if c in sub_list:
                        sub_class_file_list[c].append(name)
                        
        print("Checking image&label pair {} list done! ".format(split))
        return image_label_list, sub_class_file_list
