
import os
import cv2
import random
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from data import transformsp as transform
from skimage import segmentation
from torch.utils.data.dataloader import default_collate
import json

def my_collate(batch):
    image = [samples[0] for samples in batch]
    image = torch.stack(image, 0)
    ori_size = [samples[-3] for samples in batch]
    names = [samples[-2] for samples in batch]
    spl = [samples[-1] for samples in batch]
    return [image, ori_size, names, spl]

def get_img_label(image_path=None, label_path=None):
    """
    :param image_path: str / None
    :param label_path: str
    :return:
        image: np float array of shape (H, W, 3)
        label: np int array of (H, W)
        unique_class: int list, classes exists in the image
    """
    if label_path is not None:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    else:
        label = None

    if image_path is not None:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
    else:
        image = None
    return image, label

 
class CLusterData(Dataset):
    def __init__(self, mode, data_root, data_list, dataset='pascal', use_spl=False, sample_num=100000):
        if not os.path.exists(data_list):
            raise RuntimeError(f"Image list do not exist: {data_list}\n")

        self.mode = mode
        if dataset == 'pascal' or dataset == 'coco':
            list_read = open(data_list).readlines()
        else:
            list_read = os.listdir(data_list)

        if sample_num < len(list_read) and sample_num > 0:
            sampled_idx = random.sample(range(len(list_read)), sample_num)
        else:
            sampled_idx = range(len(list_read))

        if dataset == 'pascal' or dataset == 'coco':
            sampled_img_list = []
            for l_idx in sampled_idx:
                line = list_read[l_idx]
                line = line.strip()
                line_split = line.split(' ')

                image_name = os.path.join(data_root, line_split[0])
                sampled_img_list.append(image_name)
        else:
            sampled_img_list = [os.path.join(data_list, list_read[i]) for i in sampled_idx]

        if sample_num < len(list_read) and sample_num > 0:
            record = json.dumps(sampled_img_list)
            with open('./cluster_sampled.json', 'w') as f:
                f.write(record)

        self.img_list = sampled_img_list
        self.dataset = dataset
        self.use_spl = use_spl
        self.superpixel_type = 'felzenszwalb'

        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

        if self.mode == 'assign':
            self.transform = transform.Compose([transform.Resize(size=640)],mean,std)
        else:
            self.transform = transform.Compose([transform.Resize(size=473)],mean,std)
 
    def __len__(self):
        return len(self.img_list)
 
    def __getitem__(self, index):
        image_path = self.img_list[index]
        name = image_path[:-4].split('/')[-1]
        image, _ = get_img_label(image_path=image_path, label_path=None)
        ori_size = image.shape[:2]
        
        # transform
        if self.use_spl:
            superpixel = get_superpixel(image_path=image_path, method=self.superpixel_type)
            image, [superpixel] = self.transform(image, [superpixel])
            return image, ori_size, name, superpixel
        else:
            image, _ = self.transform(image)
            return image, ori_size, name
        


def get_superpixel(image_path, method):
    """
    Generate superpixel label
    255 -> ignore label
    1 to n -> the n base classes in the image
    0 is reserved for novel class
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if method == 'slic':
        superpixel = segmentation.slic(image, compactness=10, n_segments=100)  # 250
    elif method == 'felzenszwalb':
        superpixel = segmentation.felzenszwalb(image, scale=100, sigma=0.8, min_size=200)
    elif method == 'hed':
        image_name = image_path.split('/')[-1].split('.')[0]
        superpixel = cv2.imread(f'./hed/{image_name}.png', cv2.IMREAD_GRAYSCALE)
        superpixel = np.asarray(superpixel)
    else:
        raise ValueError(f'Do not recognise superpixel method {method}')

    # # reserve 0
    # superpixel += 1

    return superpixel

def get_data_loader(args, dataset):
    if args.mode == 'assign':
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

