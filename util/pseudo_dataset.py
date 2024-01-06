r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import random
from model.extractor_model.resnet import resnet50_
from model.extractor_model.vit import vit_base
from model.extractor_model.resnet_simclr import resnet50x1
from util.extract_feature import *


class DatasetSSL(Dataset):
    def __init__(self, cluster_num, fold, shot, transform, data_root, data_list, mode='train', use_coco=False, use_split_coco=False):
        assert mode == 'train', 'ssl dataset only used for training'
        self.split = mode
        self.fold = fold
        self.nfolds = 4
        self.extractor = 'moco'
        if use_coco:
            self.nclass = 80
            self.img_path = os.path.join(data_root, 'train2014')
            self.ann_path = os.path.join(data_root, 'annotations/train2014')
            self.pseudo_path = os.path.join(data_root, 'coco_{}_pseudo'.format(self.extractor))
        else:
            self.nclass = 20
            self.img_path = os.path.join(data_root, 'JPEGImages')
            self.ann_path = os.path.join(data_root, 'SegmentationClassAug')
            self.pseudo_path = os.path.join(data_root, 'pascal_{}_pseudo'.format(self.extractor))
        self.shot = shot
        self.use_coco = use_coco
        self.use_split_coco = use_split_coco

        self.data_list = data_list
        
        self.transform = transform
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        self.transform_ft = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(img_mean, img_std)])

        self.cluster_num = cluster_num
        self.sub_set, self.sub_val_set = self.build_class_ids()
        self.class_ids = list(range(self.cluster_num))
        
        dataset = 'coco' if use_coco else 'pascal'

        fss_list_root = './lists/'+ dataset +'/train_ssl_fold_{}'.format(self.fold)
        fss_data_list_path = os.path.join(fss_list_root, 'list_fold_{}.txt'.format(self.fold))
        fss_sub_class_file_list_path = os.path.join(fss_list_root, 'clus_class_file_list_{}.txt'.format(self.fold))
        fss_sub_class_feat_list_path = os.path.join(fss_list_root, 'clus_class_feat_list_{}.pth'.format(self.fold))
        similar_dict_path = os.path.join(fss_list_root, 'similar_dict_{}.pth'.format(self.fold))

        if os.path.exists(fss_list_root):
            # # Read FSS Data
            with open(fss_data_list_path, 'r') as f:
                f_str = f.readlines()
            self.img_metadata = []
            for line in f_str:
                img = line.strip()
                self.img_metadata.append(img)
            self.img_metadata = self.img_metadata
            with open(fss_sub_class_file_list_path, 'r') as f:
                f_str = f.read()
            self.img_metadata_classwise = eval(f_str)
            self.mask_feature_classwise = torch.load(fss_sub_class_feat_list_path)
            
            # load
            self.similar_dict = torch.load(similar_dict_path)
        else:
            if self.fold == 5:
                self.img_metadata, self.img_metadata_classwise, self.mask_feature_classwise = self.make_all_dataset()
            else:
                self.img_metadata, self.img_metadata_classwise, self.mask_feature_classwise = self.make_dataset(filter_intersection=False)

            os.mkdir(fss_list_root)
            # Write FSS Data
            with open(fss_data_list_path, 'w') as f:
                for query_name in self.img_metadata:
                    f.write(query_name + '\n')
            with open(fss_sub_class_file_list_path, 'w') as f:
                f.write(str(self.img_metadata_classwise))
            torch.save(self.mask_feature_classwise, fss_sub_class_feat_list_path)

            self.similar_dict = self.compute_similar_dict()
            torch.save(self.similar_dict, similar_dict_path)


        print('Total (%s) images are : %d' % (self.split, len(self.img_metadata)))
       
    def __len__(self):
        return len(self.img_metadata) if self.split == 'train' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name = self.img_metadata[idx]

        query_img= self.read_img(query_name) 
        query_cmask = self.read_mask(query_name)
        
        ### mask filtering
        pseudo_class = self.filter_mask(query_cmask)
        class_sample = pseudo_class[random.randint(0,len(pseudo_class)-1)]

        padding_mask = np.zeros_like(query_cmask)
        query_mask= self.extract_mask(query_cmask, class_sample)
        query_img, query_mask, padding_mask = self.transform(query_img, query_mask, padding_mask)

        # sample support images
        support_names = self.similar_prior_sample_episode(query_name, class_sample)
        support_imgs = [self.read_img(name) for name in support_names]
        support_cmasks = [self.read_mask(name) for name in support_names]

        support_masks = []
        support_padding_masks = []
        for scmask in support_cmasks:
            support_mask = self.extract_mask(scmask, class_sample)
            support_padding_mask = np.zeros_like(support_mask)
            support_padding_mask[support_mask==255] = 255
            support_masks.append(support_mask)
            support_padding_masks.append(support_padding_mask)
        for k in range(self.shot):
                support_imgs[k], support_masks[k], support_padding_masks[k] = self.transform(support_imgs[k], support_masks[k], support_padding_masks[k])

        s_xs = support_imgs
        s_ys = support_masks
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        if support_padding_masks is not None:
            s_eys = support_padding_masks
            s_ey = s_eys[0].unsqueeze(0)
            for i in range(1, self.shot):
                s_ey = torch.cat([s_eys[i].unsqueeze(0), s_ey], 0)

        if support_padding_masks is not None:
            a_eys = support_padding_masks
            a_ey = a_eys[0].unsqueeze(0)
            for i in range(1, self.shot):
                a_ey = torch.cat([a_eys[i].unsqueeze(0), a_ey], 0)

        if self.split == 'train':
            return query_img, query_mask, s_x, s_y, padding_mask, s_ey

    def compute_similar_dict(self):
        similar_dict={}
        for clu, feat_list in self.mask_feature_classwise.items():
            assert len(feat_list) == len(self.img_metadata_classwise[clu]), 'nononono'
            print(clu, len(feat_list))
            if len(feat_list) == 0:
                continue
            feats = torch.stack(feat_list, dim=0).cuda()
            if len(feat_list) < 20:
                k_sample = len(feat_list)
                real_indices = torch.arange(len(feat_list)).repeat(len(feat_list), 1)
            else:
                k_sample = min(int(len(feat_list) * 0.5), 5000)
                if feats.shape[0] > 20000:
                    candis = np.random.choice(list(range(feats.shape[0])), 20000, replace=False)
                    indices = compute_cosine_similarity(feats[candis, :], feats, k=k_sample)
                    real_indices = []
                    for i in range(feats.shape[0]):
                        r = torch.tensor(candis[indices[i].cpu()])
                        real_indices.append(r)
                    real_indices = torch.stack(real_indices, dim=0)
                else:
                    real_indices = compute_cosine_similarity(feats, feats, k=k_sample)
            similar_dict[clu] = real_indices.cpu()

        return similar_dict


    def extract_mask(self, mask, class_id):
        mask_ = np.zeros_like(mask)
        mask_[mask == class_id] = 1
        mask_[mask==255] = 255

        return mask_

    def filter_mask(self, query_cmask):
        pseudo_class = np.unique(query_cmask).tolist()
        if 255 in pseudo_class:
            pseudo_class.remove(255)
        clu_class = pseudo_class.copy()
        bkg = []
        for clu in clu_class:
            tmp_label = np.zeros_like(query_cmask)
            target_pix = np.where(query_cmask == clu)
            tmp_label[target_pix[0],target_pix[1]] = 1
            if target_pix[0].shape[0] < 2 * 32 * 32:
                pseudo_class.remove(clu)
            elif self.filter_background(tmp_label):
                pseudo_class.remove(clu)
                bkg.append(clu)
                
        if len(pseudo_class) == 0:
            print('no mask remain after filtering the background')
            # pseudo_class = bkg

        return pseudo_class

    def center_prior_filter_mask(self, query_cmask):
        pseudo_class = np.unique(query_cmask).tolist()
        if 255 in pseudo_class:
            pseudo_class.remove(255)
        dists = []
        areas = []
        pseudo_scores = [] 
        clu_class = pseudo_class.copy()
        bkg = []
        for i, clu in enumerate(clu_class):
            tmp_label = np.zeros_like(query_cmask)
            target_pix = np.where(query_cmask == clu)
            tmp_label[target_pix[0],target_pix[1]] = 1
            if target_pix[0].shape[0] < 2 * 32 * 32:
                pseudo_class.remove(clu)
            elif self.filter_background(tmp_label):
                pseudo_class.remove(clu)
                bkg.append(clu)
            else:
                center = np.mean(target_pix, axis=1)
                center_gt = np.array(query_cmask.shape)/2
                dist = np.linalg.norm(center-center_gt, ord=2)
                dists.append(dist)
        assert len(pseudo_class) == len(dists)
        pseudo_scores = np.array((-torch.tensor(dists)).softmax(dim=0))

                
        if len(pseudo_class) == 0:
            print('no mask remain after filtering the background')
            pseudo_class = bkg
            pseudo_scores = np.array([])

        if pseudo_scores.shape[0] > 0:
            class_sample = np.random.choice(pseudo_class, p = pseudo_scores)
        else:
            class_sample = pseudo_class[random.randint(0,len(pseudo_class)-1)]
        return class_sample

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = np.array(Image.open(os.path.join(self.pseudo_path, img_name) + '.png')).astype(np.float32)
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        image = cv2.imread(os.path.join(self.img_path, img_name+'.jpg'), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        return image

    def sample_episode(self, query_name, class_sample):
        img_list = self.img_metadata_classwise[class_sample]
        support_names = []
        while True:  # keep sampling support set if query == support
            selected = np.random.choice(img_list, 1, replace=False)[0]
            support_name = selected
            if query_name != support_name: 
                support_names.append(support_name)
            if len(support_names) == self.shot: break

        return support_names

    def similar_prior_sample_episode(self, query_name, class_sample):
        img_list = self.img_metadata_classwise[class_sample]
        query_idx = img_list.index(query_name)
        indices = self.similar_dict[class_sample][query_idx]
        support_names = []
        while True:  # keep sampling support set if query == support
            selected = np.random.choice(indices, 1, replace=False)[0]
            support_name = img_list[selected]
            if query_name != support_name: 
                support_names.append(support_name)
            if len(support_names) == self.shot: break

        return support_names

    def build_class_ids(self):
        if not self.use_coco:
            if self.fold == 3: 
                sub_list = list(range(1, 16)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                sub_val_list = list(range(16, 21)) #[16,17,18,19,20]
            elif self.fold == 2:
                sub_list = list(range(1, 11)) + list(range(16, 21)) #[1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                sub_val_list = list(range(11, 16)) #[11,12,13,14,15]
            elif self.fold == 1:
                sub_list = list(range(1, 6)) + list(range(11, 21)) #[1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                sub_val_list = list(range(6, 11)) #[6,7,8,9,10]
            elif self.fold == 0:
                sub_list = list(range(6, 21)) #[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                sub_val_list = list(range(1, 6)) #[1,2,3,4,5]
            elif self.fold == 5: 
                sub_val_list = list(range(1, 21))
                sub_list = []

        else:
            if self.use_split_coco:
                print('INFO: using SPLIT COCO')
                class_list = list(range(1, 81))
                if self.fold == 3:
                    sub_val_list = list(range(4, 81, 4))
                    sub_list = list(set(class_list) - set(sub_val_list))                    
                elif self.fold == 2:
                    sub_val_list = list(range(3, 80, 4))
                    sub_list = list(set(class_list) - set(sub_val_list))    
                elif self.fold == 1:
                    sub_val_list = list(range(2, 79, 4))
                    sub_list = list(set(class_list) - set(sub_val_list))    
                elif self.fold == 0:
                    sub_val_list = list(range(1, 78, 4))
                    sub_list = list(set(class_list) - set(sub_val_list))    
                elif self.fold == 5: 
                    sub_val_list = [1,2,3,4,5,6,7,9,15,16,17,18,19,20,40,57,58,59,61,66]
                    sub_list = []
            else:
                print('INFO: using COCO')
                if self.fold == 3:
                    sub_list = list(range(1, 61))
                    sub_val_list = list(range(61, 81))
                elif self.fold == 2:
                    sub_list = list(range(1, 41)) + list(range(61, 81))
                    sub_val_list = list(range(41, 61))
                elif self.fold == 1:
                    sub_list = list(range(1, 21)) + list(range(41, 81))
                    sub_val_list = list(range(21, 41))
                elif self.fold == 0:
                    sub_list = list(range(21, 81)) 
                    sub_val_list = list(range(1, 21))    

        print('sub_list: ', sub_list)
        print('sub_val_list: ', sub_val_list)    

        return sub_list, sub_val_list

    def filter_background(self, label):
        h, w = label.shape
        r = np.sum(label,axis=1)
        c = np.sum(label,axis=0)
        assert r.shape[0]==h, 'Wrong match with bkg'
        if np.sum(r==0)/h < 0.01 or np.sum(c==0)/w < 0.01:
            # print(np.sum(r==0), np.sum(c==0))
            return True
        else:
            return False

    def make_dataset(self, filter_intersection=False):
        assert self.fold in [0, 1, 2, 3]
        if not os.path.isfile(self.data_list):
            raise (RuntimeError("Image list file do not exist: " + self.data_list + "\n"))

        image_label_list = []
        list_read = open(self.data_list).readlines()
        print("Processing data...".format(self.sub_set))

        # sample support-query pairs from all cluster categories
        support_file_list = {}
        for clus in self.class_ids:
            support_file_list[clus] = []

        support_feat_list = {}
        for clus in self.class_ids:
            support_feat_list[clus] = []

        ### load pre-trained model 【MOCO】
        if self.extractor == 'moco':
            pthpath = './weights/moco_v2_800ep_pretrain.pth.tar'
        elif self.extractor =='dino':
            pthpath = './weights/dino_vit_B_8.pth'
        elif self.extractor == 'simclr':
            pthpath = './weights/simclr_resnet50-1x.pth'
        else:
            print('no weight file for feature extraction!!')
            exit(1)
        if self.extractor == 'moco':
            model_ft = resnet50_(pretrained=False)
            ckpt = torch.load(pthpath)['state_dict']
            format_dict = {k.replace('module.encoder_q.', ''): v  for k, v in ckpt.items() if not 'fc' in k}
            ckpt = format_dict
        elif self.extractor =='dino':
            model_ft = vit_base(patch_size=8)
            ckpt = torch.load(pthpath)
        else:
            model_ft = resnet50x1()
            ckpt = torch.load(pthpath)['state_dict']

        model_ft.load_state_dict(ckpt)
        device_ft = "cuda:0"
        model_ft.to(device_ft)
        model_ft.eval()

        for l_idx in tqdm(range(len(list_read))):
            line = list_read[l_idx]
            line_split = line.strip().split(' ')
            image_name = line_split[0].split('/')[-1][:-4]
            img = self.read_img(image_name)

            ### extract img feature
            img = self.transform_ft(img)
            with torch.no_grad():
                img_feat = model_ft(img.unsqueeze(0).to(device_ft)).detach()
            
            label = cv2.imread(os.path.join(self.ann_path, image_name) + '.png', cv2.IMREAD_GRAYSCALE)
            label_class = np.unique(label).tolist()
            
            if 0 in label_class:
                label_class.remove(0)
            if 255 in label_class:
                label_class.remove(255)

            new_label_class = []     

            if filter_intersection:  # filter images containing objects of novel categories during meta-training
                if set(label_class).issubset(set(self.sub_set)):
                    for c in label_class:
                        if c in self.sub_set:
                            tmp_label = np.zeros_like(label)
                            target_pix = np.where(label == c)
                            tmp_label[target_pix[0],target_pix[1]] = 1 
                            if tmp_label.sum() >= 2 * 32 * 32:      
                                new_label_class.append(c)     
            else:
                for c in label_class:
                    if c in self.sub_set:
                        tmp_label = np.zeros_like(label)
                        target_pix = np.where(label == c)
                        tmp_label[target_pix[0],target_pix[1]] = 1 
                        if tmp_label.sum() >= 2 * 32 * 32:      
                            new_label_class.append(c)            

            label_class = new_label_class

            if len(label_class) > 0:
                pseudo_label = torch.tensor(np.asarray(Image.open(os.path.join(self.pseudo_path, image_name) + '.png'), dtype=np.int32))
                pseudo_class = self.filter_mask(pseudo_label)
                if len(pseudo_class) > 0:
                    image_label_list.append(image_name)
                p_label = np.asarray(Image.open(os.path.join(self.pseudo_path, image_name) + '.png'), dtype=np.int32)
                clus_class = np.unique(p_label).tolist()
                if 255 in clus_class:
                    clus_class.remove(255)
                for clu in clus_class:
                    tmp_label = np.zeros_like(p_label)
                    target_pix = np.where(p_label == clu)
                    tmp_label[target_pix[0],target_pix[1]] = 1 
                    if not self.filter_background(tmp_label):
                        if tmp_label.sum() >= 2 * 32 * 32:    
                            support_file_list[clu].append(image_name)
                            support_feat_list[clu].append(get_mask_embedding(img_feat, torch.tensor(tmp_label).to(device_ft)).cpu())
                        
        print("Checking image&label pair {} list done! ".format(self.fold))
        return image_label_list, support_file_list, support_feat_list


    def make_all_dataset(self):
        assert self.fold == 5
        if not os.path.isfile(self.data_list):
            raise (RuntimeError("Image list file do not exist: " + self.data_list + "\n"))
 
        image_label_list = []
        # list_read = json.load(open(self.data_list))
        list_read = open(self.data_list).readlines()

        # sample support-query pairs from all cluster categories
        support_file_list = {}
        for clus in self.class_ids:
            support_file_list[clus] = []

        support_feat_list = {}
        for clus in self.class_ids:
            support_feat_list[clus] = []

        ### load pre-trained model 【MOCO】
        if self.extractor == 'moco':
            pthpath = './weights/moco_v2_800ep_pretrain.pth.tar'
        elif self.extractor =='dino':
            pthpath = './weights/dino_vit_B_8.pth'
        elif self.extractor == 'simclr':
            pthpath = './weights/simclr_resnet50-1x.pth'
        else:
            print('no weight file for feature extraction!!')
            exit(1)
        if self.extractor == 'moco':
            model_ft = resnet50_(pretrained=False)
            ckpt = torch.load(pthpath)['state_dict']
            format_dict = {k.replace('module.encoder_q.', ''): v  for k, v in ckpt.items() if not 'fc' in k}
            ckpt = format_dict
        elif self.extractor =='dino':
            model_ft = vit_base(patch_size=8)
            ckpt = torch.load(pthpath)
        else:
            model_ft = resnet50x1()
            ckpt = torch.load(pthpath)['state_dict']
        
        model_ft.load_state_dict(ckpt)
        device_ft = "cuda:0"
        model_ft.to(device_ft)
        model_ft.eval()

        for l_idx in tqdm(range(len(list_read))):
            line = list_read[l_idx]
            # line_split = line
            line_split = line.strip().split(' ')
            image_name = line_split[0].split('/')[-1][:-4]
            img = self.read_img(image_name)

            ### extract img feature
            # print('name', image_name, 'img', img.shape)
            img = self.transform_ft(img)
            with torch.no_grad():
                img_feat = model_ft(img.unsqueeze(0).to(device_ft)).detach()

            pseudo_label = torch.tensor(self.read_mask(image_name))
            pseudo_class = self.filter_mask(pseudo_label)
            if len(pseudo_class) > 0:
                image_label_list.append(image_name)
                proto_feats = get_mask_pool(img_feat, pseudo_label.to(device_ft), pseudo_class)
                for clu in pseudo_class:
                    support_file_list[clu].append(image_name)
                    support_feat_list[clu].append(proto_feats[clu])
                        
        print("Checking image&label pair {} list done! ".format(self.fold))
        return image_label_list, support_file_list, support_feat_list