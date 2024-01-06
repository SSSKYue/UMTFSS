# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from unittest.mock import patch
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import time
from data.cluster_dataset import CLusterData, get_data_loader
from assign_labels import *
from cluster import run_mini_batch_kmeans
import sys 
sys.path.append("..") 
from model.extractor_model.resnet import resnet50_
from model.extractor_model.vit import vit_base
from model.extractor_model.resnet_simclr import resnet50x1


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pascal', help='dataset to cluseter')
    parser.add_argument('--data_root', type=str, default='../data', help='dataset root')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--arch', type=str, default='moco', 
                        choices=['moco', 'dino', 'simclr'], 
                        help='feature extractor architecture')
    parser.add_argument('--mode', type=str, default='cluster', choices=['cluster', 'assign'])
    parser.add_argument('--in_dim', type=int, default=2048, help='dim of feature embeddings to cluster')
    parser.add_argument('--sample_num', type=int, default=100000, help='number of images to perform cluster')
    parser.add_argument('--pca_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=30, help='number of cluster centroids')
    parser.add_argument('--niter', type=int, default=30, help='kmeans n iters')
    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.dataset == 'pascal':
        data_root = os.path.join(args.data_root, 'pascal')
    elif args.dataset == 'coco':
        data_root = os.path.join(args.data_root, 'COCO2014')

    data_list = {x: '../lists/'+args.dataset+'/{}.txt'.format(x) for x in ['train', 'val']}
    label_root = os.path.join(data_root, args.dataset+'_'+args.arch+'_pseudo')

    # chose pre-trained model
    if args.arch =='moco':
        pthpath = '../weights/moco_v2_800ep_pretrain.pth.tar'
        model_ft = resnet50_(pretrained=False)
        ckpt = torch.load(pthpath)['state_dict']
        format_dict = {k.replace('module.encoder_q.', ''): v  for k, v in ckpt.items() if not 'fc' in k}
        model_ft.load_state_dict(format_dict)
    elif args.arch == 'dino':
        pthpath = '../weights/dino_vit_B_8.pth'
        model_ft = vit_base(patch_size=8)
        ckpt = torch.load(pthpath)
        model_ft.load_state_dict(ckpt)
    elif args.arch == 'simclr':
        pthpath = '../weights/simclr_resnet50-1x.pth'
        model_ft = resnet50x1()
        ckpt = torch.load(pthpath)['state_dict']
        model_ft.load_state_dict(ckpt,strict=True)
    
    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.cuda()

    if not os.path.exists('centroids'):
            os.makedir('centroids')
    centroids_path = os.path.join('centroids', 'centroids_{}_{}_{}.npy'.format(args.arch, args.dataset, args.k))

    if args.mode == 'cluster':
        # build dataloader
        # load dataset
        dataset = CLusterData(args.mode, data_root, data_list['train'], args.dataset, use_spl=False, sample_num=args.sample_num)
        # print('train_datasete_size:', len(dataset))
        dataloader = get_data_loader(args,dataset)
        
        # K-means Cluster
        print("run")
        t0 = time.time()
        args.num_init_batches = 100  # set according to your cuda memory, real batch_size = num_init_batches * batch_size
        centroids, obj = run_mini_batch_kmeans(args, model_ft, dataloader)
        print("final objective: %.4g" % obj)
        np.save(centroids_path, centroids.cpu().numpy())
        t1 = time.time()
        print("total runtime: %.3f s" % (t1 - t0))

    elif args.mode == 'assign':
        # Assign labels per image
        # args.batch_size = 1

        dataset_sup = CLusterData(args.mode, data_root, data_list['train'], args.dataset, use_spl=True, sample_num=-1)
        dataloader_sup = get_data_loader(args,dataset_sup)

        print('centroids saved at ', centroids_path)
        centroids = np.load(centroids_path)
        centroids = torch.from_numpy(centroids).cuda()

        # print('centroids', centroids.shape)

        compute_labels(args, centroids, dataloader=dataloader_sup, model=model_ft, savepath=label_root)
 
if __name__ == '__main__':
    main()
