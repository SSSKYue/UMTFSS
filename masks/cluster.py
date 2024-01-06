import torch
import numpy as np
from utils import *
from tqdm import tqdm
import torch.nn.functional as F

def feature_flatten(feats):
    if len(feats.size()) == 2:
        # feature already flattened. 
        return feats
    
    feats = feats.view(feats.size(0), feats.size(1), -1).transpose(2, 1)\
            .contiguous().view(-1, feats.size(1))
    
    return feats 

def run_mini_batch_kmeans(args, model, dataloader):
    """
    num_init_batches: (int) The number of batches/iterations to accumulate before the initial k-means clustering.
    num_batches     : (int) The number of batches/iterations to accumulate before the next update. 
    """
    kmeans_loss  = AverageMeter()
    faiss_module = get_faiss_module(args)
    data_count   = np.zeros(args.k)
    featslist    = []
    num_batches  = 0
    first_batch  = True
    args.seed = 2022
    
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(tqdm(dataloader)):
            images, ori_size, name = batch
            # 1. Compute initial centroids from the first few batches. 
            if isinstance(images, list):
                mlvs = len(images)
                mlv_fmap = []
                for i in range(mlvs):
                    image = images[i].cuda(non_blocking=True)
                    feature = model(image)
                    scaled_featuer = F.interpolate(feature, size=(int(ori_size[0]/8), int(ori_size[1]/8)), mode='bilinear', align_corners=True)
                    mlv_fmap.append(scaled_featuer)
                fuse_feature = torch.cat(mlv_fmap, dim=1)
            else:
                images = images.cuda(non_blocking=True)
                fuse_feature = model(images).detach()

            # Normalize.
            # print('feats', fuse_feature.shape)
            fuse_feature = F.normalize(fuse_feature, p=2, dim=1)
            # feats = fuse_feature.squeeze().flatten(1)
            # feats = feats.permute(1,0)
            if i_batch == 0:
                print('Batch input size : {}'.format(list(images.shape)))
                print('Batch feature : {}'.format(list(fuse_feature.shape)))
            
            feats = feature_flatten(fuse_feature).detach().cpu()
            # print('feats', feats.shape, feats.unique())            

            if num_batches < args.num_init_batches:
                featslist.append(feats)
                num_batches += 1
                
                if num_batches == args.num_init_batches or num_batches == len(dataloader):
                    if first_batch:
                        # Compute initial centroids. 
                        # By doing so, we avoid empty cluster problem from mini-batch K-Means. 
                        featslist = torch.cat(featslist).numpy().astype('float32')
                        print('featslist', featslist.shape)
                        centroids = get_init_centroids(args, args.k, featslist, faiss_module).astype('float32')
                        D, I = faiss_module.search(featslist, 1)

                        kmeans_loss.update(D.mean())
                        print('Initial k-means loss: {:.8f} '.format(kmeans_loss.avg))
                        
                        # Compute counts for each cluster. 
                        for k in np.unique(I):
                            data_count[k] += len(np.where(I == k)[0])
                        first_batch = False
                    else:
                        b_feat = torch.cat(featslist).cpu().numpy().astype('float32')
                        faiss_module = module_update_centroids(faiss_module, centroids)
                        D, I = faiss_module.search(b_feat, 1)

                        kmeans_loss.update(D.mean())
                        print('k-means loss: {:.8f} '.format(kmeans_loss.avg))

                        # Update centroids. 
                        for k in np.unique(I):
                            idx_k = np.where(I == k)[0]
                            data_count[k] += len(idx_k)
                            centroid_lr    = len(idx_k) / (data_count[k] + 1e-6)
                            centroids[k]   = (1 - centroid_lr) * centroids[k] + centroid_lr * b_feat[idx_k].mean(0).astype('float32')
                    
                    # Empty. 
                    featslist   = []
                    num_batches = 0

            if (i_batch % args.num_init_batches) == 0:
                print('[Loading features]: {} / {} | [K-Means Loss]: {:.4f}'.format(i_batch, len(dataloader), kmeans_loss.avg))

    centroids = torch.tensor(centroids, requires_grad=False).cuda()

    return centroids, kmeans_loss.avg