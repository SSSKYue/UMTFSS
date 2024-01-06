import torch
import os
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import PIL.Image as Image
import imgviz
from utils import *

def compute_labels(args, centroids, dataloader, model, savepath):
    """
    Label all images with the obtained cluster centroids. 
    The distance is efficiently computed by setting centroids as convolution layer. 
    """
    os.makedirs(savepath, exist_ok=True)
    
    if args.dataset == 'pascal':
        num_cls = 21
    elif args.dataset == 'coco':
        num_cls = 81
    else:
        RuntimeError('Currently metric only support pascal or coco.')
        
    # Define metric function with conv layer. 
    metric_function = get_metric_as_conv(centroids)
    assert model is not None
    model.eval()
    with torch.no_grad():
        for images, ori_size, name, superpixel in tqdm(dataloader):
            image = images.cuda(non_blocking=True)
            feats = model(image).detach()
            # Normalize.
            feats = F.normalize(feats, p=2, dim=1)
            # Compute distance and assign label.
            scores  = compute_negative_euclidean(feats, centroids, metric_function) 
            for b in range(feats.shape[0]):                
                score = F.interpolate(scores[b].unsqueeze(0), size=superpixel[b].shape[-2:],mode='bilinear',align_corners=True)
                score, label = score[0].max(dim=0)
                spl = superpixel[b].long()

                # compute superpixel max pool
                spl_num = spl.max()

                # calculate vote cluster labels
                sp_label = torch.zeros_like(label)

                for i in range(spl_num):
                    if label[spl==i].shape[0] == 0:
                        continue
                    votes = torch.bincount(label[spl==i])
                    l = votes.argmax()
                    sp_label[spl==i] = l

                # sp_label[gt==255] = 255
                save_label(sp_label, os.path.join(savepath, name[b] +'.png'))

     
def save_label(label, path):
    label = label.cpu().numpy()
    dst = Image.fromarray(label.astype(np.uint8), 'P')
    colormap = imgviz.label_colormap()
    dst.putpalette(colormap.flatten())
    # dst.putpalette()
    dst.save(path)
