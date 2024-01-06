import torch
import torch.nn.functional as F
# import faiss
import numpy as np

def get_mask_embedding(embedding,label):
    embedding = F.normalize(embedding, p=2, dim=1)
    embedding = F.interpolate(embedding,
                    size=label.shape,
                    mode='bilinear',
                    align_corners=True)
    embedding = embedding.squeeze(0).permute(1,2,0)
    avg_embedding = torch.mean(embedding[label==1], 0)
    return avg_embedding

def calculate_prototypes_from_labels(embedding, labels):

    # reshape embedding to a 2-D tensors
    embedding = embedding.permute(0, 2, 3, 1)  # (B, H, W, C)
    embedding = embedding.reshape(-1, embedding.shape[-1])  # (B * H * W, C)
    # convert label to one-hot encoding
    labels = labels.reshape(-1).long()
    one_hot_labels = F.one_hot(labels, num_classes=-1).float()
    # calculate prototypes
    one_hot_labels = one_hot_labels.permute(1, 0)  # (max_label, B * H * W)
    prototypes = torch.matmul(one_hot_labels, embedding)  # (max_label, C)

    pixel_count = one_hot_labels.sum(1, keepdim=True)
    if 0 in pixel_count:
        # there may be cluster label with zero pixels
        pixel_count[pixel_count == 0] = 1
    prototypes = prototypes / pixel_count  # (max_label, C)
    return prototypes

def get_mask_pool(embedding, label, classes):
    embedding = F.normalize(embedding, p=2, dim=1)
    embedding = F.interpolate(embedding,
                    size=label.shape,
                    mode='bilinear',
                    align_corners=True)

    clus_prototype = {}
    label = label.unsqueeze(0).unsqueeze(0)
    clus, label = torch.unique(label, sorted=False, return_inverse=True)
    prototypes = calculate_prototypes_from_labels(embedding, label)
    for i, p in enumerate(prototypes):
        if int(clus[i].cpu()) not in classes:
            continue
        clus_prototype[int(clus[i].cpu())] = p.cpu()
    
    return clus_prototype

def compute_cosine_similarity(vectors, query_vector, k=10):
    num_vectors, vector_dim = vectors.shape

    vector, query_vector = F.normalize(vectors,p=2,dim=1), F.normalize(query_vector,p=2,dim=1)

    simi = query_vector.mm(vector.t())
    _, I = torch.topk(simi, k, dim=1)

    return I

def compute_L2_distances(vectors, query_vector, k=10):
    A = (query_vector**2).sum(dim=1, keepdims=True)
    B = (vectors**2).sum(dim=1, keepdims=True).t() 
    dist = - (A + B - 2* query_vector.mm(vectors.t())) 

    _, I = torch.topk(dist, k, dim=1)

    return I

