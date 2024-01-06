import torch
import numpy as np
import faiss

class PCA(object):
    def __init__(self, output_dim):
        self.output_dim = output_dim

    def fit(self, X):
        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj_mat = eigenvectors[:, 0:self.output_dim]

    def transform(self, X):
        X = X - self.mean
        return X.matmul(self.proj_mat)


def get_metric_as_conv(centroids):
    N, C = centroids.size()

    centroids_weight = centroids.unsqueeze(-1).unsqueeze(-1)
    metric_function  = torch.nn.Conv2d(C, N, 1, padding=0, stride=1, bias=False)
    metric_function.weight.data = centroids_weight
    metric_function = torch.nn.DataParallel(metric_function)
    metric_function = metric_function.cuda()
    
    return metric_function

def compute_negative_euclidean(featmap, centroids, metric_function):
    centroids = centroids.unsqueeze(-1).unsqueeze(-1)
    return - (1 - 2*metric_function(featmap)\
                + (centroids*centroids).sum(dim=1).unsqueeze(0)) # negative l2 squared 

def get_init_centroids(args, K, featlist, index):
    clus = faiss.Clustering(args.in_dim, K)
    clus.seed  = np.random.randint(args.seed)
    clus.niter = args.niter
    clus.max_points_per_centroid = 10000000
    clus.train(featlist, index)

    return faiss.vector_float_to_array(clus.centroids).reshape(K, args.in_dim)

def kmeans(args, K, d, featlist, index):
    clus = faiss.Clustering(d, K)
    clus.seed  = np.random.randint(args.seed)
    clus.niter = args.niter
    clus.max_points_per_centroid = 10000000
    clus.train(featlist, index)

    return faiss.vector_float_to_array(clus.centroids).reshape(K, d)

def module_update_centroids(index, centroids):
    index.reset()
    index.add(centroids)

    return index 

def get_faiss_module(args):
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False 
    cfg.device     = 0 #NOTE: Single GPU only. 
    idx = faiss.GpuIndexFlatL2(res, args.in_dim, cfg)

    return idx

def get_multi_gpu_module(args):
    ngpu = int(args.ngpu) #gpu数

    res = [faiss.StandardGpuResources() for i in range(ngpu)]
    # first we get StandardGpuResources of each GPU
    # ngpu is the num of GPUs
    useFloat16 = False 
    indexes = [faiss.GpuIndexFlatL2(res[i], i, args.in_dim, useFloat16)
            for i in range(ngpu)]
    # then we make an Index array
    # useFloat16 is a boolean value

    index = faiss.IndexProxy()
    for sub_index in indexes:
        index.addIndex(sub_index)
    # build the index by IndexProxy

    # cpu_index = faiss.IndexFlatL2(args.in_dim)

    # gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    #     cpu_index
    # )
    # idx = faiss.GpuIndexFlatL2(res, args.in_dim, cfg)

    return index

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count