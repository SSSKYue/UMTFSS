import torch
import torch.nn as nn
from model.clusterutils.hashing import compute_hashes
from model.clusterutils.clustering.hamming import cluster
from torch.nn.init import normal_
import time

class PCA(nn.Module):
    def __init__(self, output_dim):
        super(PCA, self).__init__()
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
    
    def forward(self, x):
        stat = time.time()
        self.fit(x)
        x_down = self.transform(x)
        end = time.time()
        print('time used for pca: ', end-stat)
        return x_down

class Cluster(nn.Module):
    def __init__(self, n_iter, down_dim=-1):
        super(Cluster, self).__init__()
        self.n_iter = n_iter
        # bits: How many bits to use for the hash (default: 32)
        self.bits = 32
        if down_dim > 0:
            self.pca = PCA(down_dim)
        else:
            self.pca = None
        # hash_bias: If true, hamming distance proportional to L2 distance, else cosine distance
        self.hash_bias = True

    def _create_supp_groups(self, Q, query_lengths, cluster_num):
        N, H, L, E = Q.shape

        # Compute the hashes for all the queries
        planes = Q.new_empty((self.bits, E+1))
        normal_(planes)
        if not self.hash_bias:
            planes[:, -1] = 0
        hashes = compute_hashes(Q.view(N*H*L, E), planes).view(N, H, L)
        _ = torch.unique(hashes)

        # Cluster the hashes and return the cluster index per query
        clusters, counts =  cluster(
            hashes,
            query_lengths,
            clusters=cluster_num,
            iterations=self.n_iter,
            bits=self.bits
        )
        return clusters.squeeze().long(), counts.squeeze()


    def lsh_kmeans(self, x, k):
        '''
        args:
            x: features to cluster, shape [L, E],
            k: cluster nums
        return:
            cluster_idxs: projection from x to k centroids, shape [L], value [0 ~ k)
            counts: numbers of x belongs to k-th centroids, shape [k]
        '''
        L, E = x.shape
        x = x.unsqueeze(0).unsqueeze(1)
        length = x.new_full((1,), L, dtype=torch.int32)
        return self._create_supp_groups(x, length, k)
    
    def forward(self, x, k):
        if self.pca:
            x = self.pca(x)
            return self.lsh_kmeans(x, k)
        else:
            return self.lsh_kmeans(x, k)
            