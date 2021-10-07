import torch
import numpy as np
import time
from utils import kmeans_dist, error
from sklearn.metrics.cluster import adjusted_rand_score
from graspologic.match import GraphMatch as GMP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gmp_dist(graphs, gmp):
    n = len(graphs)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1):
            G1 = graphs[i].cpu().detach().numpy()
            G2 = graphs[j].cpu().detach().numpy()

            if G1.shape[0] > G2.shape[0]:
                G2 = np.pad(G2, [(0, G1.shape[0] - G2.shape[0]), (0, G1.shape[0] - G2.shape[0])], mode='constant')
            elif G1.shape[0] < G2.shape[0]:
                G1 = np.pad(G1, [(0, G2.shape[0] - G1.shape[0]), (0, G2.shape[0] - G1.shape[0])], mode='constant')

            gmp = gmp.fit(G2, G1)
            G1_rotated = G1[np.ix_(gmp.perm_inds_, gmp.perm_inds_)]
            dist[i][j] = np.linalg.norm(G1_rotated - G2)
            dist[j][i] = dist[i][j]
    dist = torch.tensor(dist).to(device=device)
    return dist

def simulate_ncgmm(graphs, gt, num_clusters):
    time_start = time.time()
    gmp = GMP(padding='naive')
    dist = gmp_dist(graphs, gmp)
    print('dist of matched graphs calculated')
    labels = kmeans_dist(dist, num_clusters=num_clusters)
    print('Time for NCGMM ', time.time() - time_start)
    err = error(gt, labels)
    ri = adjusted_rand_score(gt, labels)
    print('err, ri NCGMM ', err, ri)
    return err, ri, dist