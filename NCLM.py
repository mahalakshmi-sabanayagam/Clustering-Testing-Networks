import torch
import time
from utils import kmeans_dist, error
from sklearn.metrics.cluster import adjusted_rand_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# clustering algorithm using log moments
def log_moments(adj, j=8):
    n = adj.shape[0]
    m = torch.FloatTensor([torch.trace(torch.matrix_power(adj/n, i)) for i in range(2,j+2)]).to(device=device)
    m = torch.log(m)
    #sometimes -inf appears because of trace being 0 -- in those cases replace it with the second smallest value
    m_inf = torch.isinf(m)
    x = torch.clone(m)
    x[m_inf] = 0
    return x

def distance_matrix_log_moments(m):
    # m : n*j calc dist between all pairs of n using j dim features
    n = m.shape[0]
    dist = torch.zeros((n, n), dtype=torch.float64).to(device=device)
    for i in range(n):
        for j in range(n):
            fi = m[i]
            fj = m[j]
            dist[i][j] = torch.norm(fi-fj)
    return dist

def nclm(all_graphs, j=8, return_dist=False, num_clusters=2):
    n = len(all_graphs)
    m = torch.zeros((n,j)).to(device=device)
    for i in range(n):
        m[i] = log_moments(all_graphs[i],j)

    dist = distance_matrix_log_moments(m)
    labels = kmeans_dist(dist, num_clusters)
    if return_dist:
        return dist
    return labels

def simulate_nclm(graphs, gt, j=[8], num_clusters=2):
    err = []
    rand_score = []
    time_nclm = []
    for i in j:
        time_start = time.time()
        labels_exact = nclm(graphs, j=i, num_clusters=num_clusters)
        time_nclm.append(time.time()-time_start)
        err.append(error(gt, labels_exact))
        rand_score.append(adjusted_rand_score(gt,labels_exact))
    print('Time for NCLM ',time_nclm)
    return err, rand_score