import ot, torch
import numpy as np
import scipy.linalg
from utils import kmeans_dist, error
from sklearn.metrics.cluster import adjusted_rand_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MMD(x, y):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    rxz = (xx.diag().unsqueeze(1).expand_as(zz))
    ryz = (yy.diag().unsqueeze(0).expand_as(zz))

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rxz + ryz - 2. * zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(yy.shape).to(device),
                  torch.zeros(zz.shape).to(device))

    bandwidth = 1
    XX += torch.exp(-0.5 * dxx / bandwidth)
    XX = XX * (1 - torch.eye(XX.shape[0], XX.shape[1]).to(device=device))
    YY += torch.exp(-0.5 * dyy / bandwidth)
    YY = YY * (1 - torch.eye(YY.shape[0], YY.shape[1]).to(device=device))
    XY += torch.exp(-0.5 * dxy / bandwidth)
    return torch.sum(XX) / (XX.shape[0] * (XX.shape[0] - 1)) + torch.sum(YY) / (
                YY.shape[0] * (YY.shape[0] - 1)) - torch.sum(2. * XY) / (XX.shape[0] * YY.shape[0])


def align_embedding(x, y):
    n, d = x.shape
    m = y.shape[0]
    w = torch.eye(d, dtype=torch.float64).to(device=device)
    pi = (1 / (n * m)) * torch.ones((n, 1), dtype=torch.float64).to(device=device) @ torch.ones((1, m),
                                                                                                dtype=torch.float64).to(
        device=device)
    cost = torch.zeros((n, m), dtype=torch.float64).to(device=device)
    eps = 0.01
    itr = 0

    while itr < 1:
        M = w.t() @ x.t() @ pi @ y
        u, s, v = torch.svd(M)
        w_new = u @ v.t()
        for i in range(n):
            row_norm = x[i] * torch.ones(y.shape, dtype=torch.float64).to(device=device) - y
            cost[i] = torch.norm(row_norm, dim=1) ** 2

        c = cost.cpu().detach().numpy()
        a = ot.utils.unif(n)
        b = ot.utils.unif(m)
        pi = torch.DoubleTensor(ot.sinkhorn(a, b, c, eps, numItermax=100)).to(device=device)
        d = torch.trace(cost @ pi.t())
        w = w_new
        itr += 1
    return w, pi, d


def calc_mmd(graphs):
    m = len(graphs)
    dist = torch.zeros((m, m), dtype=torch.float64).to(device=device)
    for i in range(m):
        for j in range(i + 1):
            w, pi, d = align_embedding(graphs[i], graphs[j])
            dist[i][j] = MMD(graphs[i], graphs[j] @ w)
            dist[j][i] = dist[i][j]
    return dist


def simulate_ncmmd(graphs, gt, rank, num_clusters):
    graphs_eig_vec = []
    graphs_eig_val = []
    alpha_graphs = []
    m = len(graphs)
    skipped_graphs = []
    for i in range(m):
        g = graphs[i]
        a = g.t() @ g
        a = a.detach().cpu().numpy().astype(np.float_)
        adj = torch.from_numpy(scipy.linalg.sqrtm(a).real).to(device=device)
        adj_nan = adj.isnan().nonzero().nelement()
        if adj_nan != 0:
            skipped_graphs.append(i)
            w = torch.empty(1)
            v = torch.empty(1)
        else:
            w, v = torch.eig(adj, eigenvectors=True)
        # calc edge density
        alpha = torch.sqrt(torch.sum(g) / (g.shape[0] * (g.shape[0] - 1)))
        graphs_eig_val.append(w)
        graphs_eig_vec.append(v)
        alpha_graphs.append(alpha)

    print('skipped graphs ', skipped_graphs)
    gt = np.delete(gt, skipped_graphs)
    labels_rank = []
    err_rank = []
    rand_idx_rank = []
    for r in rank:
        reduced_graphs = []
        for i in range(m):
            if i not in skipped_graphs:
                w = graphs_eig_val[i]
                v = graphs_eig_vec[i]
                w_real = w[:, 0]
                sorted_w = torch.argsort(-w_real)
                to_pick_idx = sorted_w[:r]
                eig_vec = v[:, to_pick_idx]
                eig_val = torch.diag(w_real[to_pick_idx]).to(device=device)
                eig_val = torch.sqrt(eig_val)
                adj_embedding = (eig_vec @ eig_val) / alpha_graphs[i]
                reduced_graphs.append(adj_embedding)
        print('reduced graphs ', len(reduced_graphs), reduced_graphs[0].shape)

        dist = calc_mmd(reduced_graphs)
        print('mmd calculated')
        labels = kmeans_dist(dist, num_clusters=num_clusters)
        labels_rank.append(labels)
        err = error(gt, labels)
        err_rank.append(err)
        ri = adjusted_rand_score(gt, labels)
        rand_idx_rank.append(ri)
        print('err, ri ', err, ri)
    return err_rank, rand_idx_rank, dist