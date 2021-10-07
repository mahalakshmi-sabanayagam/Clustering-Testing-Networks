import torch
import numpy as np
import cvxpy as cp
from utils import spectral_clustering, kmeans_dist, error
from sklearn.metrics.cluster import adjusted_rand_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hist_apprx(graphs, n0=30):
    graphs_appr = []
    for graph in graphs:
        # degree sort
        nn = graph.shape[0]
        h = int(nn / n0)

        deg = torch.sum(graph, axis=1)
        id_sort = torch.argsort(-deg)

        graph_sorted = graph[id_sort]
        graph_sorted = graph_sorted[:, id_sort]

        # histogram approximation
        graph_apprx = torch.zeros((n0, n0), dtype=torch.float64).to(device=device)
        for i in range(n0):
            for j in range(i + 1):
                graph_apprx[i][j] = torch.sum(graph_sorted[i * h:i * h + h, j * h:j * h + h]) / (h * h)
                graph_apprx[j][i] = graph_apprx[i][j]

        graphs_appr.append(graph_apprx)

    return graphs_appr


def distance_matrix(all_graphs):
    m = len(all_graphs)
    dist = torch.zeros((m, m), dtype=torch.float64).to(device=device)
    for i in range(m):
        for j in range(i + 1):
            dist[i][j] = torch.norm(all_graphs[i] - all_graphs[j])  # /all_graphs[i].shape[0] not needed
            dist[j][i] = dist[i][j]
    return dist


def sim_matrix(all_graphs, sigma):
    m = len(all_graphs)
    sim = torch.zeros((m, m), dtype=torch.float64).to(device=device)
    for i in range(m):
        n = all_graphs[i].shape[0]
        for j in range(i + 1):
            sim[i][j] = torch.exp(-torch.norm(all_graphs[i] - all_graphs[j]) ** 2 / (sigma[i] * sigma[j]))
            sim[j][i] = sim[i][j]
    return sim


def sdp(sim, num_clusters=2):
    sim = sim.cpu().detach().numpy()
    # Define and solve the CVXPY problem.
    b = np.ones((sim.shape[0], 1), dtype=np.float64)
    # Create a symmetric matrix variable.
    X = cp.Variable((sim.shape[0], sim.shape[1]), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]
    constraints += [X @ b == b]
    constraints += [cp.trace(X) == num_clusters]
    prob = cp.Problem(cp.Maximize(cp.trace(sim @ X)),
                      constraints)
    prob.solve()

    return X


def simulate_histogram(graphs, gt, check_n0=[5,10,15,20,25,30], sigma=[5], num_clusters=3):
    frac_err_spect = []
    rand_idx_spect = []
    rand_idx_sdp = []
    all_err_sdp = []

    for n0 in check_n0:
        graphs_apprx = hist_apprx(graphs, n0)

        dist = distance_matrix(graphs_apprx)
        labels = kmeans_dist(dist, num_clusters=num_clusters)
        rand_idx_spect.append(adjusted_rand_score(gt, labels))
        frac_err_spect.append(error(gt, labels))

        # [5] for real data
        # [1,2,3] for simulation
        dist_sorted = torch.sort(dist, 1).values
        err_f = []
        rand_idx_f = []

        for s in sigma:
            sim = sim_matrix(graphs_apprx, dist_sorted[:, s])
            X = sdp(sim, num_clusters)

            l = spectral_clustering(X.value, num_clusters=num_clusters)
            err = error(gt, l)
            print('spec error for sigma ', n0, s, err)
            err_f.append(err)
            rand_idx_f.append(adjusted_rand_score(gt, l))

        all_err_sdp.append(err_f)
        rand_idx_sdp.append(rand_idx_f)

    frac_err_spect = np.array(frac_err_spect)

    print('err, ari spect  ', frac_err_spect, rand_idx_spect)
    print('error sdp all sigmas ', all_err_sdp)
    print('rand idx all sigmas ', rand_idx_sdp)
    return frac_err_spect, all_err_sdp, rand_idx_spect, rand_idx_sdp
