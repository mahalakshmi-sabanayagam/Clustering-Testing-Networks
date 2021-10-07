from wwl import wwl
import igraph
from utils import spectral_clustering, error
from sklearn.metrics.cluster import adjusted_rand_score


def numpy_array_to_igraph(a):
    sources, targets = a.nonzero()
    edgelist = zip(sources.tolist(), targets.tolist())
    g = igraph.Graph(edgelist)
    return g

def simulate_wwlgk(graphs_tensor, gt, num_clusters=3):
    graphs = []
    for g in graphs_tensor:
        ig = numpy_array_to_igraph(g.cpu().detach().numpy())
        graphs.append(ig)

    wwl_kernel = wwl(graphs)

    l = spectral_clustering(wwl_kernel, num_clusters=num_clusters)
    err = error(gt, l)
    rand_idx = adjusted_rand_score(gt, l)
    print('wwl error ', err)
    print('wwl rand idx  ', rand_idx)

    return err, rand_idx