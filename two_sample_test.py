import torch
import numpy as np
from pyunlocbox import functions
from pyunlocbox import solvers
from utils import generate_graphs
from DSC_SSDP import hist_apprx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f1 = functions.norm_tv(maxit=50, dim=2)
tau = 1
g_fn = lambda x: x
solver = solvers.forward_backward(step=0.5/tau)

def estimators(graphon1, graphon2, n, m, n0,s=0):

    g1 = generate_graphs(graphon1,[n])
    g2 = generate_graphs(graphon2,[m])

    #for our considered n,m -- log() is 2, so hist size has to be n/2 and m/2
    h1 = hist_apprx(g1,n0=int(n/2))[0]
    h2 = hist_apprx(g2,n0=int(m/2))[0]
    h1 = h1.cpu().detach().numpy()
    h2 = h2.cpu().detach().numpy()

    f2 = functions.norm_l2(y=h1, A=g_fn, lambda_=tau)
    ret1 = solvers.solve([f1, f2], np.array(h1), solver, maxit=100)
    f2 = functions.norm_l2(y=h2, A=g_fn, lambda_=tau)
    ret2 = solvers.solve([f1, f2], np.array(h2), solver, maxit=100)

    if s == 0:
        k = int(max(n,m)/n0)
    else:
        k = int(s/n0)
    est1 = np.kron(ret1['sol'],np.ones((k,k)))
    est2 = np.kron(ret2['sol'],np.ones((k,k)))

    #to calc our distance use n0 and get the histograms
    h1 = hist_apprx(g1,n0=n0)[0]
    h2 = hist_apprx(g2,n0=n0)[0]
    T = torch.norm(h1-h2) /h1.shape[0]

    return est1, est2, T

def compute_T_boot(est,b,n,m,n0,T):
    est = torch.Tensor(est).to(device=device)
    p_est = []
    for i in range(b):
        gs = est.shape[0]
        labels = torch.randint(0, gs, size=(n,)).to(device=device)
        graph_prob = est[labels]
        graph_prob = graph_prob[:,labels]
        graph = torch.distributions.binomial.Binomial(1,graph_prob).sample()
        graph = torch.triu(graph, diagonal=1)
        graph = graph + graph.t() #n*n
        g_n = graph

        labels = torch.randint(0, gs, size=(m,)).to(device=device)
        graph_prob = est[labels]
        graph_prob = graph_prob[:,labels]
        graph = torch.distributions.binomial.Binomial(1,graph_prob).sample()
        graph = torch.triu(graph, diagonal=1)
        graph = graph + graph.t() #m*m
        g_m = graph

        h_n = hist_apprx([g_n],n0=n0)[0]
        h_m = hist_apprx([g_m],n0=n0)[0]

        t = torch.norm(h_n-h_m) /h_n.shape[0]
        p_est.append(t>=T)
    p_est = torch.Tensor(p_est).to(device=device)
    print("p estimated ",torch.sum(p_est),(torch.sum(p_est)+0.5)/b)
    return (torch.sum(p_est)+0.5)/b

def bootstraping(graphon1, graphon2, n,m,n0, bootstrap=100,trials=50, alpha=0.05, s=0):
    p_values = []
    for i in range(trials):
        print('trial ', i)
        est1, est2, T = estimators(graphon1, graphon2, n=n, m=m, n0=n0,s=s)
        print('T between graphs ',T)
        p1 = compute_T_boot(est1,bootstrap,n,m,n0,T)
        if p1 <= alpha:
            p2 = compute_T_boot(est2,bootstrap,n,m,n0,T)
            p_v = torch.max(p1,p2)
        else:
            p_v = p1
        print('p_val ', p_v)
        p_values.append(p_v)
    power = torch.sum(torch.Tensor(p_values) <= alpha)/trials
    return p_values, power