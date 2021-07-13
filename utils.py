import numpy as np
import gudhi as gd

from rpforest import RPForest

from scipy.spatial.distance import directed_hausdorff

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

    
def copy_tree(simplex_tree):
    new_tree = gd.SimplexTree()
    simplexes = simplex_tree.get_filtration()
    for sim, rad in simplexes:
        new_tree.insert(sim, filtration = rad)
    return new_tree

def add_t_hull(ax, X, simplex_tree, t, transparency = 1):
    ax.axis('off')
    ax.set_facecolor("white")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.patch.set_visible(False)
    
    tree = copy_tree(simplex_tree)
    tree.prune_above_filtration(t**2)
    simplexes = tree.get_filtration()

    isolated_points = []
    for sim, rad in simplexes:
        simplex = X[sim,:]
        if len(sim) == 2:
            line_segment = LineCollection([simplex], alpha = transparency)
            ax.add_collection(line_segment)
        if len(tree.get_star(sim)) <= 1: #that is sim is a maximal simplex
            if len(sim) == 1:
                isolated_points.append(sim[0])
            
            if len(sim) == 3:
                triangle = plt.Polygon([simplex[0,:], simplex[1,:], simplex[2,:]], alpha = transparency)    
                ax.add_artist(triangle)
    ax.scatter(X[isolated_points,0], X[isolated_points,1], marker = '.', s = 10)
    
    
lin = [np.linspace(0,1,L) for L in range(10)]

def hauss_line(X, a, b, neighbors, t):
    '''
    Compute the Hausdorff distance between the edge with endpoints X[a] and X[b], of length t, and the set X[neighbors]
    '''
    Z = X[neighbors,:]
    x = (Z-X[a,:]) @ (X[b,:]-X[a,:]).T/(4*t**2)
    
    r = np.linalg.norm((Z-X[a,:]),axis=1)**2

    y = np.sqrt(np.maximum(r/(2*t)**2 -x**2,np.zeros(r.shape)))
    points = np.vstack((x,y)).T
    return 2*t*directed_hausdorff(np.vstack((lin[9],np.zeros(9))).T,points)[0]

def conv_defect(X, tree, tmax = np.Inf, tmin = 0, max_val = 0, n_neigh = 0, array_type = True):
    #Step 1: get the edges of lenth smaller than 2*tmax and order them by increasing length
    edges  = []
    neighbors = []
    length = []
    n = X.shape[0]
    if n_neigh == 0:
        n_neigh = n
    
    choose_small_scale = False
    if tmax == np.Inf:
        choose_small_scale = True
    for i in range(n):
        nns = tree.query(X[i], n_neigh)
        dist = [0.5*np.linalg.norm(X[i]-X[b]) for b in nns]
        to_add = [(i,j) for j in nns if j>i]
        dist_to_add = [dist[k] for k in range(len(nns)) if nns[k]>i]
        edges += to_add
        length += dist_to_add 
        #neighbors += [nns for j in to_add]
        if choose_small_scale:
            tmax = min(tmax, max(dist))

    permut = np.argsort(length)
    length = [length[x] for x in permut]
    #neighbors = [neighbors[x] for x in permut]
    L = len(edges)

    #Step 2: for each edge, compute the hausdorff distance between the edge and the point cloud
    radii = []
    conv = []

    for i in range(L):
        if length[i] >= tmin and length[i] <= tmax:
            (a,b) = edges[permut[i]]
            t = length[i]
            radii.append(t)

            midpoint = 0.5*(X[a]+X[b])
            neighbors = get_ball_radius(midpoint, X, tree, t, n_neigh)


            if len(neighbors) <= 2 or t==0:
                conv.append(t)
                max_val = t
            else:
                max_val = max(hauss_line(X,a,b,neighbors,t), max_val)
                conv.append(max_val)
    
    if array_type:
        return np.array(radii), np.array(conv)
    else:
        return radii, conv

def get_ball_radius(point, X, tree, t, n_neigh):
    nns = tree.query(point, n_neigh)
    return [b for b in nns if np.linalg.norm(point-X[b])< 1.001*t]
          
def t_opt(lambd,t,conv):
    b = (conv >lambd*t)
    for i in range(len(b)):
        if not(b[i]) and t[i]>0:
            break
    return t[i]


def select_lambda(list_t, conv_X, display):

    size_jump = 1/t_opt(0.5,list_t,conv_X)

    K = 200
    lambdas = np.linspace(0,1,K).reshape((K,1))
    y = np.array([1/t_opt(l,list_t,conv_X) for l in lambdas])
   
    ind_max = np.argmax(np.diff(y)>size_jump)
    lambd_max = lambdas[ind_max,0]
    lambd = lambd_max*0.8
    
    if display:
        plt.plot(lambdas,y,  '--',color='tab:red',  linewidth=2, label='$1/t_\lambda(X)$')
        
        plt.axvline(x=lambd_max)
        plt.ylim(0, 4/t_opt(0.5,list_t,conv_X))
        plt.xlim(0, 1)
        plt.yticks(fontsize=22, alpha=.7)
        plt.xticks(fontsize=22, alpha=.7)
        plt.grid(axis='both', alpha=.3)
        
        # Remove borders
        plt.gca().spines["top"].set_alpha(0.0)    
        plt.gca().spines["bottom"].set_alpha(0.3)
        plt.gca().spines["right"].set_alpha(0.0)    
        plt.gca().spines["left"].set_alpha(0.3)   
        plt.xlabel("$\lambda$",fontsize=32)
        
        
        plt.legend(loc='upper left', fontsize=20)
        plt.show()    
        
        print('Lambda max: ' + str(lambdas[ind_max,0]))
        print('Selected value of lambda: ' + str(lambd))
    return t_opt(lambd,list_t,conv_X), lambd


    
def select_t(X, display = False, tlim = np.Inf):

    
    tree = RPForest(leaf_size=500, no_trees=100)
    try:
        tree.fit(X)
    except:
        X = X.copy(order='C')
        tree.fit(X)
        
    list_t, conv_X = [], []
    n_neigh = 4
    max_val = 0
    tmax = 0
    while tmax <= tlim:
        
        radii, conv = conv_defect(X, tree, tmax = np.Inf, tmin = tmax, max_val = max_val, n_neigh = n_neigh, array_type = False)
        tmax = radii[-1]
        max_val = conv[-1]
        list_t = list_t + radii
        conv_X = conv_X + conv
        
        #print(n_neigh)
        #print(tmax)
        #plt.plot(list_t,conv_X)
        #plt.show()
        tsel, lambd = select_lambda(np.array(list_t),np.array(conv_X), display)
        if not(np.isclose(tsel,tmax, rtol=0.001)):
            break
    
        n_neigh *= 2
        
    return np.array(list_t), np.array(conv_X), tsel, lambd, n_neigh, tmax