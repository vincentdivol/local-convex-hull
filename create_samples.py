"""
Generate random samples around different shapes
"""

import numpy as np
import matplotlib.pyplot as plt
import bezier

def sample_circle(n_pts):
    """ 
    Returns an array of shape (n_pts,2) of points uniformly sampled on the circle.
    """
    u =  2*np.pi*np.random.rand(n_pts)
    return np.transpose(np.concatenate((np.cos(u),np.sin(u))).reshape(2,n_pts))

def sample_torus(n_pts, r, R):
    """
    Return an array of shape (n_pts,3) of points uniformly sampled on the torus with transverse cirlce of radius r, 
    and distance from the center of the torus to any center of the transverse circle equal to R.
    """
    u = 2*np.pi*np.random.rand(n_pts)
    v = 2*np.pi*np.random.rand(n_pts)
    return np.transpose(np.vstack([(R+r*np.cos(v))*np.cos(u),(R+r*np.cos(v))*np.sin(u),r*np.sin(v)]))

def sample_sphere(n_pts,dim):
    """
    Returns a n-sample on the unit sphere of dimension dim
    """
    a = (2*np.random.rand((dim+1)*n_pts)-1).reshape(n_pts,dim+1)
    norm = np.sqrt(np.sum(a[:,i]**2 for i in range(dim+1)))
    return a/np.vstack(norm for i in range(dim+1)).T

def sample_bean(n_pts) :
    u =  np.pi*np.random.rand(n_pts)
    r = np.sin(u)**3+np.cos(u)**3
    return np.transpose(np.concatenate((r*np.cos(u),r*np.sin(u))).reshape(2,n_pts))

def display(list_of_points):
    plt.figure()
    ax = plt.axes()
    # Setting the background color
    ax.set_facecolor("white")
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
    plt.gca().set_aspect('equal', adjustable = 'box')

    plt.scatter([a[0] for a in list_of_points],[a[1] for a in list_of_points])
    plt.show()
    
def sample_s_shape(n):
    deg =2
    a=np.loadtxt('s_points')
    nodes =np.zeros((2,len(a)+deg+1))
    for i in range(len(a)):
        nodes[:,i]=a[i]
    nodes[:,len(a):]=nodes[:,:(deg+1)]
    nodes1= np.asfortranarray(nodes)
    curves = []
    
    for i in range(len(a)):
        curve = bezier.Curve(nodes1[:,i:i+deg+1], degree=deg)
        curves.append(curve)
#    ax = curves[0].plot(num_pts=10)
#    for i in range(len(a)):
#        _=curves[i].plot(10, ax=ax)
    data = np.zeros((n,2))
    for i in range(n):
        j = np.random.randint(0,len(curves)-1)
        data[i,:]=curves[j].evaluate(np.random.rand(1)[0]).T
    return data

        