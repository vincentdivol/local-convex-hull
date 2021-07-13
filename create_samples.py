"""
Generate random samples around different shapes
"""

import numpy as np

def sample_circle(n_pts):
    """ 
    Returns an array of shape (n_pts,2) of points uniformly sampled on the circle.
    """
    u =  2*np.pi*np.random.rand(n_pts)
    return np.transpose(np.concatenate((np.cos(u),np.sin(u))).reshape(2,n_pts))

def sample_torus(n_pts, r, R):
    """
    Return an array of shape (n_pts,3) of points (non-uniformly) sampled on the torus with transverse cirlce of radius r, 
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
    """
    Returns a n-sample on the ''bean curve''
    """
    u =  np.pi*np.random.rand(n_pts)
    r = np.sin(u)**3+np.cos(u)**3
    return np.transpose(np.concatenate((r*np.cos(u),r*np.sin(u))).reshape(2,n_pts))
       
