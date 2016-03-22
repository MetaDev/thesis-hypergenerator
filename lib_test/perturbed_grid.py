# -*- coding: utf-8 -*-
import numpy as np
def find_point_set(lower,upper, n_points):
    upper=np.array(upper)
    lower=np.array(lower)
    n_dim_points= n_points ** (1/len(upper))
    n_dim_points=int(round(n_dim_points))
    cell_sizes=(upper-lower)/n_dim_points
    #move lower bound half a cell_size up and upper half a cell_down
    upper=upper-cell_sizes/2
    lower=lower+cell_sizes/2
    #when using perturbed grid sampling the number of points should be rootable by the num of dimensions
    if not n_dim_points** len(upper) == n_points:
        raise ValueError("The number of point is not a perfect root for the number of dimensions")
    mesh = np.array(np.meshgrid(*[np.linspace(i,j,n_dim_points) for i,j in zip(upper,lower)]))
    #perturb each point in mesh with max half a cell_size

    #make it an n_dim list
    mesh=mesh.T
    points=mesh.reshape(n_points,-1)
    return points
