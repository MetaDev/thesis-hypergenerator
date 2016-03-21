# -*- coding: utf-8 -*-
import numpy as np
def find_point_set(upper,lower, n_points):

    n_dim_points= n_points ** (1/len(mins))
    n_dim_points=int(round(n_dim_points))
    cell_sizes=(upper-lower)/n_dim_points
    #when using perturbed grid sampling the number of points should be rootable by the num of dimensions
    if not n_dim_points** len(mins) == n_points:
        raise ValueError("The number of point is not a perfect root for the number of dimensions")
    mesh = np.array(np.meshgrid(*[np.linspace(i,j,n_dim_points) for i,j in zip(mins,maxs)]))
    #choose random point in each cell of mesh

    mesh.flatten()
    return mesh
