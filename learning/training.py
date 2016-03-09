import model.mapping as mp
import model.fitness as fn

import numpy as np

def feauture_fitness_extraction(root,fitness_funcs,X_var_names,Y_var_names,fitness_order=8):
    #only get children of a certain name
    fitness=[]
    data=[]
    for child in root.get_children("child"):
        child_X_vars=[]
        for name in X_var_names:
            child_X_vars.extend(child.independent_vars[name])
        parent_Y_vars=[]
        for name in Y_var_names:
            parent_Y_vars.extend(root.independent_vars[name])
        data.append(np.hstack((child_X_vars,parent_Y_vars)).flatten())

        fitness_s=np.prod([f(child,root)**fitness_order for f in  fitness_funcs])
        fitness.append(fitness_s)
    return data,fitness

def fitness_polygon_overl(sample0,sample1):
    polygons=mp.map_layoutsamples_to_geometricobjects([sample0,sample1],shape_name="shape")
    return fn.pairwise_overlap(polygons[0],polygons[1])

def fitness_polygon_alignment(sample0,sample1):
    polygons=mp.map_layoutsamples_to_geometricobjects([sample0,sample1],shape_name="shape")
    return fn.pairwise_closest_line_alignment(polygons[0],polygons[1],threshold=30)
def fitness_min_dist(sample0,sample1):
    pos0=sample0.independent_vars["position"]
    pos1=sample1.independent_vars["position"]
    return fn.pairwise_min_dist(pos0,pos1,threshold=2)
