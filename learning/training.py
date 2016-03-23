import model.mapping as mp
import model.fitness as fn

import numpy as np
from itertools import combinations


#todo each fitness should return the name of the variables it depends upon
#this is the minimal set for learning, more variables can be added
#also each variable should be checked wether it's stochastic
#when re-learning the order of the siblings will be set and should be respected->don't use combinations
#the use of combinations instead of more data still needs to be validated
#sibling order 1 is child-parent relation only
#the fitness funcs should include their order

#TODO add child name as argument
def parent_child_variable_training(n_samples,root,parental_fitness_funcs,sibling_fitness_funcs,
                                vars_parent,vars_children, sibling_order=0):
    data=[]
    fitness={}
    fitness["parent"]=[]
    fitness["children"]=[]
    root_samples=root.sample(n_samples)
    print(len(root_samples))
    #only get children of a certain name
    for i in range(n_samples):
        root_sample=root_samples[i]
        parent_Y_vars=[root_sample.values["ind"][name] for name in vars_parent]
        for children in combinations(root_sample.children["child"],sibling_order+1):
            child_vars=[]
            parent_fitness=1
            #calculate fitness of next child to parent
            for child in children:
                child_vars.extend([child.values["ind"][name] for name in vars_children])
                parent_fitness*=sum([func(child,root_sample)**order for func,order in  parental_fitness_funcs.items()])
            fitness["parent"].append(parent_fitness)
            #the independent variable is
            child_vars=np.array(child_vars).flatten()
            parent_Y_vars=np.array(parent_Y_vars).flatten()
            data.append(np.hstack((child_vars,parent_Y_vars)))
            child_fitness=1
            if sibling_order>0:
                #calculate pairwise fitness between children
                for child0,child1 in combinations(children,2):
                    child_fitness*=sum([func(child0,child1)**order for func,order in  sibling_fitness_funcs.items()])
                fitness["children"].append(child_fitness)

    return data,fitness



def fitness_polygon_overl(sample0,sample1):
    polygons=mp.map_layoutsamples_to_geometricobjects([sample0,sample1],shape_name="shape")
    return fn.pairwise_overlap(polygons[0],polygons[1])

def fitness_polygon_alignment(sample0,sample1):
    polygons=mp.map_layoutsamples_to_geometricobjects([sample0,sample1],shape_name="shape")
    return fn.pairwise_closest_line_alignment(polygons[0],polygons[1],threshold=30)
def fitness_min_dist(sample0,sample1):
    pos0=sample0.values["ind"]["position"]
    pos1=sample1.values["ind"]["position"]
    return fn.pairwise_min_dist(pos0,pos1,threshold=1)
