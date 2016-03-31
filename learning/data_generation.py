import model.mapping as mp
import model.fitness as fn
import util.data_format as dtfr

import numpy as np
from itertools import combinations,permutations
import util.utility as ut
import math
from scipy.misc import comb
import warnings


#todo each fitness should return the name of the variables it depends upon
#this is the minimal set for learning, more variables can be added
#also each variable should be checked wether it's stochastic
#when re-learning the order of the siblings will be set and should be respected->don't use combinations
#the use of combinations instead of more data still needs to be validated
#sibling order 1 is child-parent relation only
#the fitness funcs should include their order
from enum import Enum
#the dimensionality of the fitness per instance in the data
#single means all fitness value are combined into a single fitness function
#parent children means that there the fitness between parent-child and siblings is combined
#parent siblings means that each sibling has a seperate fitness value, calculated relative to the previously placed siblings

class FitnessInstanceDim(Enum):
    single=1
    parent_children=2
    parent_sibling=3

#the difference between parent_children and parent_sibling dimensionality can be seen semanticalliy as
#how you look at the child placement concept, if each child is placed after the other the fitness of the next only depends on the previous
#however if you see it as if all children are placed simultanously than their fitness has to be calculated likewise
#->in relation to all children
#the dimensionality of the fitness per fitness function

class FitnessFuncDim(Enum):
    single=1
    seperate=2


#in the case that the number of requested siblings is smaller than
class SiblingData(Enum):
    combination=1
    window=2
    first=3


def data_generation(n_data,parent_def,
                                parent_node,parent_var_names,parental_fitness,
                                child_name,sibling_fitness,sibling_var_names,n_siblings,
                                sibling_data=SiblingData.combination,
                                fitness_dim=(FitnessInstanceDim.single,FitnessFuncDim.single)):
    #generate sample data

    sample_fraction=1
    min_children,max_children=parent_def.children_range(child_name)
    n_sample_children=n_siblings
    if n_siblings>max_children:
        raise ValueError("Number of requested siblings in the data is larger than the possible number of children in the model.")

    if n_siblings<min_children:
        #generate for smallest possible amount of children
        n_sample_children=min_children
        if sibling_data is SiblingData.combination:
            sample_fraction=comb(min_children,n_siblings)
        elif sibling_data is SiblingData.window:
            sample_fraction=min_children+1-n_siblings

    parent_node.freeze_n_children(child_name,n_sample_children)
    #the sample fraction is necessary because the number to generate samples is used to generate in poisson disk sampling
    #to generate diversified samples
    parent_samples=parent_node.sample(math.ceil(n_data/sample_fraction))
    #generate data with n_siblings
    data=[]
    fitness=[]
    for parent in parent_samples:
        siblings=parent.children[child_name]
        #if the requested amount of children is lower than the lowest amount of children defined in the parent
        #you have 3 options: either respect the order of the children and train on a windowed view of the children
        #or only the first n
        #or don't respect the order (the order is not real anyway it is only implied by the trained variable)
        if n_siblings<min_children:
            if sibling_data is SiblingData.combination:
                sibling_collection=combinations(siblings,n_siblings)
            elif sibling_data is SiblingData.window:
                sibling_collection=ut.window(siblings,n_siblings)
            else:
                sibling_collection=[siblings[:n_siblings]]
            for sibling_sub_collection in sibling_collection:
                fitness_value=_fitness_calc(parent,parental_fitness,
                                       sibling_sub_collection,sibling_fitness,
                                       fitness_dim)
                if fitness_value is not None:
                    fitness.append(fitness_value)
                    data.append(dtfr.format_data_for_training(parent,parent_var_names,
                                                              sibling_sub_collection,sibling_var_names))
        else:
            fitness_value=_fitness_calc(parent,parental_fitness,
                                       siblings,sibling_fitness,
                                       fitness_dim)
            if fitness_value is not None:
                fitness.append(fitness_value)
                data.append(dtfr.format_data_for_training(parent,parent_var_names,siblings,sibling_var_names))

        if len(data)>=n_data:
            break
    #unfreeze model children
    parent_node.freeze_n_children(child_name,None)
    return np.array(data),np.array(fitness)

#calculate fitness with given options
#the options will define the fitness dimensionality
def _fitness_calc(parent,parental_fitness,
                       siblings,sibling_fitness,
                       fitness_dim=(FitnessInstanceDim.single,FitnessFuncDim.single)):


    #calcualte fitness parent->child
    fitness_value_parent_child=dict([(child,[]) for child in siblings])
    fitness_value_sibling=dict([(child,[]) for child in siblings[1:]])
    for child in siblings:
        fitness_value_parent_child[child]=[]
        #the sibling fitness of the first child is always 1
        fitness_value_sibling[child]=[]
        for func,order,cap in parental_fitness:
            temp_fitness_parent_child=func(child,parent)**order
            capped =  temp_fitness_parent_child < cap
            if not temp_fitness_parent_child < cap:
                fitness_value_parent_child[child].append(temp_fitness_parent_child)
            else:
                break

    #calculate fitness between siblings, respecting the order of the fitness funcs
    for func,order,cap in sibling_fitness:
        #add an index to children in siblings
        sibling_pairs=combinations(zip(siblings,range(len(siblings))),2)

        fitness_func_value_sibling=dict([(child,[])for child in siblings[1:]])
        for indexed_child0,indexed_child1 in sibling_pairs:
            #child0 is the child with the largest index and is the child for which the fitness value will be saved
            child0=indexed_child0[0] if indexed_child0[1]>indexed_child1[1] else indexed_child1[0]
            child1=indexed_child0[0] if indexed_child0[1]<indexed_child1[1] else indexed_child1[0]
            temp_sibling_fitness=func(child0,child1)**order
            capped=temp_sibling_fitness < cap
            if not capped:
                fitness_func_value_sibling[child0].append(temp_sibling_fitness)
            else:
                break
        #a set of siblings is invalid if one of its fitness funcs is below the cap
        if capped:
            break
        #save product of fitness with previous siblings
        for child in siblings[1:]:
            fitness_value_sibling[child].append(np.prod(fitness_func_value_sibling[child]))

    if capped:
        return None

    #depending on the dimensionality options combine the fitness values

    #combine fitness functions

    if fitness_dim[0] is FitnessInstanceDim.parent_children and fitness_dim[1] is FitnessFuncDim.seperate:
        fitness_axis=0
    if fitness_dim[0] is FitnessInstanceDim.parent_sibling and fitness_dim[1] is FitnessFuncDim.single:
        fitness_axis=1
    if fitness_dim[0] is FitnessInstanceDim.single or (fitness_dim[0] is FitnessInstanceDim.parent_children
    and fitness_dim[1] is FitnessFuncDim.single):
        fitness_axis=None

    if fitness_dim[0] is FitnessInstanceDim.single and fitness_dim[1] is FitnessFuncDim.seperate:
        warnings.warn("The fitness will have dimension 1 even though seperate functions are requested, this is because the instance dimension combines sibling and parent functions.")

    #convert dict to array of fitness values in correct order
    fitness_value_parent_child=[fitness_value_parent_child[child] for child in siblings]
    fitness_value_sibling=[fitness_value_sibling[child] for child in siblings[1:]]
    if not (fitness_dim[0] is FitnessInstanceDim.parent_sibling and fitness_dim[1] is FitnessFuncDim.seperate):
        #combine fitness for parent and all siblings
        #order needs to be respected for siblings
        fitness_value_parent_child=np.prod(fitness_value_parent_child,fitness_axis)
        fitness_value_sibling=np.prod(fitness_value_sibling,fitness_axis)

    #combine both parent and child fitness
    if fitness_dim[0] is FitnessInstanceDim.single:
        return np.array(fitness_value_parent_child*fitness_value_sibling).flatten()

    return list(ut.flatten([fitness_value_parent_child,fitness_value_sibling]))





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
