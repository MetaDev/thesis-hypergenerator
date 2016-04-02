import model.mapping as mp
import model.fitness as fn
import util.data_format as dtfr
from util.data_format import FitnessInstanceDim,FitnessFuncDim
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



#in the case that the number of requested siblings is smaller than
class SiblingData(Enum):
    combination=1
    window=2
    first=3

#TODO data generation for evaluation, no poisson sampling, variable amount of children, return list of siblings


def training_data_generation(n_data,parent_def,
                                parent_node,parent_var_names,parental_fitness,
                                child_name,sibling_fitness,sibling_var_names,n_children,
                                sibling_data=SiblingData.combination,
                                fitness_dim=(FitnessInstanceDim.single,FitnessFuncDim.single)):
    #generate sample data

    sample_fraction=1
    min_children,max_children=parent_def.children_range(child_name)
    n_sample_children=n_children
    if n_children>max_children:
        raise ValueError("Number of requested siblings in the data is larger than the possible number of children in the model.")


    if min_children>2 and n_children>=2 and n_children<min_children:
        #generate for smallest possible amount of children
        n_sample_children=min_children
        if sibling_data is SiblingData.combination:
            sample_fraction=comb(min_children,n_children)
        elif sibling_data is SiblingData.window:
            sample_fraction=min_children+1-n_children

    parent_node.freeze_n_children(child_name,n_sample_children)
    #the sample fraction is necessary because the number to generate samples is used to generate in poisson disk sampling
    #to generate diversified samples
    parent_samples=parent_node.sample(math.ceil(n_data/sample_fraction),expressive=True)
    #generate data with n_siblings
    data=[]
    fitness=[]
    for parent in parent_samples:
        siblings=parent.children[child_name]
        #if the requested amount of children is lower than the lowest amount of children defined in the parent
        #you have 3 options: either respect the order of the children and train on a windowed view of the children
        #or only the first n
        #or don't respect the order (the order is not real anyway it is only implied by the trained variable)
        if min_children>2 and n_children>=2 and n_children<min_children:
            if sibling_data is SiblingData.combination:
                sibling_collection=combinations(siblings,n_children)
            elif sibling_data is SiblingData.window:
                sibling_collection=ut.window(siblings,n_children)
            else:
                sibling_collection=[siblings[:n_children]]
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
        for fitn in parental_fitness:
            temp_fitness_parent_child=fitn.calc(parent,child)
            capped =  temp_fitness_parent_child is None
            if not capped:
                fitness_value_parent_child[child].append(temp_fitness_parent_child)
            else:
                break
    if capped:
        return None
    #when only a single child no sibling fitness aplies
    fitness_func_value_sibling=None
    if len(siblings)>1:
        #calculate fitness between siblings, respecting the order of the fitness funcs
        for fitn in sibling_fitness:
            #add an index to children in siblings
            sibling_pairs=combinations(zip(siblings,range(len(siblings))),2)

            fitness_func_value_sibling=dict([(child,[])for child in siblings[1:]])
            for indexed_child0,indexed_child1 in sibling_pairs:
                #child0 is the child with the largest index and is the child for which the fitness value will be saved
                child0=indexed_child0[0] if indexed_child0[1]>indexed_child1[1] else indexed_child1[0]
                child1=indexed_child0[0] if indexed_child0[1]<indexed_child1[1] else indexed_child1[0]
                temp_sibling_fitness= fitn.calc(child0,child1)
                capped=temp_sibling_fitness is None
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

    #format the fitness according convention
    parental_fitness_values,sibling_fitness_values=dtfr.format_fitness_values_training(fitness_value_parent_child,
                                                                                       fitness_value_sibling,
                                                                                      siblings)
    #combine fitness functions
    return [parental_fitness_values,sibling_fitness_values]






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
