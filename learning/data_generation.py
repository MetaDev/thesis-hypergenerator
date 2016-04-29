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


def training_data_generation(n_data,parent_def,
                                parent_node,parent_var_names,
                                child_name,sibling_var_names,n_children,
                                fitness_funcs,
                                sibling_data=SiblingData.combination):
    #generate sample data

    sample_fraction=1
    min_children,max_children=parent_def.variable_range(child_name)
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
                fitness_value=fn.fitness_calc(parent,
                                       sibling_sub_collection,fitness_funcs)
                fitness.append(fitness_value)
                data.append(dtfr.format_data_for_training(parent,parent_var_names,
                                                              sibling_sub_collection,sibling_var_names))
        else:
            fitness_value=fn.fitness_calc(parent,
                                       siblings,fitness_funcs)
            fitness.append(fitness_value)
            data.append(dtfr.format_data_for_training(parent,parent_var_names,siblings,sibling_var_names))

        if len(data)>=n_data:
            break
    #unfreeze model children
    parent_node.freeze_n_children(child_name,None)
    return np.array(data),np.array(fitness)

