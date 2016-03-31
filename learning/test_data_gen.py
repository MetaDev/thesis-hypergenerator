import numpy as np

import learning.evaluation as ev
from util import setting_values
import model.test_models as tm

from learning.gmm import GMM

from operator import itemgetter

import learning.data_generation as dg

#this sequence indicates the order of the markov chain between siblings [1,2]-> second child depends on the first
#the third on the first and second
#the first child is always independent
sibling_order_sequence=[0,1,2,3,4,4,4,4]
sibling_data=dg.SiblingData.combination
fitness_dim=(dg.FitnessInstanceDim.single,dg.FitnessFuncDim.single)
#sibling_order_sequence=[0,1,2,3,4]
#the sibling order defines the size of the joint distribution that will be trained
sibling_order=np.max(sibling_order_sequence)
max_n_children=sibling_order+1
#gmm marginalisation [order_1,order_2,..,order_sibling_order]
#0->train full joint
#-1->derive (marginalise) from closest higher order
#n->derive from order n, ! n< order, n<n_children-1

#sibling_order_marginilasation=[]
#
#if len(sibling_order_marginilasation) != sibling_order
child_name="child"

#training variables and fitness functions
#this expliicitly also defines the format of the data
sibling_var_names=["position","rotation"]
parent_var_names=["shape3"]

#this expliicitly also defines the format of the data
#fitness func, order cap and regression target
sibling_fitness_funcs=[(dg.fitness_min_dist,4,0,1),(dg.fitness_polygon_overl,32,0,1)]
#only the func order and cap is used for training
sibling_fitness_funcs_sampling=[f[0:3] for f in sibling_fitness_funcs]

parent_child_fitness_funcs=[(dg.fitness_polygon_overl,32,0,1)]
parent_child_fitness_funcs_sampling=[f[0:3] for f in parent_child_fitness_funcs]
#model to train on
parent_node,parent_def=tm.test_model_var_child_position_parent_shape()
child_nodes=parent_node.children[child_name]
n_data=100

#check sibling sequence
wrong_sequence = any(sibling_order_sequence[i]>i for i in range(len(sibling_order_sequence)))
if wrong_sequence:
    raise ValueError("Some orders of the sibling order sequence exceed the number of previous siblings.")
max_children=parent_def.children_range(child_name)[1]
if len(sibling_order_sequence) != max_children:
    raise ValueError("The number of siblings implied by the sibling order sequence can not be different than the maximum number of children in the model.")
#do n_iter number of retrainings using previously best model
#find out the performance of the current model
data,fitness=dg.data_generation(n_data,parent_def,
                                parent_node,parent_var_names,parent_child_fitness_funcs_sampling,
                                child_name,sibling_fitness_funcs_sampling,sibling_var_names,n_siblings=6,
                                sibling_data=sibling_data,
                                fitness_dim=fitness_dim)
print(len(fitness[0]))
print((data[0]))