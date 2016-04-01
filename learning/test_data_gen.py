import numpy as np

import learning.evaluation as ev
from util import setting_values
import model.test_models as tm
from model import fitness as fn

import util.data_format as dtfr

from learning.gmm import GMM

from operator import itemgetter

import learning.data_generation as dg

#this sequence indicates the order of the markov chain between siblings [1,2]-> second child depends on the first
#the third on the first and second
#the first child is always independent
sibling_order_sequence=[0,1,2,3,4,4,4,4]
sibling_data=dg.SiblingData.combination
fitness_dim=(dg.FitnessInstanceDim.parent_sibling,dg.FitnessFuncDim.seperate)
#sibling_order_sequence=[0,1,2,3,4]
#the sibling order defines the size of the joint distribution that will be trained
sibling_order=np.max(sibling_order_sequence)
n_siblings=sibling_order+1
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
parental_fitness=[fn.Fitness(fn.fitness_min_dist,4,0,1)]
#only the func order and cap is used for training
sibling_fitness=[fn.Fitness(fn.fitness_polygon_overl,32,0,1)]
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
                                parent_node,parent_var_names,parental_fitness,
                                child_name,sibling_fitness,sibling_var_names,n_siblings=n_siblings,
                                sibling_data=sibling_data,
                                fitness_dim=fitness_dim)
gmm = GMM(n_components=15,random_state=setting_values.random_state)

#check if regression works better this way
train_data=np.column_stack((data,fitness))
#for regression calculate full joint
gmm.fit(train_data,infinite=False,min_covar=0.01)
indices,targets = dtfr.format_fitness_for_regression(parental_fitness,sibling_fitness,
                                                     n_siblings,len(data[0]),fitness_dim)
print(indices,targets)
#condition on fitness
gmm.condition(indices,targets)