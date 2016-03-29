# -*- coding: utf-8 -*-



# -*- coding: utf-8 -*-
#polygon imports
from importlib import reload

import model.search_space as sp
import model.mapping as mp
import util.visualisation as vis
import util.utility as ut
import model.fitness as fn

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as ss


from util import setting_values
import matplotlib.cm as cm
import model.test_models as tm
from operator import add
from learning.gmm import GMM

import learning.training as tr

#experiment hyperparameters:
#this sequence indicates the order of the markov chain between siblings [1,2]-> second child depends on the first
#the third on the first and second
#the first child is always independent
sibling_order_sequence=[0,1,2,3,4,4,4,4]
#sibling_order_sequence=[0,1,2,3,4]
#the sibling order defines the size of the joint distribution that will be trained
sibling_order=np.max(sibling_order_sequence)
child_name="child"
ndata=100
n_components=15
regression=False
infinite=False
sibling_train_order=tr.SiblingTrainReOrder.no_reorder
order_variable="position"
respect_sibling_order=False

#training variables and fitness functions
#this expliicitly also defines the format of the data
vars_children=["position"]
vars_parent=["shape3"]

#this expliicitly also defines the format of the data
#fitness func, order cap and regression target
sibling_fitness_funcs=[(tr.fitness_min_dist,4,0,1)]
#only the func order and cap is used for training
sibling_fitness_funcs_sampling=[f[0:3] for f in sibling_fitness_funcs]

parent_child_fitness_funcs=[(tr.fitness_polygon_overl,32,0,1)]
parent_child_fitness_funcs_sampling=[f[0:3] for f in parent_child_fitness_funcs]
#model to train on
root_node,root_def=tm.test_model_var_child_position_parent_shape()

#check sibling sequence
wrong_sequence = any(sibling_order_sequence[i]>i for i in range(len(sibling_order_sequence)))
if wrong_sequence:
    raise ValueError("Some orders of the sibling order sequence exceed the number of previous siblings.")
max_children=root_def.max_children(child_name)
if len(sibling_order_sequence) != max_children:
    raise ValueError("The number of siblings implied by the sibling order sequence can not be different than the maximum number of children in the model.")


#train parent-child

#get size of vars
sibling_vars=[root_def.children[child_name].variables[name] for name in vars_children]
parent_vars=[root_def.variables[name] for name in vars_parent]

#the order of the data is sibling0,sibling1,..,parent
data,fitness=tr.data_generation(ndata,root_node,parent_child_fitness_funcs_sampling,
                                               sibling_fitness_funcs_sampling,vars_parent,
                                               vars_children,child_name=child_name,
                                               sibling_order=sibling_order,
                                               sibling_train_order=sibling_train_order,
                                               order_variable=order_variable,
                                               respect_sibling_order=respect_sibling_order)

#print the first fitness func
print("parent fitness")
vis.print_fitness_statistics(fitness[:,0])
print("child fitness")
vis.print_fitness_statistics(fitness[:,1])

gmm = GMM(n_components=n_components,random_state=setting_values.random_state)

if not regression:
    fitness_all = np.array(fitness[:,1])*np.array(fitness[:,0])
    gmm.fit(data,fitness_all,infinite=infinite,min_covar=0.01)
else:
    train_data=np.column_stack((data,fitness))
    #for regression calculate full joint
    gmm.fit(train_data,infinite=infinite,min_covar=0.01)
    #fitness indices and values according to convention
    fitness_indices=np.arange(len(data[0]),len(data[0])+len(fitness[0]))
    all_fitness=parent_child_fitness_funcs+sibling_fitness_funcs
    fitness_values=[f[3] for f in all_fitness]
    #condition on fitness
    gmm=gmm.condition(fitness_indices,fitness_values)

import util.data_format as dtfr
gmms=dtfr.marginalise_gmm(gmm,parent_vars,sibling_vars,sibling_order)

#edit model with found variables
from model.search_space import GMMVariable
child_def=root_def.children["child"]

child_vars=[child_def.variables["position"]]
sibling_vars=[child_def.variables[name] for name in vars_children]
parent_vars=[root_def.variables[name] for name in vars_parent]


#the sibling order of the gmm is maximum the sibling order of the trained gmm
gmm_vars=[GMMVariable("test",gmm,child_vars,parent_vars,sibling_vars,sibling_order=i)
for gmm,i in zip(gmms,range(len(gmms)))]

children = root_node.children["child"]
#the gmms are ordered the same as the children

#use sibling order sequence to assign gmms[i] to the a child with order i
#assign variable child i with gmm min(i,sibling_order)
for i in range(max_children):
    children[i].set_learned_variable(gmm_vars[sibling_order_sequence[i]])

respect_sibling_order=False
#re-estimate average fitness
#repeat sampling
data,fitness=tr.data_generation(ndata,root_node,
                                               parent_child_fitness_funcs_sampling,
                                               sibling_fitness_funcs_sampling,
                                               vars_parent,vars_children,
                                               child_name=child_name,sibling_order=sibling_order,
                                               respect_sibling_order=respect_sibling_order)

print("parent fitness")
vis.print_fitness_statistics(fitness[:,0])
print("child fitness")
vis.print_fitness_statistics(fitness[:,1])

#retrain

gmm = GMM(n_components=n_components,random_state=setting_values.random_state)

if not regression:
    fitness_all = np.array(fitness[:,1])*np.array(fitness[:,0])
    gmm.fit(data,fitness_all,infinite=infinite,min_covar=0.01)
else:
    train_data=np.column_stack((data,fitness))
    #for regression calculate full joint
    gmm.fit(train_data,infinite=infinite,min_covar=0.01)
    #fitness indices and values according to convention
    fitness_indices=np.arange(len(data[0]),len(data[0])+len(fitness[0]))
    all_fitness=parent_child_fitness_funcs+sibling_fitness_funcs
    fitness_values=[f[3] for f in all_fitness]
    #condition on fitness
    gmm=gmm.condition(fitness_indices,fitness_values)

gmms=dtfr.marginalise_gmm(gmm,parent_vars,sibling_vars,sibling_order)

#edit model with found variables


#the sibling order of the gmm is maximum the sibling order of the trained gmm
gmm_vars=[GMMVariable("test",gmm,child_vars,parent_vars,sibling_vars,sibling_order=i)
for gmm,i in zip(gmms,range(len(gmms)))]

#the gmms are ordered the same as the children

#use sibling order sequence to assign gmms[i] to the a child with order i
#assign variable child i with gmm min(i,sibling_order)
for i in range(max_children):
    children[i].set_learned_variable(gmm_vars[sibling_order_sequence[i]])

respect_sibling_order=False
#re-estimate average fitness
#repeat sampling
data,fitness=tr.data_generation(ndata,root_node,
                                               parent_child_fitness_funcs_sampling,
                                               sibling_fitness_funcs_sampling,
                                               vars_parent,vars_children,
                                               child_name=child_name,sibling_order=sibling_order,
                                               respect_sibling_order=respect_sibling_order)

print("parent fitness")
vis.print_fitness_statistics(fitness[:,0])
print("child fitness")
vis.print_fitness_statistics(fitness[:,1])

