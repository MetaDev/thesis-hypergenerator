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
sibling_order=2
ndata=100
n_components=15
regression=True
infinite=False


#training variables and fitness functions
vars_children=["position"]
vars_parent=["shape3"]

#fitness func, order cap and regression target
sibling_fitness_funcs=[(tr.fitness_min_dist,2,0.5,1)]
#only the func order and cap is used for training
sibling_fitness_funcs_sampling=[f[0:3] for f in sibling_fitness_funcs]

parent_child_fitness_funcs=[(tr.fitness_polygon_overl,32,0.5,1)]
parent_child_fitness_funcs_sampling=[f[0:3] for f in parent_child_fitness_funcs]

#model to train on
root=tm.test_model_var_child_position_parent_shape()


#a model trained on 4 children can also be used for 5 children of the fifth no longer conditions on the first
#These models can be reused because there's no difference in the markov chain of order n between the n+1 and n+2 state
repeat_trained_model=True


#train parent-child

polygons_vis=[]


#get size of vars
X_var_length=np.sum([root.children["child"][1].get_variable(name).size for name in vars_children])
Y_var_length=np.sum([root.get_variable(name).size for name in vars_parent])

#fitness order parent_funcs,sibling_funcs each func in a seperate column
#the order of the data is child,parent,sibling0,sibling1,..
data,fitness=tr.parent_child_variable_training(ndata,root,parent_child_fitness_funcs_sampling,
sibling_fitness_funcs_sampling,vars_parent,vars_children,child_name="child",sibling_order=sibling_order)

#print the first fitness func
print("parent fitness")
vis.print_fitness_statistics(fitness[:,0])
print("child fitness")
vis.print_fitness_statistics(fitness[:,1])

gmm = GMM(n_components=n_components,random_state=setting_values.random_state)

if not regression:
    fitness_all = np.array(fitness["children"])*np.array(fitness["parent"])
    gmm.fit(data,fitness_all,infinite=infinite,min_covar=0.01)
else:
    train_data=np.column_stack((data,fitness))
    #for regression calculate full joint
    gmm.fit(train_data,infinite=infinite,min_covar=0.01)
    #fitness indices and values according to convention
    fitness_indices=np.arange(len(data[0]),len(data[0])+len(fitness[0]))
    all_fitness=parent_child_fitness_funcs+sibling_fitness_funcs
    fitness_values=[f[3] for f in all_fitness]
    print(fitness_indices,fitness_values)
    #condition on fitness
    gmm=gmm.condition(fitness_indices,fitness_values)

#if sibling order is 0 you have only a single gmm
gmms=[None]*(sibling_order+1)
gmms[sibling_order]=gmm
#marginalise for each child
#P(c_n,c_n-1,..,c0,p)->P(c_i,c_i+1,..,c_0,p)
for i in range(sibling_order):
    indices=np.arange((sibling_order-i)*X_var_length,(sibling_order+1)*X_var_length+Y_var_length)
    gmms[i]=gmm.marginalise(indices)

##construct new model
#from model.search_space import GMMVariable,DummyStochasticVariable
#first_child=root.children["child"][0]
#
#X_dummy_vars=[DummyStochasticVariable(first_child.get_variable("position"))]
#Y_sibling_vars=[first_child.get_variable(name) for name in vars_children]
#Y_vars=[root.get_variable(name) for name in vars_parent]
##the sibling order of the gmm is maximum the sibling order of the trained gmm
#gmm_vars=[GMMVariable("test",gmm,X_dummy_vars,Y_vars,Y_sibling_vars,sibling_order=i) for gmm,i in zip(gmms,range(len(gmms)))]
#
#children = root.children["child"]
##the gmms are ordered the same as the children
#
##assign variable child i with gmm min(i,sibling_order)
#if repeat_trained_model:
#    learned_var_range=len(children)
#else:
#    learned_var_range=len(gmms)
#for i in range(learned_var_range):
#    children[i].set_learned_variable(gmm_vars[min(i,sibling_order)])
#
##re-estimate average fitness
##repeat sampling
#data,fitness=tr.parent_child_variable_training(ndata,root,parent_child_fitness_funcs,
#sibling_fitness_funcs,vars_parent,vars_children,sibling_order)
#
#
##cap the data
#
#
#vis.print_fitness_statistics(fitness["parent"])
#if sibling_order>0:
#    vis.print_fitness_statistics(fitness["children"])