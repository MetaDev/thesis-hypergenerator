# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
#polygon imports

import model.search_space as sp
import model.mapping as mp
import util.visualisation as vis
import util.utility as ut
import model.fitness as fn
import learning.learning_utility as lut

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as ss

from model.search_space import VectorVariableUtility
from sklearn import mixture

from util import setting_values
import matplotlib.cm as cm
import model.test_models as tm
from operator import add
from model.search_space import GMMVariable,DummyStochasticVariable
from gmr import GMM


from itertools import combinations

from itertools import product
#train parent-child

ndata=200

data=[]
fitness_values=[]
polygons_vis=[]

#BUG

#use root
def feauture_fitness_extraction(root,fitness_funcs,X_var_names,Y_var_names):
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
        fitness_sum=sum([f(child,root) for f in  fitness_funcs])
        fitness.append(fitness_sum)
    return data,fitness
def fitness_extraction_dist(samples):
    return fn.dist_between_parent_child(ut.extract_samples_vars(samples,sample_name="parent")[0],ut.extract_samples_vars(samples,sample_name="child"))
def fitness_polygon_overl(sample1,sample2):
    polygons=mp.map_layoutsamples_to_geometricobjects([sample1,sample2],shape_name="shape")
    return fn.pairwise_overlap(polygons[0],polygons[1],normalized=True)
def fitness_polygon_alignment(sample1,sample2):
    polygons=mp.map_layoutsamples_to_geometricobjects([sample1,sample2],shape_name="shape")
    return fn.pairwise_closest_line_alignment(polygons[0],polygons[1])

model_root=tm.test_samples_var_child_pos_size_rot_parent_shape()
X_var_names=["position"]
Y_var_names=["shape3"]
fitness_funcs=[fitness_polygon_overl,fitness_polygon_alignment]
for i in range(ndata):
    #train per child
    sample_features,sample_fitness=feauture_fitness_extraction(model_root.sample(),
                                                               fitness_funcs,
                                                               X_var_names,Y_var_names)
    data.extend(sample_features)
    fitness_values.extend(sample_fitness)

#give heavy penalty for intersection
#inverse and normalise fitness
#higher fitness means less overlap
fitness_values=ut.normalise_array((fn.invert_fitness(fitness_values))**4)

#better statistics of fittnes
vis.print_fitness_statistics(fitness_values)

x,y,rotation,sizex,sizey,x_shape,y_shape=zip(*data)

n_components=10
n_samples=100

plt.scatter(x,y,c=fitness_values,cmap=cm.Blues)
plt.colorbar()


#both full and tied covariance work
#but tied prevents overfitting but is worse to find conditional estimatation of sharp corners
gmm = lut.GMM_weighted(n_components=n_components, covariance_type='full',min_covar=0.01,random_state=setting_values.random_state)

gmm.weighted_fit(data,fitness_values)
#convert grm GMM
gmm = GMM(n_components=len(gmm.weights_), priors=gmm.weights_, means=gmm.means_, covariances=gmm._get_covars(), random_state=setting_values.random_state)
position_samples=gmm.sample(n_samples=n_samples)


#construct new model
child = model_root.get_child_node("child")

X_dummy_vars=[DummyStochasticVariable(child.get_variable("position"))]
Y_vars=[model_root.get_variable(name) for name in Y_var_names]
gmm_var=GMMVariable("test",gmm,X_dummy_vars,Y_vars)

model_root.get_child_node("child").set_learned_variable(gmm_var)


#TODO
#train model P(C0|C1,P)

#repeat sampling
data=[]
fitness_values=[]

#the variable of the child has been replaced by trained model
#use root
def feauture_fitness_extraction_child(root,fitness_funcs,X_var_names,Y_var_names):
    #only get children of a certain name
    fitness=[]
    data=[]
    for pair in combinations(root.get_children("child")):
        child0=pair[0]
        child1=pair[1]
        parent_Y_vars=[root.independent_vars[name] for name in Y_var_names]
        child0_Y_vars=[child.independent_vars[name] for name in X_var_names]
        child1_X_vars=[child.independent_vars[name] for name in X_var_names]
        data.append(np.hstack((child1_X_vars,child0_Y_vars,parent_Y_vars)).flatten())
        fitness_sum=sum([f(child1,root) for f in  fitness_funcs]) + sum([f(child1,child0) for f in  fitness_funcs])
        fitness.append(fitness_sum)
    return data,fitness
for i in range(ndata):
    #train per child
    sample_features,sample_fitness=feauture_fitness_extraction(model_root.sample(),
                                                               fitness_funcs,
                                                               X_var_names,Y_var_names)
    data.extend(sample_features)
    fitness_values.extend(sample_fitness)

x,y,x_shape,y_shape=zip(*data)

#train child1->child2
#better statistics of fittnes
