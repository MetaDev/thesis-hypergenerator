# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
#polygon imports
from importlib import reload

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

from learning import training_utility as tu

from itertools import combinations

from itertools import product
#train parent-child
ndata=200

data=[]
fitness_values=[]
polygons_vis=[]


model_root=tm.test_model_var_child_position_parent_shape()

X_var_names=["position"]
Y_var_names=["shape3"]
child_child_fitness_funcs=[tu.fitness_min_dist]
parent_child_fitness_funcs=[tu.fitness_polygon_overl]

#TODO
#train model P(C0|C1,P)
ndata=400

#repeat sampling
data=[]
fitness_values=[]

#the variable of the child has been replaced by trained model
#use root
def feauture_fitness_child_child(root,child_child_fitness_funcs,parent_child_fitness_funcs,
                                 X_var_names,Y_var_names):
    #only get children of a certain name
    fitness=[]
    data=[]
    for pair in combinations(root.get_children("child"),2):
        child0=pair[0]
        child1=pair[1]
        parent_Y_vars=[root.independent_vars[name] for name in Y_var_names]
        child0_Y_vars=[child0.independent_vars[name] for name in X_var_names]
        child1_X_vars=[child1.independent_vars[name] for name in X_var_names]
        data.append(np.hstack((child1_X_vars,child0_Y_vars)).flatten())
        fitness_sum= sum([f(child1,child0)
        for f in  child_child_fitness_funcs])
        fitness.append(fitness_sum)
    return data,fitness
for i in range(ndata):
    #train per child
    sample_features,sample_fitness=feauture_fitness_child_child(model_root.sample(),
                                                               child_child_fitness_funcs,
                                                               parent_child_fitness_funcs,
                                                               X_var_names,Y_var_names)
    data.extend(sample_features)
    fitness_values.extend(sample_fitness)

child1_x,child1_y,child0_x,child0_y_shape=zip(*data)

plt.scatter(child1_x,child1_y,c=fitness_values,cmap=cm.Blues)
plt.colorbar()
plt.show()

n_components=10
#both full and tied covariance work
#but tied prevents overfitting but is worse to find conditional estimatation of sharp corners
gmm = lut.GMM_weighted(n_components=n_components, covariance_type='full',min_covar=0.0001,random_state=setting_values.random_state)

gmm.weighted_fit(data,fitness_values)
#convert grm GMM
gmm = GMM(n_components=len(gmm.weights_), priors=gmm.weights_, means=gmm.means_, covariances=gmm._get_covars(), random_state=setting_values.random_state)
n_samples=100



#train child1->child2
#better statistics of fittnes
position=(1,2)
x= np.array(child0_x)+position[0]
y=np.array(child0_x)+position[1]
n=1;
values=np.arange(0.5,2,0.25)
print("stront")
#for p in values:
#    child_positions=np.arange(-1,1,0.5)
#    for cp in child_positions:
#        plt.scatter(cp+position[0],cp+position[1],color="blue",marker="o")
#        print(cp+position[0],cp+position[1])
#        shape=[(0, 0), (0, 1),(0.5,1),(p,p),(1, 0.5),(1,0)]
#        #the original gmm needs to have u full covariance matrix, to estimate conditional matrix
#        polygon = mp.map_to_polygon(shape,[0.5,0.5],position,0,(1,1))
#        gmm_cond=gmm.condition([2,3,4,5],[cp,cp,p,p])
#        position_samples=gmm_cond.sample(n_samples)
#        ax = plt.gca()
#        ax.set_aspect(1)
#
#        gmm_cond.n_components
#        x_new,y_new=zip(*position_samples)
#        x_new=np.array(x_new)+position[0]
#        y_new=np.array(y_new)+position[1]
#
#        (xrange,yrange)=((min(x),max(x)),(min(y),max(y)))
#        ax.set_xlim(*xrange)
#        ax.set_ylim(*yrange)
#
#        vis.draw_polygons(ax,[polygon])
#        #still need to deterministically determine position
#        gmm_cond.means=[list(map(add, c, position))for c in gmm_cond.means]
#        #plot marginal to express expressiveness
#        vis.visualise_gmm_marg_2D_density_gmr(ax,gmm_cond,colors=["g"])
#        plt.scatter(x_new,y_new,color='r')
#        n+=1
#        plt.show()
child_positions=np.arange(-1,1,0.5)
for cp0 in child_positions:
    for cp1 in child_positions:
        plt.scatter(cp0+position[0],cp1+position[1],color="blue",marker="o")
        print(cp+position[0],cp+position[1])
        shape=[(0, 0), (0, 1),(0.5,1),(p,p),(1, 0.5),(1,0)]
        #the original gmm needs to have u full covariance matrix, to estimate conditional matrix
        polygon = mp.map_to_polygon(shape,[0.5,0.5],position,0,(1,1))
        gmm_cond=gmm.condition([2,3],[cp0,cp1])
        position_samples=gmm_cond.sample(n_samples)
        ax = plt.gca()
        ax.set_aspect(1)

        gmm_cond.n_components
        x_new,y_new=zip(*position_samples)
        x_new=np.array(x_new)+position[0]
        y_new=np.array(y_new)+position[1]

        (xrange,yrange)=((min(x),max(x)),(min(y),max(y)))
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)

        vis.draw_polygons(ax,[polygon])
        #still need to deterministically determine position
        gmm_cond.means=[list(map(add, c, position))for c in gmm_cond.means]
        #plot marginal to express expressiveness
        vis.visualise_gmm_marg_2D_density_gmr(ax,gmm_cond,colors=["g"])
        plt.scatter(x_new,y_new,color='r')
        n+=1
        plt.show()
