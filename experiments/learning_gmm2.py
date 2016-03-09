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
ndata=200

import learning.training as tr


model_root=tm.test_samples_var_child_pos_size_rot_parent_shape()
X_var_names=["position","rotation","size"]
Y_var_names=["shape3"]
fitness_funcs=[tr.fitness_polygon_alignment,tr.fitness_polygon_overl]
data=[]
fitness_values=[]
for i in range(ndata):
    #train per child
    sample_features,sample_fitness=tr.feauture_fitness_extraction(model_root.sample(),
                                                               fitness_funcs,
                                                               X_var_names,Y_var_names)
    data.extend(sample_features)
    fitness_values.extend(sample_fitness)

#give heavy penalty for intersection
#inverse and normalise fitness
#higher fitness means less overlap
print(max(fitness_values))
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

#visualise
if(plt.gcf()==0):
    fig = plt.figure("name")
fig=plt.gcf()
fig.clear()


ax = plt.gca()
ax.set_title("x")
vis.visualise_gmm_marg_density_1D_gmr(ax,0,gmm,verbose=True)
plt.show()

ax = plt.gca()
ax.set_title("y")
vis.visualise_gmm_marg_density_1D_gmr(ax,1,gmm,verbose=True)
plt.show()

ax = plt.gca()
ax.set_title("rotation")
vis.visualise_gmm_marg_density_1D_gmr(ax,2,gmm,verbose=True)
plt.show()

#construct new model
child = model_root.get_child_node("child")

X_dummy_vars=[DummyStochasticVariable(child.get_variable("position"))]
Y_vars=[model_root.get_variable(name) for name in Y_var_names]
gmm_var=GMMVariable("test",gmm,X_dummy_vars,Y_vars)

model_root.get_child_node("child").set_learned_variable(gmm_var)


#repeat sampling
data=[]
fitness_values=[]

#the variable of the child has been replaced by trained model

for i in range(ndata):
    #train per child
    sample_features,sample_fitness=tr.feauture_fitness_extraction(model_root.sample(),
                                                               fitness_funcs,
                                                               X_var_names,Y_var_names)
    data.extend(sample_features)
    fitness_values.extend(sample_fitness)

x,y,rotation,sizex,sizey,x_shape,y_shape=zip(*data)

#better statistics of fittnes
plt.scatter(x,y,c=fitness_values,cmap=cm.Blues)
plt.colorbar()
plt.show()
vis.print_fitness_statistics(fitness_values)