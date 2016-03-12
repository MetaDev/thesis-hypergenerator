# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
#polygon imports

import model.search_space as sp
import model.mapping as mp
import util.visualisation as vis
import util.utility as ut
import model.fitness as fn
import learning.gmm_sk_ext as lut


from learning.gmm import GMM
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as ss
from model.search_space import VectorVariableUtility

from util import setting_values
import matplotlib.cm as cm
import model.test_models as tm
from operator import add
from model.search_space import GMMVariable,DummyStochasticVariable

ndata=200

data=[]
fitness_values=[]
polygons_vis=[]

import learning.training as tr


model_root=tm.test_model_var_child_position_parent_shape()
X_var_names=["position"]
Y_var_names=["shape3"]
fitness_funcs=[tr.fitness_polygon_overl]

for i in range(ndata):
    #train per child
    sample_features,sample_fitness=tr.feauture_fitness_extraction(model_root.sample(),
                                                              fitness_funcs,
                                                               X_var_names,Y_var_names)
    data.extend(sample_features)
    fitness_values.extend(sample_fitness)


#better statistics of fittnes
vis.print_fitness_statistics(fitness_values)


x,y,x_shape,y_shape=zip(*data)

n_components=10
n_samples=100




#both full and tied covariance work
#but tied prevents overfitting but is worse to find conditional estimatation of sharp corners
gmm0 = GMM(n_components=n_components,random_state=setting_values.random_state)

gmm0.fit(data,weights=fitness_values,min_covar=0.01)
#convert grm GMM
position_samples=gmm0.sample(n=n_samples)

#visualise
if(plt.gcf()==0):
    fig = plt.figure("name")
fig=plt.gcf()
fig.clear()


ax = plt.gca()
vis.visualise_gmm_marg_density_1D_gmr(ax,0,gmm0,verbose=True)

plt.show()

ax = plt.gca()

plt.scatter(x,y,c=fitness_values,cmap=cm.Blues)
plt.colorbar()
plt.show()

#construct new model
child = model_root.get_child_node("child")

X_dummy_vars=[DummyStochasticVariable(child.get_variable("position"))]
Y_vars=[model_root.get_variable(name) for name in Y_var_names]
gmm_var=GMMVariable("test",gmm0,X_dummy_vars,Y_vars)

model_root.get_child_node("child").set_learned_variable(gmm_var)

position=(1,2)
x= np.array(x)+position[0]
y=np.array(y)+position[1]
n=1;
values=np.arange(0.5,2,0.25)
for p in values:
    shape=[(0, 0), (0, 1),(0.5,1),(p,p),(1, 0.5),(1,0)]
    #the original gmm needs to have u full covariance matrix, to estimate conditional matrix
    polygon = mp.map_to_polygon(shape,[0.5,0.5],position,0,(1,1))
    gmm_cond=gmm0.condition([2,3],[p,p])
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
    gmm_cond._means=[list(map(add, c, position))for c in gmm_cond._means]
    #plot marginal to express expressiveness
    vis.visualise_gmm_marg_2D_density_gmr(ax,gmm_cond,colors=["g"])
    plt.scatter(x_new,y_new,color='r')
    n+=1
    plt.show()




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

x,y,x_shape,y_shape=zip(*data)

#better statistics of fittnes
fitness_values=fitness_values
plt.scatter(x,y,c=fitness_values,cmap=cm.Blues)
plt.colorbar()
plt.show()
vis.print_fitness_statistics(fitness_values)