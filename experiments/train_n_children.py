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

#experiment hyperparameters: sibling_order,
sibling_order=0
ndata=100
n_components=15

#training variables and fitness functions
vars_children=["position"]
vars_parent=["shape3"]
sibling_fitness_funcs={tr.fitness_min_dist:2}
parent_child_fitness_funcs={tr.fitness_polygon_overl:32}

#model to train on
root_sample=tm.test_model_var_child_position_parent_shape()


#a model trained on 4 children can also be used for 5 children of the fifth no longer conditions on the first
#These models can be reused because there's no difference in the markov chain of order n between the n+1 and n+2 state
repeat_trained_model=True


#train parent-child

polygons_vis=[]





#get size of vars
X_var_length=np.sum([root_sample.get_children("child",as_list=True)[1].get_variable(name).size for name in vars_children])
Y_var_length=np.sum([root_sample.get_variable(name).size for name in vars_parent])


data,fitness=tr.parent_child_variable_training(ndata,root_sample,parent_child_fitness_funcs,
sibling_fitness_funcs,vars_parent,vars_children,sibling_order)

print(len(data))
#cap the data



vis.print_fitness_statistics(fitness["parent"])
if sibling_order>0:
    vis.print_fitness_statistics(fitness["children"])
vis.print_fitness_statistics(fitness["all"],print_hist=True)

data_cap, fitness_prod_cap= zip(*[(d,f) for d,f in zip(data,fitness["all"]) if f>0.001])


gmm = GMM(n_components=n_components,random_state=setting_values.random_state)
gmm.fit(data_cap,fitness_prod_cap,min_covar=0.01)
#if sibling order is 0 you have only a single gmm
gmms=[None]*(sibling_order+1)
gmms[sibling_order]=gmm
#marginalise for each child
#P(c_n,c_n-1,..,c0,p)->P(c_i,c_i+1,..,c_0,p)
for i in range(sibling_order):
    indices=np.arange((sibling_order-(i))*X_var_length,(sibling_order+1)*X_var_length+Y_var_length)
    gmms[i]=gmm.marginalise(indices)

(xrange,yrange)=((-2,2),(-2,2))

parent_position=np.array([1,2])

(xrange,yrange)=((-2,4),(-1,5))
p_shape_points=[0.5,1,1.5]
#TODO
#visualisation code needs cleaning up
for p_shape in zip(p_shape_points,p_shape_points):
    shape=[(0, 0), (0, 1),(0.5,1),p_shape,(1, 0.5),(1,0)]

    polygon = mp.map_to_polygon(shape,[0.5,0.5],parent_position,0,(1,1))

    previous_gmm_cond=None
    from copy import deepcopy
    #visualise the conditional distribution of children
    #P(c_i,c_i-1,..,c_0,p) -> P(c_i|c_i-1,..,c_0,p)
    values=p_shape
    points=[]
    for i in range(sibling_order+1):

        ax = plt.gca()
        ax.set_aspect(1)
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        print("child: ",i)
        indices=np.arange(X_var_length,(i+1)*X_var_length+Y_var_length)

        if previous_gmm_cond:
            sample=np.array(previous_gmm_cond.sample(1)).flatten()
            values=np.hstack((np.array(sample),np.array(values)))
            point= np.array(sample) + np.array(parent_position)
            points.append(point)
            x,y=zip(*points)
            ax.scatter(x,y,color="r")

        values=np.array(values).flatten()
        gmm_cond=gmms[i].condition(indices,values)
        previous_gmm_cond=gmm_cond

        gmm_show=deepcopy(gmm_cond)
        gmm_show._means=[list(map(add, c, np.array(parent_position))) for c in gmm_show._means]
        gmm_show._set_gmms()
        vis.visualise_gmm_marg_2D_density(ax,gmm=gmm_show,colors=["g"])

        vis.draw_polygons([polygon],ax)
        plt.show()

#construct new model
from model.search_space import GMMVariable,DummyStochasticVariable
first_child=root_sample.get_children("child",as_list=True)[0]

X_dummy_vars=[DummyStochasticVariable(first_child.get_variable("position"))]
Y_sibling_vars=[first_child.get_variable(name) for name in vars_children]
Y_vars=[root_sample.get_variable(name) for name in vars_parent]
#the sibling order of the gmm is maximum the sibling order of the trained gmm
gmm_vars=[GMMVariable("test",gmm,X_dummy_vars,Y_vars,Y_sibling_vars,sibling_order=i) for gmm,i in zip(gmms,range(len(gmms)))]

children = root_sample.get_children("child",as_list=True)
#the gmms are ordered the same as the children

#assign variable child i with gmm min(i,sibling_order)
if repeat_trained_model:
    learned_var_range=len(children)
else:
    learned_var_range=len(gmms)
for i in range(learned_var_range):
    children[i].set_learned_variable(gmm_vars[min(i,sibling_order)])

#re-estimate average fitness
#repeat sampling
data,fitness=tr.parent_child_variable_training(ndata,root_sample,parent_child_fitness_funcs,
sibling_fitness_funcs,vars_parent,vars_children,sibling_order)


#cap the data


vis.print_fitness_statistics(fitness["parent"])
if sibling_order>0:
    vis.print_fitness_statistics(fitness["children"])