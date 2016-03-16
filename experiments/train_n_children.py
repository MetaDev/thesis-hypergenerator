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

from itertools import combinations

from itertools import product
#train parent-child

data=[]
fitness_values=[]
polygons_vis=[]

root_sample=tm.test_model_var_child_position_parent_shape()

X_var_names=["position"]
Y_var_names=["shape3"]
child_child_fitness_funcs=[tr.fitness_min_dist]
parent_child_fitness_funcs=[tr.fitness_polygon_overl]


#TODO
#train model P(C0|C1,P)
ndata=100

#repeat sampling
data=[]
fitness_values=[]
#n_children=model_root.get_child("child")[0].sample(None)
#get size of vars
X_var_length=np.sum([root_sample.get_children("child",as_list=True)[1].get_variable(name).size for name in X_var_names])
Y_var_length=np.sum([root_sample.get_variable(name).size for name in Y_var_names])
n_children=len(root_sample.get_children("child",as_list=True))


n_children=len(root_sample.get_children("child",as_list=True))


#use root
def feauture_fitness_child_child(root,child_child_fitness_funcs,parent_child_fitness_funcs,
                                 X_var_names,Y_var_names):
    #only get children of a certain name

    data=[]
    fitness=[]
    parent_Y_vars=[root.independent_vars[name] for name in Y_var_names]
    for children in combinations(root.get_children("child"),n_children):

        child_vars=[]
        fitness_prod=1

        #calculate fitness of each child to parent
        for child in children:
            child_vars.extend([child.independent_vars[name] for name in X_var_names])
            fitness_prod*=sum([f(child,root)**32 for f in  parent_child_fitness_funcs])
        #the independent variable is
        child_vars=np.array(child_vars).flatten()
        parent_Y_vars=np.array(parent_Y_vars).flatten()
        data.append(np.hstack((child_vars,parent_Y_vars)))

        #calculate pairwise fitness between children
        for child0,child1 in combinations(children,2):
            fitness_prod*= sum([f(child0,child1)**2 for f in  child_child_fitness_funcs])

        fitness.append(fitness_prod)
    return data,fitness



for i in range(ndata):
    #train per child
    root_sample.sample()
    s_data,s_fitness=feauture_fitness_child_child(root_sample,
                                                               child_child_fitness_funcs,
                                                               parent_child_fitness_funcs,
                                                               X_var_names,Y_var_names)
    data.extend(s_data)
    fitness_values.extend(s_fitness)


#cap the data

data, fitness_values= zip(*[(d,f) for d,f in zip(data,fitness_values) if f>0.0001])


#vis.print_fitness_statistics(fitness_values)

n_components=15

gmm = GMM(n_components=n_components,random_state=setting_values.random_state)
gmm.fit(data,fitness_values,min_covar=0.01)

gmms=[None]*n_children
gmms[n_children-1]=gmm
#marginalise for each child
#P(c_n,c_n-1,..,c0,p)->P(c_i,c_i+1,..,c_0,p)
for i in range(n_children-1):
    indices=np.arange((n_children-(i+1))*X_var_length,n_children*X_var_length+Y_var_length)
    print(indices)
    gmms[i]=gmm.marginalise(indices)

(xrange,yrange)=((-2,2),(-2,2))

parent_position=np.array([1,2])

(xrange,yrange)=((-2,4),(-1,5))
p_shape_points=[0.5,1,1.5]
for p_shape in zip(p_shape_points,p_shape_points):
    shape=[(0, 0), (0, 1),(0.5,1),p_shape,(1, 0.5),(1,0)]

    polygon = mp.map_to_polygon(shape,[0.5,0.5],parent_position,0,(1,1))

    previous_gmm_cond=None
    from copy import deepcopy
    #visualise the conditional distribution of children
    #P(c_i,c_i-1,..,c_0,p) -> P(c_i|c_i-1,..,c_0,p)
    values=p_shape
    points=[]
    for i in range(0,n_children):

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
        print(indices,values)
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
Y_sibling_vars=[first_child.get_variable(name) for name in X_var_names]
Y_vars=[root_sample.get_variable(name) for name in Y_var_names]

gmm_vars=[GMMVariable("test",gmm,X_dummy_vars,Y_vars,Y_sibling_vars,i) for gmm,i in zip(gmms,range(len(gmms)))]
for var in gmm_vars:
    print(var.size,var.sibling_order,var.gmm_size)
children = root_sample.get_children("child")
#the gmms are ordered the same as the children
for child,gmm in zip(children,gmm_vars):
    print(child.index,gmm.sibling_order)
    child.set_learned_variable(gmm)
import util.visualisation as vis
#sample from it
for i in range(10):
    root_sample.sample()
    vis.draw_node_sample_tree(root_sample)
    plt.show()


