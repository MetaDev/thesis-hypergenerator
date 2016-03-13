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

model_root=tm.test_model_var_child_position_parent_shape()

X_var_names=["position"]
Y_var_names=["shape3"]
child_child_fitness_funcs=[tr.fitness_min_dist]
parent_child_fitness_funcs=[tr.fitness_polygon_overl]


#TODO
#train model P(C0|C1,P)
ndata=500

#repeat sampling
data=[]
fitness_values=[]
#n_children=model_root.get_child("child")[0].sample(None)
n_children=2
#get size of vars
X_var_length=np.sum([model_root.get_child("child")[1].get_variable(name).size for name in X_var_names])
Y_var_length=np.sum([model_root.get_variable(name).size for name in Y_var_names])

data_length=n_children*X_var_length+Y_var_length



#use root
def feauture_fitness_child_child(root,child_child_fitness_funcs,parent_child_fitness_funcs,
                                 X_var_names,Y_var_names):
    #only get children of a certain name

    data=[]

    children = root.get_children("child")[:n_children]

    parent_Y_vars=[root.independent_vars[name] for name in Y_var_names]
    child_vars=[]
    fitness_prod=1

    #calculate fitness of each child to parent
    #TODO watch out for the order of children
    for child in children:
        child_vars.extend([child.independent_vars[name] for name in X_var_names])
        fitness_prod*=sum([f(child,root)**32 for f in  parent_child_fitness_funcs])
    #the independent variable is
    child_vars=np.array(child_vars).flatten()
    parent_Y_vars=np.array(parent_Y_vars).flatten()
    data.append(np.hstack((child_vars,parent_Y_vars)))

    #calculate pairwise fitness between children
    for child0,child1 in combinations(children,2):
        fitness_prod*= sum([f(child0,child1) for f in  child_child_fitness_funcs])
    return data,fitness_prod



for i in range(ndata):
    #train per child
    s_data,s_fitness=feauture_fitness_child_child(model_root.sample(),
                                                               child_child_fitness_funcs,
                                                               parent_child_fitness_funcs,
                                                               X_var_names,Y_var_names)
    data.extend(s_data)
    fitness_values.append(s_fitness)


#cap the data

capped_data, capped_fitness= zip(*[(d,f) for d,f in zip(data,fitness_values) if f>0.3])

print(len(capped_data),len(capped_fitness))
vis.print_fitness_statistics(capped_fitness)
data=capped_data
fitness_values=capped_fitness
n_components=10

gmm = GMM(n_components=n_components,random_state=setting_values.random_state)
gmm.fit(data,fitness_values,min_covar=0.01)

gmms=[None]*n_children
#marginalise for each child
#P(c_n,c_n-1,..,c0,p)->P(c_i,c_i+1,..,c_0,p)
for i in range(n_children):
    indices=np.arange((n_children-(i+1))*X_var_length,n_children*X_var_length+Y_var_length)
    print(indices)
    gmms[i]=gmm.marginalise(indices)

#plot marginal for each child
#plot up to 2 dimensions
for i in range(n_children):
    print(i)
    ax = plt.gca()
    ax.set_aspect(1)
    vis.visualise_gmm_marg_2D_density(ax,gmm=gmms[i].marginalise([0,1]),colors=["g"])
    plt.show()

parent_position=np.array([1,2])

(xrange,yrange)=((-1,3),(0,4))

values=[0.5,1,1.5]
parent_shape=[1,1]
shape=[(0, 0), (0, 1),(0.5,1),parent_shape,(1, 0.5),(1,0)]
polygon = mp.map_to_polygon(shape,[0.5,0.5],parent_position,0,(1,1))

previous_gmm_cond=None
values=parent_shape
#visualise the conditional distribution of children
#P(c_i,c_i-1,..,c_0,p) -> P(c_i|c_i-1,..,c_0,p)
for i in range(n_children):
    print(i)
    indices=np.arange((i+1)*X_var_length,(i+1)*X_var_length+Y_var_length)
    if previous_gmm_cond:
        values=previous_gmm_cond.sample(1) + values
    values=np.array(values).flatten()
    gmm_cond=gmms[i].condition(indices,values)
    previous_gmm_cond=gmm_cond

    ax = plt.gca()
    ax.set_aspect(1)
    gmm_show=gmm_cond.marginalise([0,1])
    gmm_show._means=[list(map(add, c, np.array(parent_position))) for c in gmm_show._means]
    gmm_show._set_gmms()
    vis.visualise_gmm_marg_2D_density(ax,gmm=gmm_show,colors=["g"])

    vis.draw_polygons(ax,[polygon])
    plt.show()

