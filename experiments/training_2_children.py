

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
ndata=300

#repeat sampling
data0=[]
data1=[]
fitness_values0=[]
fitness_values1=[]
#the variable of the child has been replaced by trained model
#use root
def feauture_fitness_child_child(root,child_child_fitness_funcs,parent_child_fitness_funcs,
                                 X_var_names,Y_var_names):
    #only get children of a certain name
    fitness0=[]
    data0=[]
    fitness1=[]
    data1=[]
    #add a pair in both ways to the data child1->child0 and child0 ->child1
    for pair in combinations(root.get_children("child"),2):
        child0=pair[0]
        child1=pair[1]
        child_vars=[]
        parent_Y_vars=[root.independent_vars[name] for name in Y_var_names]
        child0_vars=[child0.independent_vars[name] for name in X_var_names]
        child1_vars=[child1.independent_vars[name] for name in X_var_names]
        #the independent variable is
#        data0.append(np.hstack((child0_vars,parent_Y_vars)).flatten())
        data1.append(np.hstack((child0_vars,child1_vars,parent_Y_vars)).flatten())
#        fitness_sum0= sum([f(child1,root)**32 for f in  parent_child_fitness_funcs])
        fitness_sum1= sum([f(child1,child0) for f in  child_child_fitness_funcs]) * sum([f(child1,root)**32 for f in  parent_child_fitness_funcs]) * sum([f(child0,root)**32 for f in  parent_child_fitness_funcs])

#        fitness0.append(fitness_sum0)
        fitness1.append(fitness_sum1)
    return data0,data1,fitness0,fitness1



for i in range(ndata):
    #train per child
    s_data0,s_data1,s_fitness0,s_fitness1=feauture_fitness_child_child(model_root.sample(),
                                                               child_child_fitness_funcs,
                                                               parent_child_fitness_funcs,
                                                               X_var_names,Y_var_names)
#    data0.extend(s_data0)
    data1.extend(s_data1)
#    fitness_values0.extend(s_fitness0)
    fitness_values1.extend(s_fitness1)

vis.print_fitness_statistics(fitness_values1)

child0_x,child0_y,_,_,shape_x,shape_y=zip(*data1)

#plt.scatter(child1_x,child1_y,c=fitness_values,cmap=cm.Blues)
#plt.colorbar()
#plt.show()

n_components=10
#both full and tied covariance work
#but tied prevents overfitting but is worse to find conditional estimatation of sharp corners

#data0, fitness_values0= zip(*[(d,f) for d,f in zip(data0,fitness_values0) if f>0.3])

data1, fitness_values1= zip(*[(d,f) for d,f in zip(data1,fitness_values1) if f>0.3])
#
#gmm0 = GMM(n_components=n_components)
#gmm0.fit(data0,fitness_values0,min_covar=0.01)
print(len(data1),len(fitness_values1))
gmm1 = GMM(n_components=n_components,random_state=setting_values.random_state)
gmm1.fit(data1,fitness_values1,min_covar=0.01)


#
#gmm= mixture.GMM(n_components=n_components, covariance_type='full',min_covar=0.01,random_state=setting_values.random_state)
#gmm.fit(data)
#convert grm GMM


gmm0 = gmm1.marginalise([2,3,4,5])

(xrange,yrange)=((-2,4),(-1,5))

parent_position=np.array([1,2])


values=[0.5,1,1.5]
for p in values:
    shape=[(0, 0), (0, 1),(0.5,1),(p,p),(1, 0.5),(1,0)]

    polygon = mp.map_to_polygon(shape,[0.5,0.5],parent_position,0,(1,1))
    gmm_cond=gmm0.condition([2,3],[p,p])
    cps0=gmm_cond.sample(3)
    for cp0 in cps0:

        #the original gmm needs to have u full covariance matrix, to estimate conditional matrix
        gmm_cond=gmm1.condition([2,3,4,5],[cp0[0],cp0[1],p,p])

        cp0=np.array(cp0)+parent_position
        print(cp0)


        plt.scatter(*cp0,color="blue",s=16)
        ax = plt.gca()
        ax.set_aspect(1)

        cp1xs,cp1ys=zip(*gmm_cond.sample(100))
        x_new=[x+parent_position[0] for x in cp1xs]
        y_new=[y+parent_position[1] for y in cp1ys]
        plt.scatter(x_new,y_new,color='r')

        vis.draw_polygons(ax,[polygon])


        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)

        #still need to deterministically determine position
        gmm_cond._means=[list(map(add, c, np.array(parent_position))) for c in gmm_cond._means]
        gmm_cond._set_gmms()
        #plot marginal to express expressiveness
        vis.visualise_gmm_marg_2D_density(ax,gmm_cond,colors=["g"])
        plt.show()
# -*- coding: utf-8 -*-