

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
        parent_Y_vars=[root.independent_vars[name] for name in Y_var_names]
        child0_vars=[child0.independent_vars[name] for name in X_var_names]
        child1_vars=[child1.independent_vars[name] for name in X_var_names]
        #the independent variable is
        data0.append(np.hstack((child0_vars,parent_Y_vars)).flatten())
        data1.append(np.hstack((child1_vars,child0_vars,parent_Y_vars)).flatten())
        fitness_sum0= sum([f(child1,root)**32 for f in  parent_child_fitness_funcs])
        fitness_sum1= sum([f(child1,child0) for f in  child_child_fitness_funcs]) * sum([f(child1,root)**32 for f in  parent_child_fitness_funcs])

        fitness0.append(fitness_sum0)
        fitness1.append(fitness_sum1)
    return data0,data1,fitness0,fitness1



for i in range(ndata):
    #train per child
    s_data0,s_data1,s_fitness0,s_fitness1=feauture_fitness_child_child(model_root.sample(),
                                                               child_child_fitness_funcs,
                                                               parent_child_fitness_funcs,
                                                               X_var_names,Y_var_names)
    data0.extend(s_data0)
    data1.extend(s_data1)
    fitness_values0.extend(s_fitness0)
    fitness_values1.extend(s_fitness1)

#vis.print_fitness_statistics(fitness_values)

child0_x,child0_y,shape_x,shape_y=zip(*data0)

#plt.scatter(child1_x,child1_y,c=fitness_values,cmap=cm.Blues)
#plt.colorbar()
#plt.show()

n_components=10
#both full and tied covariance work
#but tied prevents overfitting but is worse to find conditional estimatation of sharp corners
gmm0 = lut.GMM_weighted(n_components=n_components, covariance_type='full',min_covar=0.01,random_state=setting_values.random_state)
gmm0.weighted_fit(data0,fitness_values0)

gmm1 = lut.GMM_weighted(n_components=n_components, covariance_type='full',min_covar=0.01,random_state=setting_values.random_state)
gmm1.weighted_fit(data1,fitness_values1)


#
#gmm= mixture.GMM(n_components=n_components, covariance_type='full',min_covar=0.01,random_state=setting_values.random_state)
#gmm.fit(data)
#convert grm GMM
gmm0 = GMM(n_components=len(gmm0.weights_), priors=gmm0.weights_, means=gmm0.means_, covariances=gmm0._get_covars(), random_state=setting_values.random_state)

gmm1 = GMM(n_components=len(gmm1.weights_), priors=gmm1.weights_, means=gmm1.means_, covariances=gmm1._get_covars(), random_state=setting_values.random_state)
from gmr import MVN
def marginalise_gmm(gmm,indices):
    mvns=[]
    for k in range(len(gmm.means)):
        mvn = MVN(mean=gmm.means[k], covariance=gmm.covariances[k],random_state=gmm.random_state)
        mvn= mvn.marginalize(indices)
        mvns.append(mvn)
    means=np.array([mvn.mean for mvn in mvns])
    covariances=np.array([mvn.covariance for mvn in mvns])
    return GMM(gmm.n_components,gmm.priors,means,covariances,gmm.random_state)
test_gmm0 = marginalise_gmm(gmm1,[2,3,4,5])

print(test_gmm0.means,gmm0.means)

gmm0=test_gmm0
parent_position=np.array([1,2])
x= np.array(child0_x)+parent_position[0]
y=np.array(child0_y)+parent_position[1]
(xrange,yrange)=((min(x),max(x)),(min(y),max(y)))

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
        gmm_cond.means=[list(map(add, c, np.array(parent_position))) for c in gmm_cond.means]

        #plot marginal to express expressiveness
        vis.visualise_gmm_marg_2D_density_gmr(ax,gmm_cond,colors=["g"])
        plt.show()
# -*- coding: utf-8 -*-