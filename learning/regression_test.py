# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
#polygon imports

import model.search_space as sp
import model.mapping as mp
import util.visualisation as vis
import util.utility as ut
import model.fitness as fn

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
from learning.gmm import GMM


ndata=200
n_components=15
import learning.training as tr

from scipy.stats import multivariate_normal
from scipy.stats import mvn
#from sklearn_bayes.mixture import VBGMMARD

#experiment hyperparameters: sibling_order,
#sibling order 0 is
sibling_order=2
ndata=100
n_components=15

#training variables and fitness functions
vars_children=["position"]
vars_parent=["shape3"]
sibling_fitness_funcs={tr.fitness_min_dist:2}
parent_child_fitness_funcs={tr.fitness_polygon_overl:32}

#model to train on
root=tm.test_model_var_child_position_parent_shape()


#train parent-child


#get size of vars
X_var_length=np.sum([root.children["child"][1].get_variable(name).size for name in vars_children])
Y_var_length=np.sum([root.get_variable(name).size for name in vars_parent])


data,fitness=tr.parent_child_variable_training(ndata,root,parent_child_fitness_funcs,
sibling_fitness_funcs,vars_parent,vars_children,sibling_order)

#cap the data


print("parent fitness")
vis.print_fitness_statistics(fitness["parent"])
if sibling_order>0:
    print("child fitness")
    vis.print_fitness_statistics(fitness["children"])
print("total fitness")
vis.print_fitness_statistics(fitness["all"],print_hist=True)

data, fitness_p,fitness_c= zip(*[(d,fp,fc) for d,fp,fc in zip(data,fitness["parent"],fitness["children"]) if fp>0.0001 and fc > 0.0001])


#stack the data
train_data=np.column_stack((data,fitness_p,fitness_c))

#calculate the joint of both fitness and vars P(X,Y)
#gmm = GMM(n_components=n_components,random_state=setting_values.random_state)
prune_thresh = 1e-2
max_k=30
#gmm = VBGMMARD(n_components = max_k, prune_thresh = prune_thresh)
import gmm_sk_ext
gmm = gmm_sk_ext.VBGMMARD(n_components = max_k, prune_thresh = prune_thresh)
gmm.fit_weighted(data,fitness_p)
print(len(gmm.means_))
gmm= GMM(weights=gmm.weights_,means=gmm.means_,covariances=gmm.covars_)
#fitness_gmm = gmm.condition([len(train_data[0])-2,len(train_data[0])-1],[1.1,1])
#fitness_gmm = gmm.condition([len(train_data[0])-2],[1])

gmm_points=gmm.marginalise([0,1])
points=np.array(gmm_points.sample(400))
x,y=zip(*points)
ax=plt.gca()
ax.scatter(x,y,color="r")

vis.visualise_gmm_marg_2D_density(ax,gmm=gmm_points,colors=['g'])
plt.show()
#
##sample from the gmm truncuated 1 dim
#from scipy.stats import truncnorm
#import random
##randomly choose 1 component, than sample from the truncuated normal
#i = random.random()
#for w,m,c in zip( *fitness_gmm.get_params()):
#    if w > i:
#        y_= sample_trunc(m,c)
#    i -= w
#def sample_trunc(lower=0.5,upper=1.5,mean,var):
#    std=var**(1/2.0)
#    a, b = (lower - mean) / std, (fitness_upper - mean) / std
#    return truncnorm.rvs(a, b,size=1)


#sample from conditional P(X|Y=y_)

#the other approach to sampling from P(X|Y>y) would be by using inverse transform sampling on the cdf P(Y>t),
#the cdf of a gmm can be calculated using scipy
#calculate cdf values t-1 and use this to apply inverse transform sampling with interpolation
