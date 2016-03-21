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


vars_children=["position"]
vars_parent=["shape3"]
sibling_fitness_funcs={(tr.fitness_min_dist,2)}
parent_child_fitness_funcs={(tr.fitness_polygon_overl,32)}

#model to train on
root_sample=tm.test_model_var_child_position_parent_shape()
sibling_order=2

data,fitness=tr.parent_child_variable_training(ndata,root_sample,parent_child_fitness_funcs,
sibling_fitness_funcs,vars_parent,vars_children,sibling_order)

data_cap, fitness_prod_cap= zip(*[(d,f) for d,f in zip(data,fitness["all"]) if f>0.001])



#stack the data
train_data=np.hstack((data_cap,fitness_prod_cap))

#calculate the joint of both fitness and vars P(X,Y)
gmm = GMM(n_components=n_components,random_state=setting_values.random_state)

gmm.fit(train_data,min_covar=0.01)
#marginalise P(Y) ->gmm
fitness_gmm=gmm.marginalise([len(train_data)-1])

#sample from the gmm truncuated 1 dim
from scipy.stats import truncnorm
import random
#randomly choose 1 component, than sample from the truncuated normal
i = random.random()
for w,m,c in zip( *fitness_gmm.get_params()):
    if w > i:
        y_= sample_trunc(m,c)
    i -= w
def sample_trunc(lower=0.5,upper=1.5,mean,var):
    std=var**(1/2.0)
    a, b = (lower - mean) / std, (fitness_upper - mean) / std
    return truncnorm.rvs(a, b,size=1)


#sample from conditional P(X|Y=y_)

#the other approach to sampling from P(X|Y>y) would be by using inverse transform sampling on the cdf P(Y>t),
#the cdf of a gmm can be calculated using scipy
#calculate cdf values t-1 and use this to apply inverse transform sampling with interpolation
