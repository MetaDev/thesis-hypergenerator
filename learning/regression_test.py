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

import learning.training as tr

X_var_names=["position"]
Y_var_names=["shape3"]
child_child_fitness_funcs={(tr.fitness_min_dist,2)}
parent_child_fitness_funcs={(tr.fitness_polygon_overl,32)}

#calculate the joint of both fitness and vars P(X,Y)

#marginalise P(Y) ->gmm

#sample from the gmm truncuated between hyperplane P(Y>y) using truncnorm  -> y'
#http://stackoverflow.com/questions/22047786/vectorised-code-for-sampling-from-truncated-normal-distributions-with-different

#sample from conditional P(X|Y=y')

#the other approach to sampling from P(X|Y>y) would be by using inverse transform sampling on the cdf P(Y>t),
#the cdf of a gmm can be calculated using scipy
#calculate cdf values t-1 and use this to apply inverse transform sampling with interpolation
