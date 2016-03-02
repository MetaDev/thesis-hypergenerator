# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:53:27 2015

@author: Harald
"""
#polygon imports

import search_space
import statsmodels.api as sm
import visualisation
import utility
import fitness
from matplotlib import pyplot as plt


#learn imports
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import itertools
import scipy.stats as stats

ndata=100
data=[]
fitness_values=[]
fitness_test = lambda fitness , value : fitness > value
for i in range(ndata):
    result=[]
    #train per child
    result,_=search_space.test()
    sample_data = utility.extract_samples_attributes(result,attr_name="independent_attr",sample_name="chair")
    [data.append(ind_attr["position"]) for ind_attr in sample_data]
    
    fitness_v=fitness.dist_between_parent_child(utility.extract_samples_attributes(result,sample_name="table")[0],utility.extract_samples_attributes(result,sample_name="chair"))
    fitness_values.extend([np.average(fitness_v)]*len(sample_data))

accept = lambda f : f > 4
subset =np.asarray( [d for d,f in zip(data,fitness_values) if accept(f)])
dens_u = sm.nonparametric.KDEMultivariate(data=subset, var_type='cc', bw='normal_reference')
def get_data_bounds(data):
    mins = [np.min(v) for v in np.array(data).T]
    maxs = [np.max(v) for v in np.array(data).T]
    return mins,maxs

def random_distr(data_cdf):
    r = np.random.uniform(0, 1)
    s = 0
    for item, prob in data_cdf:
        s += prob
        if s >= r:
            return item
    return item
def sample(dens,data,n_samples=1,n_interp_points=None):
    if not n_interp_points:
        n_interp_points=len(data)
    (mins,maxs)= get_data_bounds(data)
    mesh = np.array(np.meshgrid(*[np.linspace(i,j,n_interp_points) for i,j in zip(mins,maxs)]))
    mesh=mesh.T
    interp_points=mesh.reshape(n_interp_points**len(mins),-1)
    x,y=zip(*interp_points)
    plt.scatter(x,y,color='g')
    cdf=dens.cdf(interp_points)
    return [random_distr(zip(interp_points,cdf)) for _ in range(n_samples)]
samples=sample(dens_u,subset,n_samples=10)
x,y=zip(*subset)
x_new,y_new=zip(*samples)
plt.scatter(x,y,color='r')
plt.scatter(x_new,y_new,color='b')
plt.set_as