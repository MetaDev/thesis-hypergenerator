# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:53:27 2015

@author: Harald
"""
#polygon imports

import search_space
import mapping

import visualisation
import utility
import fitness

#learn imports
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture


ndata=100
data=[]
fitness_values=[]
for i in range(ndata):
    result=search_space.test()
    d=[sample.children[0][0] for sample in result if sample.name.startswith("table")]
    data.append(d)
    fitness_values.append(fitness.dist_between_parent_child(utility.extract_samples_attributes(d,sample_name="table")[0],utility.extract_sample_from_name(d,sample_name="chair")))

print(fitness_values)
print(d)
##learn stuff
#
## Number of samples per component
#n_samples = 1000
#max_point=-3
#min_point=3
#
## Generate random sample, two components
#np.random.seed(0)
#points= [np.random.uniform(min_point,max_point,2) for _ in range(n_samples)]
#X= [np.append(p,polygon.contains(Point(p))) for p in points]
##X=[np.append(p,polygon.distance(Point(p))) for p in points]
#
#
## Fit a mixture of Gaussians with EM using five components
#gmm = mixture.GMM(n_components=3, covariance_type='full')
#
#
#insideX = [x[:2] for x in X if x[2] ==1]
#
#gmm.fit(insideX)
#make_ellipses(gmm,fig)
##plot the points
##plt.scatter(*zip(*points), c=list(zip(*X))[2])
#plt.scatter(*zip(*insideX))
#plt.scatter(*zip(*gmm.sample(100)),color='r')
#plt.show()
#plt.pause(1) # <-------
#
#input("<Hit Enter To Show generated samples>")
#plt.close(fig)
#
#
#plt.show()