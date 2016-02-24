# -*- coding: utf-8 -*-
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
import matplotlib as mpl
import matplotlib.cm as cm

import learning_utility
ndata=100

data={}
fitness_values=[]
fitness_test = lambda fitness , value : fitness > value
for i in range(ndata):
    result=[]
    #train per child
    result,_=search_space.test()
    sample_data = utility.extract_samples_attributes(result,attr_name="independent_attr",sample_name="chair")
    [data.append(ind_attr["position"]) for ind_attr in sample_data]
    
    fitness_v=fitness.dist_between_parent_child(utility.extract_samples_attributes(result,sample_name="table")[0],utility.extract_samples_attributes(result,sample_name="chair"))
    fitness_values.extend(([fitness_v]*len(sample_data)))


accept = lambda f : f > 10
subset,fitness_subset = zip(*[(d,f) for d,f in zip(data,fitness_values) if accept(f)])
x,y=zip(*subset)
full_subset=c = np.column_stack((x,y,fitness_subset))
print(full_subset[0])
# Fit a mixture of Gaussians with EM using five components
gmm = mixture.GMM(n_components=10, covariance_type='full',min_covar=0.0000001)

norm_f_subs=(fitness_subset-np.min(fitness_subset))/(np.max(fitness_subset)-np.min(fitness_subset))

print(gmm.n_components)


ax=visualisation.init()
gmm.fit(full_subset)

y=15

(con_cen, con_cov, new_p_k) = learning_utility.cond_dist(np.array([np.nan,np.nan, y]), gmm.means_, gmm._get_covars(), gmm.weights_)
#visualisation.make_ellipses(gmm,plt.gcf())
#plot the points
x,y=zip(*subset)
plt.scatter(x,y,c=norm_f_subs,cmap=cm.Blues)
plt.colorbar()
samples=learning_utility.sample_gaussian_mixture(con_cen, con_cov, new_p_k,samples=100)
plt.scatter(*zip(*samples),color='r')

plt.show()