# -*- coding: utf-8 -*-
#polygon imports

import search_space
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
from skimage import exposure
import matplotlib.cm as cm
import mapping
import learning_utility

from sklearn.decomposition import PCA

ndata=100

data=[]
fitness_values=[]
polygons_vis=[]

def feauture_fitness_extraction(samples,fitness_func):
    chairs=utility.extract_samples_attributes(samples,sample_name="chair")
    ind_attrs = utility.extract_samples_attributes(chairs,attr_name="independent_attr")
    indexs = utility.extract_samples_attributes(chairs,attr_name="index")
    parents = utility.extract_samples_attributes(chairs,attr_name="parent")
    data=[(ind_attr["position"][0],ind_attr["position"][1],index,len(parent.children)) for ind_attr,index,parent in zip(ind_attrs,indexs,parents)]
    fitness=[fitness_func([chair,parent]) for chair,parent in zip(chairs,parents)]
    return data,fitness
def fitness_extraction_dist(samples):
    return fitness.dist_between_parent_child(utility.extract_samples_attributes(samples,sample_name="table")[0],utility.extract_samples_attributes(samples,sample_name="chair"))
def fitness_extraction_overl(samples):
    polygons=mapping.map_layoutsamples_to_geometricobjects(samples)
    return fitness.pairwise_overlap(polygons,normalized=True)
#todo add fitness close to L
#augment data using multiple fitness
#train on parametrised shape
for i in range(ndata):
    #train per child
    samples,_=search_space.test()
    sample_features,sample_fitness=feauture_fitness_extraction(samples,fitness_extraction_overl)
    data.extend(sample_features)
    fitness_values.extend(sample_fitness)
    
f_value=0.1



filtered_data=False
if filtered_data:
    cut_off_value=0
    accept = lambda f, value: f < value
    data,fitness_values = zip(*[(d,f) for d,f in zip(data,fitness_values) if accept(f,cut_off_value)])


x,y,index,parent_children=zip(*data)

if(plt.gcf()==0):
    fig = plt.figure("name")
fig=plt.gcf()
fig.clear()

ax = fig.add_subplot(1,1,1)
ax.set_aspect(1)



position=np.array(utility.extract_samples_attributes(samples,attr_name="position",sample_name="table"))[0]


n_components=10
n_samples=len(data)*5
conditional=False
weighted=True

if conditional:
    gmm = mixture.GMM(n_components=n_components, covariance_type='full',min_covar=0.0000001)
    training = np.column_stack((x,y,fitness_values))
    gmm.fit(training)
    y_0=15
    (con_cen, con_cov, new_p_k) = learning_utility.cond_dist(np.array([np.nan,np.nan, 0.001]), gmm.means_, gmm._get_covars(), gmm.weights_)
    position_samples=learning_utility.sample_gaussian_mixture(con_cen, con_cov, new_p_k,samples=n_samples)
    x_new,y_new=zip(*samples)
elif weighted:
    wgmm = learning_utility.GMM_weighted(n_components=n_components,n_init=1,verbose=1, covariance_type='diag',min_covar=0.00001)
    training = np.column_stack((x,y))
    #give heavy penalty for intersection
    #inverse and normalise fitness
    fitness_values=fitness.normalise_fitness(fitness.invert_fitness(fitness_values))**6
    wgmm.weighted_fit(training,fitness_values)
    
    position_samples=learning_utility.sample_gaussian_mixture(wgmm.means_,wgmm._get_covars(),wgmm.weights_,samples=n_samples,loose_norm_weights=True)
else:
    gmm = mixture.GMM(n_components=n_components, covariance_type='full',min_covar=0.0000001)
    training = np.column_stack((x,y))
    gmm.fit(training)
    position_samples=gmm.sample(n_samples)
 
#don't visualise non overlapping fitness


table=mapping.map_layoutsamples_to_geometricobjects(utility.extract_samples_attributes(samples,sample_name="table"))
#visualisation.draw_polygons(ax,table)
x_new,y_new=zip(*position_samples)
x_new=np.array(x_new)+position[0]
y_new=np.array(y_new)+position[1]
plt.scatter(x_new,y_new,color='r')

x=np.array(x)+position[0]
y=np.array(y)+position[1]

plt.scatter(x,y,c=fitness_values,cmap=cm.Blues)
plt.colorbar()
#
#
(xrange,yrange)=((min(x_new),max(x_new)),(min(y_new),max(y_new)))
ax.set_xlim(*xrange)
ax.set_ylim(*yrange)
print(wgmm.converged_)


plt.show()