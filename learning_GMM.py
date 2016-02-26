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
ndata=10

data=[]
fitness_values=[]
fitness_test = lambda fitness , value : fitness > value
polygons_vis=[]
def feauture_fitness_extraction(samples,fitness_func):
    chairs=utility.extract_samples_attributes(samples,sample_name="chair")
    ind_attrs = utility.extract_samples_attributes(chairs,attr_name="independent_attr")
    indexs = utility.extract_samples_attributes(chairs,attr_name="index")
    parents = utility.extract_samples_attributes(chairs,attr_name="parent")
    data=[(ind_attr["position"][0],ind_attr["position"][1],index,len(parent.children)) for ind_attr,index,parent in zip(ind_attrs,indexs,parents)]
    fitness=[fitness_func([chair,parent])>0 for chair,parent in zip(chairs,parents)]
    return data,fitness
def fitness_extraction_dist(samples):
    return fitness.dist_between_parent_child(utility.extract_samples_attributes(samples,sample_name="table")[0],utility.extract_samples_attributes(samples,sample_name="chair"))
def fitness_extraction_overl(samples):
    polygons=mapping.map_layoutsamples_to_geometricobjects(samples)
    polygons_vis.append(polygons[0])
    return fitness.surface_overlap(polygons)
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

plt.show()
accept = lambda f, value: f < value
subset,fitness_subset = zip(*[(d,f) for d,f in zip(data,fitness_values)])

x,y,index,parent_children=zip(*subset)

#print(full_subset[0])
# Fit a mixture of Gaussians with EM using five components
gmm = mixture.GMM(n_components=4, covariance_type='full',min_covar=0.0000001)

norm_f_subs=(fitness_subset-np.min(fitness_subset))/(np.max(fitness_subset)-np.min(fitness_subset))

def normalised_fitness(fitness):
    return fitness/np.sum(fitness)

ax=visualisation.init()
table=mapping.map_layoutsamples_to_geometricobjects(utility.extract_samples_attributes(samples,sample_name="table"))[0]
polygons_vis.append(table)
visualisation.draw_polygons(ax,polygons_vis)





#plot the points
position=np.array(table.centroid)

conditional=False
weighted=False
if conditional:
    training = np.column_stack((x,y,fitness_subset))
    gmm.fit(training)
    y_0=15
    (con_cen, con_cov, new_p_k) = learning_utility.cond_dist(np.array([np.nan,np.nan, 0.001]), gmm.means_, gmm._get_covars(), gmm.weights_)
    samples=learning_utility.sample_gaussian_mixture(con_cen, con_cov, new_p_k,samples=100)
    x_new,y_new=zip(*samples)
elif weighted:
    wgmm = learning_utility.GMM_weighted(n_components=100, covariance_type='full',min_covar=0.0000001)
    training = np.column_stack((x,y))
    fitness_values=1-np.array(exposure.equalize_hist(fitness_values))
    plt.hist(fitness_values)
    wgmm.weighted_fit(training,fitness_values)
    #visualisation.make_ellipses(wgmm.means_,wgmm.covars_,plt.gcf())
    samples=learning_utility.sample_gaussian_mixture(wgmm.means_,wgmm._get_covars(),wgmm.weights_,samples=100)
else:
    training = np.column_stack((x,y))
    gmm.fit(training)
    samples=gmm.sample(100)
x_new,y_new=zip(*samples)
x_new=np.array(x_new)+position[0]
y_new=np.array(y_new)+position[1]
plt.scatter(x_new,y_new,color='r')

x=np.array(x)+position[0]
y=np.array(y)+position[1]

plt.scatter(x,y,c=fitness_values,cmap=cm.Blues)
print(x,y)
print(([np.array(p.centroid) for p in polygons_vis ]))
plt.colorbar()
(xrange,yrange)=((min(x),max(x)),(min(y),max(y)))
ax.set_xlim(*xrange)
ax.set_ylim(*yrange)
print(gmm.converged_)


plt.show()