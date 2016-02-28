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
from operator import add

ndata=100

data=[]
fitness_values=[]
polygons_vis=[]

def feauture_fitness_extraction(samples,fitness_func):
    chairs=utility.extract_samples_attributes(samples,sample_name="chair")
    ind_attrs = utility.extract_samples_attributes(chairs,attr_name="independent_attr")
    indexs = utility.extract_samples_attributes(chairs,attr_name="index")
    parents = utility.extract_samples_attributes(chairs,attr_name="parent")
    data=[(ind_attr["position"][0],ind_attr["position"][1],parent.shape[3][0],parent.shape[3][1]) for ind_attr,index,parent in zip(ind_attrs,indexs,parents)]
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


x,y,x_shape,y_shape=zip(*data)

n_components=10
n_samples=100
conditional=True
weighted=True


if weighted:
    gmm = learning_utility.GMM_weighted(n_components=n_components, covariance_type='full',min_covar=0.000001)
    #give heavy penalty for intersection
    #inverse and normalise fitness
    fitness_values=(fitness.invert_fitness(fitness_values))**4
    gmm.weighted_fit(data,fitness_values)
    position_samples=learning_utility.sample_gaussian_mixture(gmm.means_,gmm._get_covars(),gmm.weights_,samples=n_samples,loose_norm_weights=True)
else:
    gmm = mixture.GMM(n_components=n_components, covariance_type='full',min_covar=0.0000001)
    gmm.fit(data)
    position_samples=gmm.sample(n_samples)

#visualise
if(plt.gcf()==0):
    fig = plt.figure("name")
fig=plt.gcf()
fig.clear()



if conditional:
    position=(1,2)
    print
    x= np.array(x)+position[0]
    y=np.array(y)+position[1]
    n=1;
    values=np.arange(0.5,2,0.25)
    print(values)
    for p in values:
        shape=[(0, 0), (0, 1),(0.5,1),(p,p),(1, 0.5),(1,0)]
        
        polygon = mapping.map_to_polygon(shape,position,0,(1,1))
        (con_cen, con_cov, new_p_k) = learning_utility.cond_dist(np.array([np.nan,np.nan,p,p]), gmm.means_, gmm._get_covars(), gmm.weights_)
        
        position_samples=learning_utility.sample_gaussian_mixture(con_cen, con_cov, new_p_k,samples=n_samples)
        ax = plt.gca()
        ax.set_aspect(1)
        
        x_new,y_new=zip(*position_samples)
        x_new=np.array(x_new)+position[0]
        y_new=np.array(y_new)+position[1]
        
        (xrange,yrange)=((min(x_new),max(x_new)),(min(y_new),max(y_new)))
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        plt.scatter(x,y,c=fitness_values,cmap=cm.Blues)
        plt.colorbar()
        visualisation.draw_polygons(ax,[polygon])
        #TODO move centers with position
        con_cen=[list(map(add, c, position))for c in con_cen]
        visualisation.make_ellipses(con_cen,con_cov,ax)
        plt.scatter(x_new,y_new,color='r')
        n+=1
        plt.show()

