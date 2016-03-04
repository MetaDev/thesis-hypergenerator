# -*- coding: utf-8 -*-
#polygon imports

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


from sklearn import mixture

from util import setting_values
import matplotlib.cm as cm
import model.test_models as tm

from operator import add

from gmr import MVN,GMM, plot_error_ellipses
ndata=200

data=[]
fitness_values=[]
polygons_vis=[]

def feauture_fitness_extraction(samples,fitness_func):
    chairs=ut.extract_samples_vars(samples,sample_name="child")
    ind_attrs = ut.extract_samples_vars(chairs,var_name="independent_attr")
    indexs = ut.extract_samples_vars(chairs,attr_name="index")
    parents = ut.extract_samples_attributes(chairs,attr_name="parent")
    data=[(ind_attr["position"][0],ind_attr["position"][1],parent.shape[3][0],parent.shape[3][1]) for ind_attr,index,parent in zip(ind_attrs,indexs,parents)]
    fitness=[fitness_func([chair,parent]) for chair,parent in zip(chairs,parents)]
    return data,fitness
def fitness_extraction_dist(samples):
    return fn.dist_between_parent_child(ut.extract_samples_attributes(samples,sample_name="parent")[0],ut.extract_samples_attributes(samples,sample_name="chair"))
def fitness_extraction_overl(samples):
    polygons=mp.map_layoutsamples_to_geometricobjects(samples)
    return fn.pairwise_overlap(polygons,normalized=True)


for i in range(ndata):
    #train per child
    samples,_=tm.test_samples_var_child_position_parent_shape()
    sample_features,sample_fitness=feauture_fitness_extraction(samples,fitness_extraction_overl)
    data.extend(sample_features)
    fitness_values.extend(sample_fitness)




filtered_data=False
if filtered_data:
    cut_off_value=0
    accept = lambda f, value: f < value
    data,fitness_values = zip(*[(d,f) for d,f in zip(data,fitness_values) if accept(f,cut_off_value)])


x,y,x_shape,y_shape=zip(*data)

n_components=10
n_samples=100
conditional=True



#better statistics of fittnes
vis.print_fitness_statistics(fitness_values)

#both full and tied covariance work
#but tied prevents overfitting but is worse to find conditional estimatation of sharp corners
gmm = lut.GMM_weighted(n_components=n_components, covariance_type='full',min_covar=0.0000001,random_state=setting_values.random_state)
#give heavy penalty for intersection
#inverse and normalise fitness
fitness_values=(fn.invert_fitness(fitness_values))**8
gmm.weighted_fit(data,fitness_values)
#convert grm GMM
gmm = GMM(n_components=len(gmm.weights_), priors=gmm.weights_, means=gmm.means_, covariances=gmm._get_covars(), random_state=setting_values.random_state)
position_samples=gmm.sample(n_samples=n_samples)

#visualise
if(plt.gcf()==0):
    fig = plt.figure("name")
fig=plt.gcf()
fig.clear()


ax = plt.gca()
vis.visualise_gmm_marg_density_1D_gmr(ax,0,gmm)

plt.show()
if conditional:
    position=(1,2)
    x= np.array(x)+position[0]
    y=np.array(y)+position[1]
    n=1;
    values=np.arange(0.5,2,0.25)
    for p in values:
        shape=[(0, 0), (0, 1),(0.5,1),(p,p),(1, 0.5),(1,0)]
        #the original gmm needs to have u full covariance matrix, to estimate conditional matrix
        polygon = mp.map_to_polygon(shape,[0.5,0.5],position,0,(1,1))
        gmm_cond=gmm.condition(np.array([2,3]),np.array([p,p]))
        position_samples=gmm_cond.sample(n_samples)
        ax = plt.gca()
        ax.set_aspect(1)

        gmm_cond.n_components
        x_new,y_new=zip(*position_samples)
        x_new=np.array(x_new)+position[0]
        y_new=np.array(y_new)+position[1]

        (xrange,yrange)=((min(x),max(x)),(min(y),max(y)))
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        plt.scatter(x,y,c=fitness_values,cmap=cm.Blues)
        plt.colorbar()
        vis.draw_polygons(ax,[polygon])
        #still need to deterministically determine position
        gmm_cond.means=[list(map(add, c, position))for c in gmm_cond.means]
        #plot marginal to express expressiveness
        vis.visualise_gmm_marg_2D_density_gmr(ax,gmm_cond,colors=["g"])
        plt.scatter(x_new,y_new,color='r')
        n+=1
        plt.show()

