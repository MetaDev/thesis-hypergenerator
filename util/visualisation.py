# -*- coding: utf-8 -*-
import networkx as nx

from descartes.patch import PolygonPatch

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

from scipy.stats import binned_statistic

import scipy.stats as ss


from gmr import MVN,GMM
import util.utility as ut

def init(name="visualisation",n_plots=1):
    if(plt.gcf()==0):
        fig = plt.figure(name)
    fig=plt.gcf()
    fig.clear()

def draw_graph(ax,graph,with_labels=False,node_size=0):
    pos=dict([ (n, n) for n in graph.nodes() ])
    labels =dict([ (n,str(n)) for n in graph.nodes() ])
    nx.draw(graph, pos=pos,with_labels=with_labels,ax=ax,labels=labels,node_size=node_size)
def draw_graph_path(ax,graph,path,color='b',with_labels=False,node_size=50,node_shape="p"):
    pos=dict([ (n, n) for n in graph.nodes() ])
    nx.draw_networkx_nodes(graph, nodelist=path,node_color=color,node_shape=node_shape,
                           pos=pos,with_labels=with_labels,ax=ax,node_size=node_size)

def draw_polygons(ax,polygons,colors=[],size=1.2,show_edges=False,set_range=False):
    color="b"
    for i in range(len(polygons)):
        polygon=polygons[i]
        if colors:
            color=colors[i]
        #color=
        x, y = polygon.exterior.xy
        if show_edges:
            #plot edges of polygon
            ax.plot(x, y, 'o', color='#999999', zorder=1)
        #plot surface
        patch = PolygonPatch(polygon, facecolor=color, edgecolor=color, alpha=0.5, zorder=1)
        ax.add_patch(patch)
    #finalise figure properties
    ax.set_title('Layout visualisation')
    if set_range:
        (xrange,yrange)=ut.range_from_polygons(polygons, size)
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
#plot marginal density of a single dimension in a GMM
#this is can be used as indicator for expressivens of this dimension
#it's limitation is that it assumes independence between vars but it is very hard to visualise multi dimensional data

#TODO line for not learned expressiveness should reflect the original distribution

def visualise_gmm_marg_1D_density(ax,marg_index,gmm_means,gmm_cov,gmm_weights,
                                  random_state=None,factor=2,xrange=None,verbose=False):
    means=[]
    weights=gmm_weights
    stdevs=[]
    for k in range(len(gmm_means)):
        means.append(gmm_means[k][marg_index])
        #calc std
        cov = np.asanyarray(gmm_cov[k])
        stdevs.append(np.sqrt(np.diag(cov))[marg_index])

    min_x=np.min([mean-factor*stdev for mean,stdev in zip(means,stdevs)])
    max_x=np.max([mean+factor*stdev for mean,stdev in zip(means,stdevs)])
    x = np.arange(min_x, max_x, 0.01)
    pdfs = [w * ss.norm.pdf(x, mu, sd) for mu, sd, w in zip(means, stdevs, weights)]
    density = np.sum(np.array(pdfs), axis=0)
    if verbose:
        print("Expected mean value: "+ format_float(np.mean([ w*m for m,w in zip(means, weights)])))
        print("Expected mean standard dev: " + format_float(np.mean([ w*sd for sd,w in zip(stdevs, weights)])))
    #unfirom distribution line
    y=[1/(max_x-min_x)]*len(x)
    ax.plot(x, density)
    ax.plot(x,y,color="r")
#same as above but for GMM from the gmr package
def visualise_gmm_marg_density_1D_gmr(ax,marg_index,gmr_gmm,factor=3,verbose=False):
    visualise_gmm_marg_1D_density(ax,marg_index,gmr_gmm.means,gmr_gmm.covariances,gmr_gmm.priors,
                                  gmr_gmm.random_state,factor,verbose)
#to visualise the density of 2 dimensional data
#respect covariance between the data
def visualise_gmm_marg_2D_density_gmr(ax,gmr_gmm,min_factor=.5,max_factor=3,steps=5,
                                      colors=["r,g,b"]):
    visualise_gmm_marg_2D_density(ax, gmr_gmm.means,gmr_gmm.covariances,gmr_gmm.priors,
                                  gmr_gmm.random_state,min_factor,max_factor,steps,
                                  colors=colors)
#TODO also visualise weight of component (using the alpha)
def visualise_gmm_marg_2D_density(ax,gmm_means,gmm_cov,gmm_weights,random_state=None,
                                  min_factor=.5,max_factor=3,steps=5, colors=["r,g,b"]):
    gmm=GMM(len(gmm_weights),gmm_weights,gmm_means,gmm_cov,random_state)
    from matplotlib.patches import Ellipse
    from itertools import cycle
    if colors is not None:
        colors = cycle(colors)
    min_alpha=0.25
    max_alpha=0.75

    for factor in np.linspace(min_factor, max_factor, steps):
        for (mean, (angle, width, height)),weight in zip(gmm.to_ellipses(factor),gmm_weights):
            ell = Ellipse(xy=mean, width=width, height=height,
                          angle=np.degrees(angle))
            ell.set_alpha((max_alpha-min_alpha)*weight)
            if colors is not None:
                ell.set_color(next(colors))
            ax.add_artist(ell)
def print_fitness_statistics(fitness_values):
    fitness_statistics_count, fitness_bin_edges,_ = binned_statistic(fitness_values, fitness_values,
                                                                     statistic='count', bins=20)
    fitness_statistics_mean = binned_statistic(fitness_values, fitness_values,  statistic='mean'
    , bins=20)[0]
    for edges,count,mean,i in zip(ut.pairwise(fitness_bin_edges),fitness_statistics_count,
        fitness_statistics_mean,range(len(fitness_statistics_count))):

        edges ="bin nr. "+ str(i) + ": "+format(edges[0], '.5g') + " - "+format(edges[1], '.5g')
        count = ": count= " + str(count)
        mean=": mean= "+ format(mean, '.5g')
        print(edges + count + mean)

    print("total mean= {0:f}: variance= {1:f}".format(np.mean(fitness_values),np.var(fitness_values),'.5g'))

def format_float(f):
    return format(f, '.5g')