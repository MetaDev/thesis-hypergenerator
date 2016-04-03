# -*- coding: utf-8 -*-
import networkx as nx

from descartes.patch import PolygonPatch

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

from scipy.stats import binned_statistic

import scipy.stats as ss
import model.mapping as mp


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
def draw_node_sample_tree(root,ax=None):
    samples=root.get_flat_list()
    polygons = mp.map_layoutsamples_to_geometricobjects(samples,"shape")
    colors = [s.values["rel"]["color"] for s in samples]
    draw_polygons(polygons=polygons,ax=ax,colors=colors,size=1.3,set_range=True)

def get_ax(ax):
    if ax is None:
        return plt.gca()
    else:
        return ax
def draw_polygons(polygons,ax=None,colors=[],size=1.2,show_edges=False,set_range=False):
    ax=get_ax(ax)
    ax.set_aspect(1)
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

#TODO check dimensionality, move to gmm (where a dimension can be chosen)
def visualise_gmm_marg_1D_density(ax,marg_index,gmm_weights,gmm_means,gmm_cov,
                                 factor=2,xrange=None,verbose=False):
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
def visualise_gmm_marg_density_1D_GMM(ax,marg_index,gmm,factor=3,verbose=False):
    visualise_gmm_marg_1D_density(ax,marg_index,*gmm.get_params(),
                                 factor,verbose)
#to visualise the density of 2 dimensional data
#respect covariance between the data

#TODO check dimensionality
def visualise_gmm_marg_2D_density(ax,gmm,
                                  min_factor=.5,max_factor=3,steps=5, colors=["r","g","b"]):
    from matplotlib.patches import Ellipse
    from itertools import cycle
    if colors is not None:
        colors = cycle(colors)
    min_alpha=0.03
    max_alpha=0.4

    for factor in np.linspace(min_factor, max_factor, steps):
        for (mean, (angle, width, height)),weight in zip(gmm.gmm_gmr.to_ellipses(factor),gmm.weights_):
            ell = Ellipse(xy=mean, width=width, height=height,
                          angle=np.degrees(angle))
            ell.set_alpha(min_alpha+(max_alpha-min_alpha)*weight)
            if colors is not None:
                ell.set_color(next(colors))
            ax.add_artist(ell)
def print_fitness_statistics(fitness_values,print_hist=False):
    fitness_statistics_count, fitness_bin_edges,_ = binned_statistic(fitness_values, fitness_values,
                                                                     statistic='count', bins=20)
    if print_hist:
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