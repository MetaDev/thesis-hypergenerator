# -*- coding: utf-8 -*-
import networkx as nx

from descartes.patch import PolygonPatch

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

from scipy.stats import binned_statistic

import scipy.stats as ss
import model.mapping as mp
import matplotlib.patches as patches


from gmr import MVN,GMM
import util.utility as ut
import util.data_format as dtfr

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
def draw_node_sample_tree(root,color,ax=None):
    samples=root.get_flat_list()
    polygons = mp.map_layoutsamples_to_geometricobjects(samples,"shape")
    draw_polygons(polygons=polygons,ax=ax,color=color,size=1.3,set_range=True)

def get_new_ax():
    fig=plt.figure()
    return fig,fig.add_subplot(111)
#return current or new ax
def get_ax(ax,option="current"):
    if ax is None:
        if option is "current":
            return plt.gca()
        elif option is "new":
            return plt.figure().add_subplot(111)
        else:
            raise ValueError("wrong option for get ax function.")

    else:
        return ax
def draw_polygons(polygons,ax=None,color="b",size=1.2,show_edges=False,set_range=False):
    ax=get_ax(ax)
    ax.set_aspect(1)
    for i in range(len(polygons)):
        polygon=polygons[i]

        #color=
        x, y = polygon.exterior.xy
        if show_edges:
            #plot edges of polygon
            ax.plot(x, y, 'o', color='#999999', zorder=1)
        #plot surface
        patch = PolygonPatch(polygon, facecolor=color, edgecolor=color, alpha=0.5, zorder=1)
        ax.add_patch(patch)
    #finalise figure properties
    ax.set_title('Polygon representation')
    if set_range:
        (xrange,yrange)=ut.range_from_polygons(polygons, size)
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)

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
def draw_1D_stoch_variable(var,ax):
    ax.plot((var.low, var.low), (0, 1), 'r-')
    ax.plot((var.high, var.high), (0, 1), 'r-')
    width=var.high-var.low
    ax.set_xlim([var.low- (width*0.1), var.high+(width*0.1)])

def draw_2D_stoch_variable(var,ax):
    rect=patches.Rectangle(
                (var.low[0],var.low[1]),   # (x,y)
                var.high[0]-var.low[0],          # width
                var.high[1]-var.low[1],          # height
             fill=False)
    rect.set_color("r")
    ax.add_patch(rect)
    width=var.high[0]-var.low[0]
    height=var.high[1]-var.low[1]
    ax.set_xlim([var.low[0]- (width*0.1), var.high[0] + (width*0.1)])
    ax.set_ylim([var.low[1]- (height*0.1), var.high[1] + (height*0.1)])
from textwrap import wrap

def draw_1D_2D_GMM_variable_sampling(gmmvar,title,ax_title,ax=None):
    #marginalise variables from gmm by name

    #gives indices (tuples) starting from 0
    indices = dtfr.variables_indices(gmmvar.sibling_vars)
    #trasnform to indices ending at gmm_dimension
    gmm_dim=gmmvar.get_dim()
    ind_diff= (gmm_dim)-indices[-1][1]
    indices = [(i[0]+ind_diff,i[1]+ind_diff) for i in indices]
    vis_vars= gmmvar.sibling_vars
    title='\n'.join(wrap(title,60))
    for (index0,index1),var in zip(indices,vis_vars):
        fig,ax=get_new_ax()
        fig.suptitle(title)

        new_ax_title=ax_title+", variable name: " + var.name + " sibling order: "+ str(gmmvar.sibling_order)
        new_ax_title='\n'.join(wrap(new_ax_title,60))
        ax.set_title(new_ax_title)
        if index1 - index0 == 1:
            #visualise the sibling var range
            draw_1D_stoch_variable(var,ax)
            visualise_gmm_marg_1D_density(gmmvar.gmm,index0,ax=ax)
        elif index1-index0 == 2:
            #visualise the sibling var range
            draw_2D_stoch_variable(var,ax)
            visualise_gmm_marg_2D_density(gmmvar.gmm,np.arange(index0,index1),ax=get_ax(ax,"new"))
        else:
            raise ValueError("Only variables up to 2 dimensions can be visualised. Variable ",
                                                                     var.name,"has indices: ",index0,index1 )
#        fig.subplots_adjust(top=0.85)
        fig.tight_layout( rect=[0, 0.05, 1, 0.9])




#TODO check dimensionality
def visualise_gmm_marg_1D_density(gmm,marg_index,name="",ax=None,
                                 factor=2,verbose=False):

    gmm_weights,gmm_means,gmm_cov=gmm.get_params()
    means=[]
    weights=gmm_weights
    stdevs=[]
    ax=get_ax(ax)
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
        print("Expected mean value: "+ ut.format_float(np.mean([ w*m for m,w in zip(means, weights)])))
        print("Expected mean standard dev: " + ut.format_float(np.mean([ w*sd for sd,w in zip(stdevs, weights)])))
    ax.plot(x, density,color="g")

#plot marginal density of a single dimension in a GMM
#this is can be used as indicator for expressivens of this dimension
#it's limitation is that it assumes independence between vars but it is very hard to visualise multi dimensional data

#TODO check dimensionality
def visualise_gmm_marg_2D_density(gmm,marg_index,ax=None,
                                  min_factor=.5,max_factor=3,steps=5, colors=["g"]):
    ax=get_ax(ax)
    ax.set_aspect(1)
    from matplotlib.patches import Ellipse
    from itertools import cycle
    if colors is not None:
        colors = cycle(colors)
    min_alpha=0.03
    max_alpha=0.4
    ax_range=[]
    gmm=gmm.marginalise(marg_index)
    for factor in np.linspace(min_factor, max_factor, steps):
        for (mean, (angle, width, height)),weight in zip(gmm.gmm_gmr.to_ellipses(factor),gmm.weights_):
            ell = Ellipse(xy=mean, width=width, height=height,
                          angle=np.degrees(angle))
            max_size=max(width,height)
            ax_range.append(((mean[0]-max_size,mean[0]+max_size),(mean[1]-max_size,mean[1]+max_size)))
            ell.set_alpha(min_alpha+(max_alpha-min_alpha)*weight)
            if colors is not None:
                ell.set_color(next(colors))
            ax.add_artist(ell)
