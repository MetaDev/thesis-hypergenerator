# -*- coding: utf-8 -*-

import networkx as nx

from descartes.patch import PolygonPatch
import matplotlib.patches as patches

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

from scipy.stats import binned_statistic

import scipy.stats as ss
import model.mapping as mp


from gmr import MVN,GMM
import util.data_format as dtfr

import util.utility as ut

def init(name="visualisation",n_plots=1):
    if(plt.gcf()==0):
        fig = plt.figure(name)
    fig=plt.gcf()
    fig.clear()


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


def draw_graph(ax,graph,with_labels=False,node_size=0):
    pos=dict([ (n, n) for n in graph.nodes() ])
    labels =dict([ (n,str(n)) for n in graph.nodes() ])
    nx.draw(graph, pos=pos,with_labels=with_labels,ax=ax,labels=labels,node_size=node_size)
def draw_graph_path(ax,graph,path,color='b',with_labels=False,node_size=50,node_shape="p"):
    pos=dict([ (n, n) for n in graph.nodes() ])
    nx.draw_networkx_nodes(graph, nodelist=path,node_color=color,node_shape=node_shape,
                           pos=pos,with_labels=with_labels,ax=ax,node_size=node_size)




def draw_polygons(polygons,ax=None,color="b",show_edges=False):
    ax=get_ax(ax)
    ax.set_aspect(1)
    for i in range(len(polygons)):
        polygon=polygons[i]

        x, y = polygon.exterior.xy
        if show_edges:
            #plot edges of polygon
            ax.plot(x, y, 'o', color='#999999', zorder=1)
        #plot surface
        patch = PolygonPatch(polygon, facecolor=color, edgecolor=color, alpha=0.5, zorder=1)
        ax.add_patch(patch)
    #finalise figure properties
    ax.set_title('Layout visualisation')

def set_poygon_range(polygon_list,size=1.2,ax=None):
    ax=get_ax(ax)
    (xrange,yrange)=ut.range_from_polygons(polygon_list, size)
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)

def draw_1D_stoch_variable(var,ax):
    ax.plot((var.low, var.low), (0, 1), 'r-')
    ax.plot((var.high, var.high), (0, 1), 'r-')

def draw_2D_stoch_variable(var,ax):
    rect=patches.Rectangle(
                (var.low[0],var.low[1]),   # (x,y)
                var.high[0]-var.low[0],          # width
                var.high[1]-var.low[1],          # height
             fill=False)
    rect.set_color("r")
    ax.add_patch(rect)

import model
def draw_1D_2D_GMM_variable_sampling(gmmvar,gmm_cond,parent_sample,sibling_samples,ax=None):
    #marginalise variables from gmm by name
    indices=[index for index,name in zip(dtfr.variables_indices(gmmvar.sibling_vars),
                                       gmmvar.unpack_names) if name in gmmvar.visualise_variable_names]
    vis_vars=[var for var in gmmvar.sibling_vars if var.name in gmmvar.visualise_variable_names ]
    for (index0,index1),var in zip(indices,vis_vars):
        ax=get_ax(ax,"new")
        if index1 - index0 == 1:
            #visualise the sibling var range
            draw_1D_stoch_variable(var,ax)
            visualise_gmm_marg_1D_density(gmm_cond,index0,ax=ax)
        elif index1-index0 == 2:
            #visualise the sibling var range
            draw_2D_stoch_variable(var,ax)
            visualise_gmm_marg_2D_density(gmm_cond,np.arange(index0,index1),ax=get_ax(ax,"new"))
        else:
            raise ValueError("Only variables up to 2 dimensions can be visualised. Variable ",
                                                                     var.name,"has indices: ",index0,index1 )
    #visualise the model
    color_list=[(parent_sample.name,"b")]
    if len(sibling_samples)>0:
        color_list.append((sibling_samples[0].name,"r"))
    model.search_space.LayoutTreeDefNode.visualise(parent_sample,color_list,sibling_samples,ax=get_ax(None,"new"))




#TODO check dimensionality
def visualise_gmm_marg_1D_density(gmm,marg_index,name="",ax=None,
                                 factor=2,verbose=False):

    gmm_weights,gmm_means,gmm_cov=gmm.get_params()
    means=[]
    weights=gmm_weights
    stdevs=[]
    ax=get_ax(ax)
    ax.set_aspect(1)
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
    #unfirom distribution line
    y=[1/(max_x-min_x)]*len(x)
    ax.plot(x, density)
    ax.plot(x,y,color="r")

#plot marginal density of a single dimension in a GMM
#this is can be used as indicator for expressivens of this dimension
#it's limitation is that it assumes independence between vars but it is very hard to visualise multi dimensional data

#TODO check dimensionality
def visualise_gmm_marg_2D_density(gmm,marg_index,ax=None,
                                  min_factor=.5,max_factor=3,steps=5, colors=["r"]):
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
    ax.set_xlim(min(np.array(ax_range)[:,0,0]),max(np.array(ax_range)[:,0,1]))
    ax.set_ylim(min(np.array(ax_range)[:,1,0]),max(np.array(ax_range)[:,1,1]))