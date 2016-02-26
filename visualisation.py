# -*- coding: utf-8 -*-
import networkx as nx

from descartes.patch import PolygonPatch

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np


import utility

def init(name="visualisation"):
    if(plt.gcf()==0):
        fig = plt.figure(name)
    fig=plt.gcf()
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    return ax
def draw_graph(ax,graph,with_labels=False,node_size=0):
    pos=dict([ (n, n) for n in graph.nodes() ])
    labels =dict([ (n,str(n)) for n in graph.nodes() ])
    nx.draw(graph, pos=pos,with_labels=with_labels,ax=ax,labels=labels,node_size=node_size)
def draw_graph_path(ax,graph,path,color='b',with_labels=False,node_size=50,node_shape="p"):
    pos=dict([ (n, n) for n in graph.nodes() ])
    nx.draw_networkx_nodes(graph, nodelist=path,node_color=color,node_shape=node_shape,pos=pos,with_labels=with_labels,ax=ax,node_size=node_size)
def finish():
    #show plot
    plt.show()
def draw_polygons(ax,polygons,colors=[],size=1.2,show_edges=False,set_range=True):
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
        (xrange,yrange)=utility.range_from_polygons(polygons, size)
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
   
def make_ellipses(gmm_means,gmm_covars, fig):
    for n in range(len(gmm_covars)):
        v, w = np.linalg.eigh(gmm_covars[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm_means[n, :2], v[0], v[1],
                                  180 + angle, color='#999999')
        ell.set_clip_box(fig.gca().bbox)
        ell.set_alpha(0.5)
        fig.gca().add_artist(ell)
  