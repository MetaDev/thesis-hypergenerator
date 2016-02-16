# -*- coding: utf-8 -*-
import networkx as nx

from descartes.patch import PolygonPatch

from matplotlib import pyplot as plt

import utility

def init(name="visualisation"):
    if(plt.gcf()==0):
        fig = plt.figure(name)
    fig=plt.gcf()
    fig.clear()
    ax = fig.add_subplot(111)
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
def draw_polygons(ax,polygons,colors=[],size=1.2,show_edges=False):
    color="blue"
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
    (xrange,yrange)=utility.range_from_polygons(polygons, size)
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    #aspect ratio of plot
    ax.set_aspect(1)
  