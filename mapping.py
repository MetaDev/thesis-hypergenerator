from shapely.geometry import *

import networkx as nx
from shapely import affinity
import numpy as np
import utility
import search_space

def map_polygons_to_neighbourhoud_graph(polygons,grid_range, step):
    """
    create neighbourhoud graph for points in a grid with a set of polygons as obstacles
    """
    #create simple grid graph
    G = utility.grid_2d_graph(grid_range,step)
    #remove unaccesible nodes due to edges intersecting with obstacles
    for edge in G.edges():
        free_edge=True
        for p in polygons:
            if p.intersects(LineString(edge)):
                    free_edge=False
        if not free_edge:
            G.remove_edge(edge[0],edge[1])
            if not next(nx.all_neighbors(G,edge[0]),None):
                G.remove_node(edge[0])
            if not next(nx.all_neighbors(G,edge[1]),None):
                G.remove_node(edge[1])
    return G
#geom_objs is either polygon or line string
def map_geometricobjects_to_nodes(graph,geom_objs):
    nodes=[]
    for o in geom_objs:
        dist=float("inf")
        node=None
        for n in graph.nodes():
            if o.distance(Point(n))<dist:
                node=n
                dist=o.distance(Point(n))
        nodes.append(node)
    return nodes
#map a sample to a shape of which it's center is at the position of the sample
def map_layoutsamples_to_geometricobjects(layout_samples,geom_class=Polygon):
    geom_objs=[]
    for sample in layout_samples:
        geom_obj=geom_class(sample.shape)
        centroid=np.array(geom_obj.centroid)
        geom_obj=affinity.translate(geom_obj, -centroid[0],-centroid[1],0)
        geom_obj=affinity.translate(geom_obj, sample.position[0],sample.position[1],0)
        geom_obj=affinity.scale(geom_obj,sample.size[0],sample.size[1],origin="centroid")       
        geom_obj=affinity.rotate(geom_obj,sample.rotation,origin="centroid")
        
        
        
        geom_objs.append(geom_obj)
    return geom_objs



