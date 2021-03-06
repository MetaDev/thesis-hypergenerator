from shapely.geometry import Polygon,LineString,Point

import networkx as nx
from shapely import affinity
import util.utility as ut


def map_polygons_to_neighbourhoud_graph(polygons,grid_range, step):
    """
    create neighbourhoud graph for points in a grid with a set of polygons as obstacles
    """
    #create simple grid graph
    G = ut.grid_2d_graph(grid_range,step)
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
def map_to_polygon(shape,origin,position,rotation,size):
    geom_obj=Polygon(shape)
    geom_obj=affinity.translate(geom_obj, -origin[0],-origin[1],0)
    geom_obj=affinity.scale(geom_obj,size[0],size[1],origin=(0,0))
    geom_obj=affinity.rotate(geom_obj,int(rotation*360),origin=(0,0))
    geom_obj=affinity.translate(geom_obj, position[0],position[1],0)
    return geom_obj
#map a sample to a shape of which it's center is at the position of the sample
def map_layoutsamples_to_geometricobjects(layout_samples,shape_name):
    geom_objs=[]
    for sample in layout_samples:
        geom_objs.append(map_layoutsample_to_geometricobject(sample,shape_name))
    return geom_objs

import model
def map_layoutsample_to_geometricobject(layout_sample,shape_name):
    rel_vars=layout_sample.values["rel"]
    #extract points from vars and put in list ordered on their index saved in the name
    shape=model.search_space.VectorVariableUtility.extract_ordered_list_vars_values(shape_name,rel_vars)
    polygon = map_to_polygon(shape,rel_vars["origin"],rel_vars["position"],
                                                             rel_vars["rotation"]
                                                             ,rel_vars["size"])
    return polygon



