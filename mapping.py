from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.geometry import Point

import networkx as nx
from shapely import affinity
import numpy
import utility

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

def map_polygons_to_nodes(graph,polygons):
    nodes=[]
    for p in polygons:
        dist=float("inf")
        node=None
        for n in graph.nodes():
            if p.distance(Point(n))<dist:
                node=n
                dist=p.distance(Point(n))
        nodes.append(node)
    return nodes
def map_samples_to_polygons(layout_samples):
    polygons=[]
    colors=[]
    for sample in layout_samples:
        polygon=Polygon(sample.shape)
        polygon=affinity.rotate(polygon,sample.rotation)
        polygon=affinity.translate(polygon, sample.position_x,sample.position_y,0)
        polygon=affinity.scale(polygon,sample.size,sample.size)
        polygons.append(polygon)
        colors.append(sample.color)
    return polygons,colors


