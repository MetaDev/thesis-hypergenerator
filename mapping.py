from shapely.geometry import LineString
import itertools
import networkx as nx
import numpy as np
import shapely

def convert_to_neighbourhoud_graph(polygons,grid_range, step):
    """
    create neighbourhoud graph for points in a grid with a set of polygons as obstacles
    """
    #create simple grid graph
    G = grid_2d_graph(grid_range,step)
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



def grid_2d_graph(grid_range,step):
    G=nx.Graph()
    rows=np.linspace(grid_range[0][0],grid_range[0][1],step[0])
    columns=np.linspace(grid_range[1][0],grid_range[1][1],step[1])
    m = np.array([(i,j) for i in rows for j in columns])
    m=m.reshape(step[0],step[1],2)
    print(m)
    for i in range(step[0]):
        for j in range(step[1]):
            G.add_node(tuple(m[i][j]))
            if i>0:
                G.add_edge(tuple(m[i][j]),tuple(m[i-1][j]))
            if j>0:
                G.add_edge(tuple(m[i][j]),tuple(m[i][j-1]))
    return G
def draw_graph(coord_graph):
    pos=dict([ (n, n) for n in coord_graph.nodes() ])
    labels =dict([ (n,str(n)) for n in coord_graph.nodes() ])
    print(labels)
    nx.draw(coord_graph, pos=pos,with_labels=True,labels=labels,node_size=30)

