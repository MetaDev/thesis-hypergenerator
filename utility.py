# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import shapely
import itertools as it
import operator as op
from shapely.geometry import Point

def range_from_polygons(polygons, size=1):
    allx=[]
    ally=[]
    for polygon in polygons:
        x, y = polygon.exterior.xy
        #find range of polygon coordinates
        allx.append(x)
        ally.append(y)
    xrange = [int(np.floor(np.min(allx))),int(np.ceil(np.max(allx)))]
    yrange = [int(np.floor(np.min(ally))),int(np.ceil(np.max(ally)))]
    xextrarange=((xrange[1]-xrange[0])/2)*(size-1)
    yextrarange=((yrange[1]-yrange[0])/2)*(size-1)
    xrange=(xrange[0]-xextrarange,xrange[1]+xextrarange)
    yrange=(yrange[0]-yextrarange,yrange[1]+yextrarange)
    return (xrange,yrange)
def range_offset_from_steps(xrange,yrange,steps,wideness=2):
    xdelta=((xrange[1]-xrange[0])/steps[0])*wideness
    ydelta=(yrange[1]-yrange[0])/steps[1]
    return (xdelta,ydelta)
def grid_2d_graph(grid_range,step):
    G=nx.Graph()
    rows=np.linspace(grid_range[0][0],grid_range[0][1],step[0])
    columns=np.linspace(grid_range[1][0],grid_range[1][1],step[1])
    m = np.array([(i,j) for i in rows for j in columns])
    m=m.reshape(step[0],step[1],2)
    for i in range(step[0]):
        for j in range(step[1]):
            G.add_node(tuple(m[i][j]))
            if i>0:
                G.add_edge(tuple(m[i][j]),tuple(m[i-1][j]))
            if j>0:
                G.add_edge(tuple(m[i][j]),tuple(m[i][j-1]))
    return G
def nr_of_steps_from_range(xrange,yrange,step_size=1):
    return (int((xrange[1]-xrange[0]+1)/step_size),int((yrange[1]-yrange[0]+1)/step_size))
def closest_point_to_polygon(polygon, points):
    dist=float("inf")
    point=None
    for p in points:
        if Point(p).distance(polygon)<dist:
            point=p
            dist=Point(p).distance(polygon)
    return point
def pairwise(a):
    return [(a1,a2) for a1,a2 in zip(a[1:],a)]