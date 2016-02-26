# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 12:56:37 2016

@author: Harald
"""
from shapely.geometry import Point
from shapely.ops import cascaded_union
from itertools import combinations
import networkx as nx
import numpy
import mapping
import utility
import search_space

class Direct:
    def sum_samples(samples,sample_attr):
        return sum(sample.sample_attr for sample in samples)
        
#calculate overlapping surface size
def surface_overlap(polygons):
    return cascaded_union([pair[0].intersection(pair[1]) for pair in combinations(polygons, 2)]).area
#calculate density (polygon occupation) of surface in a bounded space
def area_density(polygons, polygon_bound):
    return cascaded_union([polygon.intersection(polygon_bound) for polygon in polygons]).area
#this is a direct metric and does not require polygon conversion
#total distance between layout_objcts
#todo calculate distance bqsed on polygon centroid
def pairwise_dist(positions, dist_metric=numpy.linalg.norm):
    return [dist_metric(numpy.array([position_pair[0],position_pair[1]])) for position_pair in utility.pairwise(positions)]
def dist_between_parent_child(parent,children, dist_metric=numpy.linalg.norm):
    return [dist_metric(numpy.array([parent.position,child.position])) for child in children]
#calculate collision or intersection between layout samples
    
#add named functions, such that 2 layout object collections can be compared e.g. in surface by layout definition name

#nr of intersections between polygons

#gameplay consrtaints

#todo combine both methods in single one
#path between points in graph, if not reachable ?
#for now approximate possible movement with 8 directions and grid (orthogonal and diagonal)
def shortest_path(graph, source:(int,int), target:(int,int)):
    path = nx.shortest_path(graph,source,target)
    return path
#check if the polygons can be reached in sequence while avoiding other polygon obstacles
def polygon_path_sequence(graph,polygon_sequence):
    points=mapping.map_geometricobjects_to_nodes(graph,polygon_sequence)
    paths=[]
    for (polygon_pairs),(point_pairs) in zip(utility.pairwise(polygon_sequence),utility.pairwise(points)):
        path=shortest_path(graph,point_pairs[0],point_pairs[1])
        paths.extend(path)
    return paths
    
#fitness that calculates a sequence of occurences on a path-> from point 1 to 2 avoind a set of polygon what are the polygons you intersect with
#a direct metric that calculates the angle of intersection between polygons
#metric
#also add fitness function that evaluates a certain metric towards a treshhold
    
#MST between points->use graph kernel for likelyhood
