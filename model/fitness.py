# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 12:56:37 2016

@author: Harald
"""
from shapely.geometry import Point,LineString
from shapely.ops import cascaded_union
from itertools import combinations
import networkx as nx
import numpy as np
import model.mapping as mp
import util.utility as ut


#add interface for fitness, such that it can be either an optimisation (soft constraint) or a threshhold function (hard constraint)
#also always option to return normalised fitness
#fitness 1 should mean good 0 bad


def angle(line_segment):
    p0=line_segment[0]
    p1=line_segment[1]
    return np.rad2deg(np.arctan2((p1[1]-p0[1]),(p1[0]-p0[0])))
def normalise_fitness(fitness_values):
    return (np.array(fitness_values)-np.min(fitness_values))/(np.max(fitness_values)-np.min(fitness_values))

def pairwise_overlap(polygon0,polygon1):
    total_surface=max(polygon0.area,polygon1.area)
    return 1-((polygon0.intersection(polygon1)).area/total_surface)

def pairwise_closest_line_alignment(polygon0,polygon1,threshold=30):
    min_dist=float("inf")
    min_l0=None
    min_l1=None
    for p00,p01 in ut.pairwise(polygon0.exterior.coords):
        for p10,p11 in ut.pairwise(polygon1.exterior.coords):
            l0=LineString([p00,p01])
            l1=LineString([p10,p11])
            dist=l1.distance(l0)
            if dist<min_dist:
                min_dist=dist
                min_l0=l0.coords
                min_l1=l1.coords

    #the fitness is a thresholded difference between the 2 closest lines of a polygon
    #whether a line is 180 or 0 doesn't mather for alignment
    return 1-(max(np.abs((angle(min_l0)%180)-(angle(min_l1)%180))-threshold,0)/(180-threshold))

def pairwise_min_dist(position0,position1,threshold, dist_metric=np.linalg.norm):
    return min(dist_metric(np.array(position0)- np.array(position1))/threshold,1)

#calculate overlapping surface size
def combinatory_unique_surface(polygons):
    return (cascaded_union([pair[0].intersection(pair[1]) for pair in combinations(polygons, 2)]).area)

#calculate density (polygon occupation) of surface in a bounded space
def area_density(polygons, polygon_bound):
    return cascaded_union([polygon.intersection(polygon_bound) for polygon in polygons]).area
#this is a direct metric and does not require polygon conversion
#total distance between layout_objcts
#todo calculate distance bqsed on polygon centroid

def dist_between_parent_child(parent,children, dist_metric=np.linalg.norm):
    return [dist_metric(np.array([parent.position,child.position])) for child in children]
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
    points=mp.map_geometricobjects_to_nodes(graph,polygon_sequence)
    paths=[]
    for (polygon_pairs),(point_pairs) in zip(ut.pairwise(polygon_sequence),ut.pairwise(points)):
        path=shortest_path(graph,point_pairs[0],point_pairs[1])
        paths.extend(path)
    return paths

#fitness that calculates a sequence of occurences on a path-> from point 1 to 2 avoind a set of polygon what are the polygons you intersect with
#a direct metric that calculates the angle of intersection between polygons
#metric
#also add fitness function that evaluates a certain metric towards a treshhold

#MST between points->use graph kernel for likelyhood
