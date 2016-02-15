# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 12:56:37 2016

@author: Harald
"""
from shapely.geometry import Point
from shapely.ops import cascaded_union
from itertools import combinations
import numpy

#calculate overlapping surface size
def surface_overlap(layout_objects):
    shapes = [lo.shape for lo in layout_objects]
    cascaded_union([pair[0].intersection(pair[1]) for pair in combinations(shapes, 2)]).area
#calculate density (polygon occupation) of surface in a bounded space
def area_density(layout_objects, bounds):
    return
#total distance between layout_objcts
def total_dist(layout_objects, dist_metric=numpy.linalg.norm):
    shapes = [lo.shape for lo in layout_objects]
    total_dist=0
    for shape in shapes:
        total_dist= dist_metric(numpy.array(shape.centroid))
    return total_dist

#any fitness function can be applied on any collection of polygons
#add named functions, such that 2 layout object collections can be compared e.g. in surface by layout definition name

#nr of intersections between polygons

#gameplay consrtaints
#distance between points (for maze), -1 of not reachable
#maybe function that builds graph based on layout
#add reachability function
#maybe find periodicness
def convert_to_neighbourhoud_graph(layout_objects):

def shortest_path(objects_to_avoid, point1, point2, directions_step=0):
    #use : http://www.redblobgames.com/pathfinding/a-star/implementation.html
    return
#fitness that calculates a sequence of occurences on a path-> from point 1 to 2 avoind a set of polygon what are the polygons you intersect with
#a direct metric that calculates the angle of intersection between polygons
#metric
#also add fitness function that evaluates a certain metric towards a treshhold