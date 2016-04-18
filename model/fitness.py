from shapely.geometry import LineString
from shapely.ops import cascaded_union
from itertools import combinations
import networkx as nx
import numpy as np
import model.mapping as mp
import util.utility as ut


#add interface for fitness, such that it can be either an optimisation (soft constraint) or a threshhold function (hard constraint)
#also always option to return normalised fitness
#fitness 1 should mean good 0 bad

class Fitness:
    def __init__(self,func,order,threshhold,regression_target):
        self.func=func
        self.order=order
        self.threshhold=threshhold
        self.regression_target=regression_target
    def calc(self,*args):
        temp=self.func(*args) ** self.order
        return temp if temp > self.threshhold else None


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



def fitness_polygon_overl(sample0,sample1):
    polygons=mp.map_layoutsamples_to_geometricobjects([sample0,sample1],shape_name="shape")
    return pairwise_overlap(polygons[0],polygons[1])

def fitness_polygon_alignment(sample0,sample1):
    polygons=mp.map_layoutsamples_to_geometricobjects([sample0,sample1],shape_name="shape")
    return pairwise_closest_line_alignment(polygons[0],polygons[1],threshold=30)
def fitness_min_dist(sample0,sample1):
    pos0=sample0.values["ind"]["position"]
    pos1=sample1.values["ind"]["position"]
    return pairwise_min_dist(pos0,pos1,threshold=1)
#print
from scipy.stats import binned_statistic
import numpy as np
#calculate fitness statistics
def fitness_statistics(fitness_values,verbose=False,summary=True):
    fitness_statistics_count, fitness_bin_edges,_ = binned_statistic(fitness_values, fitness_values,
                                                                     statistic='count', bins=20)
    if not summary:
        fitness_statistics_mean = binned_statistic(fitness_values, fitness_values,  statistic='mean'
        , bins=20)[0]
        all_edges=[]
        counts=[]
        means=[]
        for edges,count,mean,i in zip(ut.pairwise(fitness_bin_edges),fitness_statistics_count,
            fitness_statistics_mean,range(len(fitness_statistics_count))):

            edges ="bin nr. "+ str(i) + ": "+format(edges[0], '.5g') + " - "+format(edges[1], '.5g')
            count = ": count= " + str(count)
            mean=": mean= "+ format(mean, '.5g')
            if verbose:
                print(edges + count + mean)
            all_edges.append(edges)
            counts.append(count)
            means.append(mean)
        return means,all_edges,counts
    else:
        if verbose:
            print("total mean= {0:f}: variance= {1:f}".format(np.mean(fitness_values),np.var(fitness_values),'.5g'))
        return np.mean(fitness_values),np.var(fitness_values)

