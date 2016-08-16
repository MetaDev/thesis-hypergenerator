from shapely.geometry import LineString,MultiPolygon
from shapely.ops import cascaded_union

import numpy as np
import model.mapping as mp
import util.utility as ut

from enum import Enum

#add interface for fitness, such that it can be either an optimisation (soft constraint) or a threshhold function (hard constraint)
#also always option to return normalised fitness
#fitness 1 should mean good 0 bad
class Fitness_Relation(Enum):
    pairwise_parent_child=1
    pairwise_siblings=2
    absolute=3

class Fitness:
    def __init__(self,name,func,fitness_relation,order,threshhold,regression_target):
        self.name=name
        self.func=func
        self.fitness_relation=fitness_relation
        self.order=order
        self.threshhold=threshhold
        self.regression_target=regression_target

    def calc(self,parent,child,siblings):
        if self.fitness_relation is Fitness_Relation.pairwise_parent_child:
            return self.func(parent,child)
        elif self.fitness_relation is Fitness_Relation.absolute:
            return self.func(parent,siblings)
        else:
            return self.func(parent,child,siblings)
    def __str__(self):
        return "\n".join(key + ": " + str(value) for key, value in vars(self).items())

#if the fitness has an extra option that acts as target
class Targetting_Fitness(Fitness):
    def __init__(self,name,func,fitness_relation,order,threshhold,regression_target,target):
        super().__init__(name,func,fitness_relation,order,threshhold,regression_target)
        self.target=target
    def calc(self,parent,child,siblings):
        if self.fitness_relation is Fitness_Relation.pairwise_parent_child:
            return self.func(parent,child,self.target)
        elif self.fitness_relation is Fitness_Relation.absolute:
            return self.func(parent,siblings,self.target)
        else:
            return self.func(parent,child,siblings,self.target)
    def __str__(self):
        return Fitness.__str__(self)


#calculate fitness of a given parent-children model
#this method should simply pass the samples to each function
#each function will take the responsibility of deciding whether it uses only parent->child, child->siblings or both
def fitness_calc(parent,siblings,fitness_list):
    fitness_func_values=[]
    for fitn in fitness_list:
        sibling_fitness=[]
        previous_siblings=[]
        if not fitn.fitness_relation is Fitness_Relation.absolute:
            for child in siblings:
                if fitn.fitness_relation is Fitness_Relation.pairwise_parent_child:
                    sibling_fitness.append(fitn.calc(parent,child,None))
                #one seperate case where the fitness is between siblings and there are no siblings yet
                elif fitn.fitness_relation is Fitness_Relation.pairwise_siblings and not len(previous_siblings) == 0 :
                    sibling_fitness.append(fitn.calc(parent,child,previous_siblings))


                previous_siblings.append(child)

        else:
            #if the fitness is absolute only one fitness per sibling relation is necessary
            sibling_fitness.append(fitn.calc(parent,None,siblings))
        fitness_func_values.append(sibling_fitness)

    return fitness_func_values

#only usable if the pairwise_fitn is normalised between 0 and 1
def _calc_pairwise_sibling(child,siblings,pairwise_norm_fitn):
    fitness_func_value_sibling=1
    for sibling in siblings:
        fitness_func_value_sibling*=(pairwise_norm_fitn(child,sibling))

    #the fitness of a child relative to its existing siblings is the average of its pairwise fitness
    return fitness_func_value_sibling

def _calc_pairwise_sibling_target(child,siblings,pairwise_norm_fitn,target):
    fitness_func_value_sibling=[]
    for sibling in siblings:
        fitness_func_value_sibling.append(pairwise_norm_fitn(child,sibling,target))

    #the fitness of a child relative to its existing siblings is the product of its pairwise fitness
    return np.prod(fitness_func_value_sibling) if len(siblings) >0 else 1

#collision constraints
def negative_overlap(sample0,sample1):
    polygons=mp.map_layoutsamples_to_geometricobjects([sample0,sample1],shape_name="shape")
    return -((polygons[0].intersection(polygons[1])).area)
#avoid collision with parent
def negative_overlap_pc(parent,child):
    return negative_overlap(parent,child)
def norm_overlap_pc(parent,child):
    polygons=mp.map_layoutsamples_to_geometricobjects([parent,child],shape_name="shape")
    noemer=min(polygons[0].area,polygons[1].area)
    return 1-((polygons[0].intersection(polygons[1])).area)/noemer
def union_negative_overlap_sb(parent,child,siblings):
    sibling_polygons=mp.map_layoutsamples_to_geometricobjects(siblings,shape_name="shape")
    child_polygon=mp.map_layoutsamples_to_geometricobjects([child],shape_name="shape")[0]

    return -(cascaded_union([sibling_polygon.intersection(child_polygon) for sibling_polygon in sibling_polygons]).area)

def positive_overlap(sample0,sample1):
    polygons=mp.map_layoutsamples_to_geometricobjects([sample0,sample1],shape_name="shape")
    return ((polygons[0].intersection(polygons[1])).area)

#constraint to keep children in the parent
def positive_overlap_pc(parent,child):
    return positive_overlap(parent,child)


#surface ratio constraint
def combinatory_surface_ratio_absolute(parent,siblings,target_ratio):
    if not 0<= target_ratio <=1:
        raise ValueError("Target ration should be between 0 and 1, it's values was: ",target_ratio)
    sibling_polygons=mp.map_layoutsamples_to_geometricobjects(siblings,shape_name="shape")
    parent_polygon=mp.map_layoutsamples_to_geometricobjects([parent],shape_name="shape")[0]

    sibling_union=cascaded_union(sibling_polygons)
    area_siblings_parent= sibling_union.intersection(parent_polygon).area
    area_diff=np.abs(area_siblings_parent/parent_polygon.area-target_ratio)

    return -area_diff




#balance constraints

def min_dist_norm(sample0,sample1,target_dist,dist_metric=np.linalg.norm):
    position0=sample0.values["rel"]["position"]
    position1=sample1.values["rel"]["position"]
    return min(dist_metric(np.array(position0)- np.array(position1))/target_dist,1)


def min_dist_sb(parent,child,siblings,target_dist):
    return _calc_pairwise_sibling_target(child,siblings,min_dist_norm,target_dist)
def min_dist_pc(parent,child,target_dist):
    return min_dist_norm(parent,child,target_dist)

#difference between centroid of parent and grouped children
def centroid_dist_sb(parent,child,siblings,dist_metric=np.linalg.norm):
    sibling_polygons=mp.map_layoutsamples_to_geometricobjects(siblings,shape_name="shape")
    child_polygon=mp.map_layoutsamples_to_geometricobjects([child],shape_name="shape")
    parent_polygon_centr=np.array(mp.map_layoutsample_to_geometricobject(parent,shape_name="shape")[0].centroid)


    sibling_polygons_group_centr = np.array(MultiPolygon(sibling_polygons).centroid)
    sibling_child_polygons_group_centr = np.array(MultiPolygon(sibling_polygons+child_polygon).centroid)

    centr_dist_without_child=dist_metric(sibling_polygons_group_centr-parent_polygon_centr)
    centr_dist_with_child=dist_metric(sibling_child_polygons_group_centr-parent_polygon_centr)
    #if the distance grows by adding a child it is bad
    return centr_dist_without_child-centr_dist_with_child

#the input is all siblings not the previous ones only
def centroid_dist_absolute(parent,siblings,dist_metric=np.linalg.norm):
    sibling_polygons=mp.map_layoutsamples_to_geometricobjects(siblings,shape_name="shape")
    parent_polygon_centr=np.array(mp.map_layoutsample_to_geometricobject(parent,shape_name="shape").centroid)

    sibling_polygons_group_centr = np.array(MultiPolygon(sibling_polygons).centroid)

    centr_dist_with_child=dist_metric(sibling_polygons_group_centr -parent_polygon_centr)

    #if the distance grows by adding a child it is bad
    return -centr_dist_with_child
def rotation_alignment_sb(parent,child,siblings):
    return _calc_pairwise_sibling(child,siblings,rotation_alignment)
#alignment cosntraints
def rotation_alignment(sample0,sample1):
    rotation0=sample0.values["rel"]["rotation"]*2*np.pi
    rotation1=sample1.values["rel"]["rotation"]*2*np.pi
    return (1+np.cos(4*(rotation0-rotation1)))/2
def closest_side_alignment_norm(sample0,sample1):
    polygons=mp.map_layoutsamples_to_geometricobjects([sample0,sample1],shape_name="shape")
    min_dist=float("inf")
    min_l0=None
    min_l1=None
    for p00,p01 in ut.pairwise(polygons[0].exterior.coords):
        for p10,p11 in ut.pairwise(polygons[1].exterior.coords):
            l0=LineString([p00,p01])
            l1=LineString([p10,p11])
            dist=l1.distance(l0)
            if dist<min_dist:
                min_dist=dist
                min_l0=l0.coords
                min_l1=l1.coords

    #the constraint of alignement between 2 closest sides of 2 polygons as defined in a paper
    #the difference is defined over 90 degrees
    angle0=_angle(min_l0)
    angle1=_angle(min_l1)
    return (1+np.cos(4*(angle0-angle1)))/2

#angle between line segments in radians
def _angle(line_segment):
    p0=line_segment[0]
    p1=line_segment[1]
    return np.arctan2((p1[1]-p0[1]),(p1[0]-p0[0]))

def closest_side_alignment_pc(parent,child):
    return closest_side_alignment_norm(parent,child)
def closest_side_alignment_sb(parent,child,siblings):
    return _calc_pairwise_sibling(child,siblings,rotation_alignment)


#TODO
#polygon convexity

#TODO
#reachibility




#print
from scipy.stats import binned_statistic
import numpy as np

#calculate fitness statistics
def fitness_statistics(fitness_values,summary=True):
    fitness_statistics_count, fitness_bin_edges,_ = binned_statistic(fitness_values, fitness_values,
                                                                     statistic='count', bins=20)
    if not summary:
        print("bin nr. and size ,count ,mean")
        fitness_statistics_mean = binned_statistic(fitness_values, fitness_values,  statistic='mean'
        , bins=20)[0]
        for edges,count,mean,i in zip(ut.pairwise(fitness_bin_edges),fitness_statistics_count,
            fitness_statistics_mean,range(len(fitness_statistics_count))):

            edges = str(i) + ": "+format(edges[0], '.5g') + " - "+format(edges[1], '.5g')
            count =  str(count)
            mean= format(mean, '.5g')
            print(edges + ","+ count + ","+ mean)

    print("total mean= {0:f}: variance= {1:f}".format(np.mean(fitness_values),np.var(fitness_values),'.5g'))

