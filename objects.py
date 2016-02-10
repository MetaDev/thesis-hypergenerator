# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:30:20 2016

@author: Harald
"""
import scipy.stats
from shapely.geometry import Polygon

from shapely import affinity
class LayoutClass:
    SHAPES={"chair": [(0, 0), (0, 1), (1, 1),(1,0)],
            "table":[(0, 0), (0, 1), (2, 1),(2,0)]
            }
    class LayoutObject:
        def __init__(self,name,size, position_x,position_y, rotation,shape):
            self.name=name + str(id(self))
            self.shape=shape
            self.shape=affinity.translate(self.shape, position_x,position_y,0)
            self.shape=affinity.rotate(self.shape,rotation)
            self.shape=affinity.scale(self.shape,size,size,1)
        def __str__(self):
            return self.name + " " + str(self.shape)
            
    
    #check for property of the object if its a distribution and return sample or static value
    @staticmethod
    def __sample_property(prop):
        sample = getattr(prop, "sample", None)
        if callable(sample):
            size=sample()
        else:
            size=prop
        return size
    #sample a layout object and its children
    def sample_object(self,parent=None):
        LC = LayoutClass
        #the standard parent relation is scale for size and additional for rotation and position
        name=self.name
        size=LC.__sample_property(self.size)
        position_x=LC.__sample_property(self.position_x)
        position_y=LC.__sample_property(self.position_y)
        rotation=LC.__sample_property(self.rotation)
        if parent != None:
            size = parent.size * size
            position_x = position_x + parent.position_x
            position_y = position_y + parent.position_y
            rotation = rotation + parent.rotation
        obj = LC.LayoutObject(name=name,size=size,position_x=position_x,position_y=position_y,rotation=rotation,shape=self.shape)
        objs=[obj]
        for child_item in self.children:
            for i in range(self.__sample_property(child_item[0])):
                objs.append(*child_item[1].sample_object())
        return objs
    #size position and rotation are either a distribution
    # shape is a polygon defined by its coordinates
    # children is a collection of tuples that indicates the amount and type of children (instance of layout object)
    def __init__(self, name="", size=1, position_x=0,position_y=0, rotation=0,shape_exterior=[(0,0),(0,1),(1,1),(1,0)],children=[(0,None)]):
        self.name=name        
        self.size = size
        self.position_x = position_x
        self.position_y=position_y
        self.rotation = rotation
        self.shape=Polygon(shape_exterior)
        self.children=children
#wrapper class around scipy.stats.rv_continuous generic distributions class
class Distribution:
    def __init__(self,distr,low=0, high=1):
        self.low=low
        self.high=high
        self.distr=distr
    def sample(self):
        return self.low + self.distr.rvs(size=1)[0]*(self.high-self.low)
        
#test Distribution 
d1 = Distribution(distr=scipy.stats.uniform,low=0,high=4)
print (d1.sample()) 
d2 = Distribution(distr=scipy.stats.norm(loc=3,scale=5),low=0,high=4)
print (d2.sample()) 


#test LayoutClass 
rot = Distribution(distr=scipy.stats.uniform,low=0,high=360)
chair = LayoutClass(size=d2,name="chair", position_x=d1,position_y=d1, rotation=rot,shape_exterior=LayoutClass.SHAPES["chair"])
table = LayoutClass(size=3,name="table",position_x=d1,position_y=d2,shape_exterior=LayoutClass.SHAPES["table"],children=[(5,chair)])
shape = affinity.scale(Polygon(LayoutClass.SHAPES["table"]),3,3)
print(shape)
table_and_chairs = table.sample_object()
print ('\n'.join(str(p) for p in table_and_chairs) )
