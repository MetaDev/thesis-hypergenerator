# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:30:20 2016

@author: Harald
"""
import scipy.stats
from shapely.geometry import Polygon
from shapely.geometry import Point

import numpy
from shapely import affinity
from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt
from matplotlib import pyplot

COLOR = {
    True:  '#6699cc',
    False: '#ff3333'
    }

def v_color(ob):
    return COLOR[ob.is_valid]


class LayoutClass:
    SHAPES={"chair": [(0, 0), (0, 1), (1, 1),(1,0)],
            "table":[(0, 0), (0, 1), (2, 1),(2,0)]
            }
    #check for property of the object if its a distribution and return sample or static value
    @staticmethod
    def __sample_property(prop):
        sample = getattr(prop, "sample", None)
        if callable(sample):
            size=sample()
        else:
            size=prop
        return size
    class LayoutObject:
        def __init__(self,name,size, position_x,position_y, rotation,shape):
            self.name=name + str(id(self))
            self.size=size
            self.position_x=position_x
            self.position_y=position_y
            self.rotation=rotation
            #transform visual representation according to properties
            self.shape=shape
            self.shape=affinity.translate(self.shape, position_x,position_y,0)
            self.shape=affinity.rotate(self.shape,rotation)
            self.shape=affinity.scale(self.shape,size,size,1)
        def __str__(self):
            return '\n'.join(key + ": " + str(value) for key, value in vars(self).items())



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
            for i in range(LC.__sample_property(child_item[0])):
                objs.append(*child_item[1].sample_object())
        return objs

    def visualise_samples(layout_objects):
        if(pyplot.gcf()==0):
            fig = pyplot.figure("main")
        fig=pyplot.gcf()
        fig.clear()
        fig.clear()
        ax = fig.add_subplot(111)
        xrange=[0,0]
        yrange=[0,0]
        for layout_object in layout_objects:
            #plot exterior of polygon
            polygon=layout_object.shape
            x, y = polygon.exterior.xy
            ax.plot(x, y, 'o', color='#999999', zorder=1)
            #find range of polygon coordinates
            xrange = [int(numpy.floor(numpy.min(numpy.append(x,xrange[0])))),int(numpy.ceil(numpy.max(numpy.append(x,xrange[1]))))]
            yrange = [int(numpy.floor(numpy.min(numpy.append(y,yrange[0])))),int(numpy.ceil(numpy.max(numpy.append(y,yrange[1]))))]
            #plot surface
            patch = PolygonPatch(polygon, facecolor=v_color(polygon), edgecolor=v_color(polygon), alpha=0.5, zorder=1)
            ax.add_patch(patch)
        #finalise figure properties

        ax.set_title('Layout visualisation')
        ax.set_xlim(*xrange)
        ax.set_xticks(list(range(*xrange)) + [xrange[-1]])
        ax.set_ylim(*yrange)
        ax.set_yticks(list(range(*yrange)) + [yrange[-1]])
        ax.set_aspect(1)
        #show plot
        plt.show()


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
    #low and high is ignored if options (enums) is set
    #if nr_of_values is -1 the distribution is continous
    def __init__(self,distr,low=0, high=1, nr_of_values=-1, options=[]):
        self.low=low
        self.high=high
        self.distr=distr
        self.nr_of_values=nr_of_values
        self.options=options
    def sample(self):
        #normalise random value
        rnd = self.normalised_sample()
        cont_val = self.low + rnd*(self.high-self.low)
        if self.options:
            return self.options[int(numpy.round((len(self.options)-1)*rnd))]
        if self.nr_of_values!=-1:
            print("kak"+ str(rnd))
            return self.low + ((self.high-self.low)/self.nr_of_values)*numpy.round(self.distr.rvs(size=1)[0]*self.nr_of_values)
        return cont_val
    def normalised_sample(self):
        #we take the interval for 99.99% of the values because values from distribution can go to infinity
        _min = self.distr.interval(0.9999)[0]
        _max = self.distr.interval(0.9999)[1]
        return (numpy.clip(self.distr.rvs(size=1)[0],_min,_max)-_min)/(_max-_min)

#test Distribution
d1 = Distribution(distr=scipy.stats.uniform,low=0,high=4)
print (d1.sample())
d2 = Distribution(distr=scipy.stats.norm(),low=0,high=4)
print (d2.sample())
d3 = Distribution(distr=scipy.stats.uniform,options=["test1","test2","test3"])
print (d3.sample())
d3 = Distribution(distr=scipy.stats.norm(),nr_of_values=8)
print (d3.sample())


#test LayoutClass
rot = Distribution(distr=scipy.stats.uniform,low=0,high=360)
chair = LayoutClass(size=d2,name="chair", position_x=d1,position_y=d1, rotation=rot,shape_exterior=LayoutClass.SHAPES["chair"])
table = LayoutClass(size=3,name="table",position_x=d1,position_y=d2,shape_exterior=LayoutClass.SHAPES["table"],children=[(5,chair)])
shape = affinity.scale(Polygon(LayoutClass.SHAPES["table"]),3,3)
print(shape.exterior.xy)
table_and_chairs = table.sample_object()
print ('\n'.join(str(p) for p in table_and_chairs) )

#test visualisation

LayoutClass.visualise_samples(table_and_chairs)