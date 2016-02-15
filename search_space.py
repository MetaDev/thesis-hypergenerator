# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:30:20 2016

@author: Harald
"""
import scipy.stats
import numpy

from shapely import affinity
from shapely.geometry import Polygon

from descartes.patch import PolygonPatch

from matplotlib import pyplot as plt

COLOR = {
    True:  '#6699cc',
    False: '#ff3333'
    }

def v_color(ob):
    return COLOR[ob.is_valid]


class LayoutDefinition:
    shape_exteriors={"chair": [(0, 0), (0, 1), (1, 1),(1,0)],
            "table":[(0, 0), (0, 1), (2, 1),(2,0)]
            }
    standard_property_map = {"size": lambda parent, child: parent.size*child.size,
                             "position_x": lambda parent, child: parent.position_x+child.position_x,
                             "position_y": lambda parent, child: parent.position_y+child.position_y,
                             "rotation": lambda parent, child: parent.rotation+child.rotation
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
    class LayoutSample:
        def __init__(self,name,parent,p_to_c_map,size, position_x,position_y, rotation,shape, color):
            #properties are saved independent of the parent
            self.name=name + str(id(self))
            self.size=size
            self.parent=parent
            self.position_x=position_x
            self.position_y=position_y
            self.rotation=rotation
            self.color=color
            #transform visual representation according to properties
            self.shape=shape
            #calculate properties based on parent properties
            if parent != None:
                position_x = p_to_c_map["position_x"](parent,self)
                position_y = p_to_c_map["position_y"](parent,self)
                rotation = p_to_c_map["rotation"](parent,self)
                size = p_to_c_map["size"](parent,self)
            self.shape=affinity.rotate(self.shape,rotation)
            self.shape=affinity.translate(self.shape, position_x,position_y,0)
            self.shape=affinity.scale(self.shape,size,size)
        def __str__(self):
            return '\n'.join(key + ": " + str(value) for key, value in vars(self).items())



    #sample a layout object and its children
    def create_sample(self,parent=None):
        #some name substitution for readability
        layout_def = self
        LC = LayoutDefinition

        #sample properties from class property distribution, a property can be defined as property or statically by a value
        size=LC.__sample_property(layout_def.size)
        position_x=LC.__sample_property(layout_def.position_x)
        position_y=LC.__sample_property(layout_def.position_y)
        rotation=LC.__sample_property(layout_def.rotation)
        color=LC.__sample_property(layout_def.color)

        #create layout sample
        layout_sample = LC.LayoutSample(name=layout_def.name,parent=parent,p_to_c_map=layout_def.property_map,size=size,position_x=position_x,position_y=position_y,rotation=rotation,shape=self.shape,color=color)


        samples=[layout_sample]
        for child_item in layout_def.children:
            for i in range(LC.__sample_property(child_item[0])):
                #sample child object given parent sample
                samples.append(*child_item[1].create_sample(layout_sample))
        return samples

    @staticmethod
    def visualise_samples(layout_objects,visualise_edges=False):
        if(plt.gcf()==0):
            fig = plt.figure("main")
        fig=plt.gcf()
        fig.clear()
        fig.clear()
        ax = fig.add_subplot(111)
        xrange=[]
        yrange=[]
        polygons=[]
        for layout_object in layout_objects:
            polygon=layout_object.shape
            polygons=numpy.append(polygons,layout_object.shape)
            x, y = polygon.exterior.xy
            if visualise_edges:
                #plot edges of polygon
                ax.plot(x, y, 'o', color='#999999', zorder=1)
            #find range of polygon coordinates
            if xrange:
                x=numpy.append(x,xrange)
                y=numpy.append(y,yrange)
            xrange = [int(numpy.floor(numpy.min(x))),int(numpy.ceil(numpy.max(x)))]
            yrange = [int(numpy.floor(numpy.min(y))),int(numpy.ceil(numpy.max(y)))]
            #plot surface
            patch = PolygonPatch(polygon, facecolor=layout_object.color, edgecolor=layout_object.color, alpha=0.5, zorder=1)
            ax.add_patch(patch)
        #finalise figure properties

        ax.set_title('Layout visualisation')
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        #aspect ratio of plot
        ax.set_aspect(1)
        #show plot
        plt.show()
        #return range for other uses
        return (xrange,yrange,polygons)


    #size position and rotation are either a distribution
    # shape is a polygon defined by its coordinates
    # children is a collection of tuples that indicates the amount and type of children (instance of layout object)
    #the color variable is used in visualisation using matplotlib
    def __init__(self, name="", size=1, position_x=0,position_y=0, rotation=0,shape_exterior=[(0,0),(0,1),(1,1),(1,0)], color="b",children=[(0,None)], property_map=standard_property_map):
        self.name=name
        self.size = size
        self.position_x = position_x
        self.position_y=position_y
        self.rotation = rotation
        self.color=color
        self.shape=Polygon(shape_exterior)
        self.children=children
        self.property_map=property_map
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
            return self.low + ((self.high-self.low)/self.nr_of_values)*numpy.round(self.distr.rvs(size=1)[0]*self.nr_of_values)
        return cont_val
    def normalised_sample(self):
        #we take the interval for 99.99% of the values because values from distribution can go to infinity
        _min = self.distr.interval(0.9999)[0]
        _max = self.distr.interval(0.9999)[1]
        return (numpy.clip(self.distr.rvs(size=1)[0],_min,_max)-_min)/(_max-_min)
def test():
    #test Distribution
    d1 = Distribution(distr=scipy.stats.uniform,low=-4,high=4)
    print (d1.sample())
    d2 = Distribution(distr=scipy.stats.norm(),low=0,high=4)
    print (d2.sample())
    d3 = Distribution(distr=scipy.stats.uniform,options=["red","green","blue"])
    print (d3.sample())
    d4 = Distribution(distr=scipy.stats.norm(),nr_of_values=8)
    print (d4.sample())
    
    
    #test LayoutDefinition
    rot = Distribution(distr=scipy.stats.uniform,low=0,high=360)
    
    chair = LayoutDefinition(size=0.2,name="chair", position_x=d1,position_y=d1, rotation=rot,shape_exterior=LayoutDefinition.shape_exteriors["chair"], color="green")
    table = LayoutDefinition(size=1,name="table",position_x=3,position_y=2,shape_exterior=LayoutDefinition.shape_exteriors["table"],children=[(4,chair)],color="blue")
    table_and_chairs = table.create_sample()
    
    #test visualisation
    LayoutDefinition.visualise_samples(table_and_chairs)