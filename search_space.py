# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:30:20 2016

@author: Harald
"""
import scipy.stats
import numpy


class MarkovTreeNode:
    
    #check for property of the object if its a distribution and return sample or static value
    @staticmethod
    def sample_search_space_distr(ss_distr):
        sample = getattr(ss_distr, "sample", None)
        if callable(sample):
            size=sample()
        else:
            size=ss_distr
        return size
    class NodeSample:
        def __init__(self,node,parent_sample):
            #properties are saved independent from the parent, for later use of its values in machine learning
            self.name=node.name + str(id(self))
            #sample properties from class property distribution, a property can be defined as property or statically by a value
            self.independent_attr = dict((ss_name,MarkovTreeNode.sample_search_space_distr(ss_distr))for ss_name, ss_distr in node.search_spaces.items())

            #properties are saved related to the parent
            for ss_name,ss_distr in node.search_spaces.items():
                if not parent_sample:
                    setattr(self,ss_name,MarkovTreeNode.sample_search_space_distr(ss_distr))
                elif ss_name not in node.p_to_c_map:
                    setattr(self,ss_name,MarkovTreeNode.sample_search_space_distr(ss_distr))
                else:
                    setattr(self,ss_name,node.p_to_c_map[ss_name](parent_sample,self.independent_attr[ss_name]))
            self.parent_sample=parent_sample
            
        def __str__(self):
            return '\n'.join(key + ": " + str(value) for key, value in vars(self).items())



    #sample a layout object and its children
    def sample(self,parent_sample:NodeSample=None):
        #create  sample
        sample = MarkovTreeNode.NodeSample(self,parent_sample=parent_sample)
        samples=[sample]
        for child in self.children:
            for i in range(self.sample_search_space_distr(child[0])):
                #sample child object given parent sample
                #the star is to create a singel list in the recursion
                samples.append(*child[1].sample(sample))
        return samples



    #size position and rotation are either a distribution
    # shape is a polygon defined by its coordinates
    # children is a collection of tuples that indicates the amount and type of children (instance of layout object)
    #the color variable is used in visualisation using matplotlib
    def __init__(self, name,search_spaces,children=[(0,None)],p_to_c_map=None):
        self.name=name
        self.search_spaces = search_spaces
        self.children=children
        self.p_to_c_map=p_to_c_map
        
        
class LayoutMarkovTreeNode(MarkovTreeNode):
    shape_exteriors={"chair": [(0, 0), (0, 1), (1, 1),(1,0)],
            "table":[(0, 0), (0, 1), (2, 1),(2,0)]
            }
    standard_property_map = {"size": lambda parent, size: parent.size*size,
                             "position_x": lambda parent, position_x: parent.position_x+position_x,
                             "position_y": lambda parent, position_y: parent.position_y+position_y,
                             "rotation": lambda parent, rotation: parent.rotation+rotation}
     
    def __init__(self, name, size=1, position_x=0,position_y=0, rotation=0,shape=[(0,0),(0,1),(1,1),(1,0)], color="b",children=[(0,None)], property_map=standard_property_map):
        super().__init__(name,{"size":size,"position_x":position_x,"position_y":position_y,"rotation":rotation,"color":color,"shape":shape},children,property_map)        

#wrapper class around scipy.stats.rv_continuous generic distributions class
class Distribution:
    #low and high is ignored if options (enums) is set
    #if nr_of_values is -1 the distribution is continous
    def __init__(self,distr:scipy.stats.rv_continuous,low=0, high=1, nr_of_values=-1, options=[]):
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
    d2 = Distribution(distr=scipy.stats.norm(),low=0,high=4)
    d3 = Distribution(distr=scipy.stats.uniform,options=["red","green","blue"])
    d4 = Distribution(distr=scipy.stats.norm(),nr_of_values=8)
      
    #test MarkovTreeNode
    rot = Distribution(distr=scipy.stats.uniform,low=0,high=360)
    
    chair = LayoutMarkovTreeNode(size=0.2,name="chair", position_x=d1,position_y=d1, rotation=rot,shape=LayoutMarkovTreeNode.shape_exteriors["chair"], color="green")
    table = LayoutMarkovTreeNode(size=1,name="table",position_x=3,position_y=2,shape=LayoutMarkovTreeNode.shape_exteriors["table"],children=[(4,chair)],color="blue")
    table_and_chairs = table.sample()
    return table_and_chairs
    
