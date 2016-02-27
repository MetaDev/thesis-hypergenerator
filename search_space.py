# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:30:20 2016

@author: Harald
"""
import scipy.stats
import numpy

class MarkovTreeNode:
    @staticmethod
    def sample_attr(attr):
        if MarkovTreeNode.attr_is_samplable(attr):
            sample=attr.sample()
        else:
            sample=attr
        return sample
    #check for property of the object if its a distribution and return sample or static value
    @staticmethod
    def sample_attr_vector(attr):
        if MarkovTreeNode.attr_is_vector(attr):
            return [MarkovTreeNode.sample_attr_vector(v) for v in attr]
        else:       
            return MarkovTreeNode.sample_attr(attr)
    @staticmethod                       
    def attr_is_vector(attr):
        return isinstance(attr, (list, tuple))
    @staticmethod
    def attr_is_samplable(attr):
        return callable(getattr(attr, "sample", None))
        
    class NodeSample:
         #an attribute is calculated
        def calc_child_attribute_from_parent(self, attr_name):
            ind_attr=self.independent_attr
            if not self.parent:
                setattr(self,attr_name,MarkovTreeNode.sample_attr_vector(ind_attr[attr_name]))
            elif attr_name not in self.p_to_c_map:
                setattr(self,attr_name,MarkovTreeNode.sample_attr_vector(ind_attr[attr_name]))
            else:
                #TODO if multiple parent attributes are given
                parent_attr=numpy.array([getattr(self.parent,attr) for attr in self.p_to_c_map[attr_name][1]])
                #TODO check if eah parent attribute has same length as attr value
                #if a user whishes to map differently for a vector, the user should define those vectors seperately
                if MarkovTreeNode.attr_is_vector(ind_attr[attr_name]):
                    attr=[self.p_to_c_map[attr_name][0](a,b)for a,b in zip(ind_attr[attr_name],parent_attr.T)]
                    #parent_attr is a vector of vector attributes
                    setattr(self,attr_name,attr)
                else:
                    setattr(self,attr_name,self.p_to_c_map[attr_name][0](ind_attr[attr_name],parent_attr))
        def __init__(self,node,index,parent_sample):
            #properties are saved independent from the parent, for later use of its values in machine learning
            self.name=node.name
            #sample properties from class property distribution, a property can be defined as property or statically by a value
            self.independent_attr = dict((ss_name,MarkovTreeNode.sample_attr_vector(ss_distr))for ss_name, ss_distr in node.search_spaces.items())
            #attributes to calc dependent values            
            self.p_to_c_map=node.p_to_c_map
            self.parent=parent_sample      
            
            #properties are saved related to the parent
            for ss_name in self.independent_attr.keys():
                self.calc_child_attribute_from_parent(ss_name)
            if parent_sample:
                self.index=str(index) + parent_sample.index
            else:
                self.index=str(index)
            
            
        def __str__(self):
            return '\n'.join(key + ": " + str(value) for key, value in vars(self).items())
   

        
    #sample a layout object and its children
    def sample(self,parent_sample=None,index=0,sample_list=None):
        #create  sample
        sample = MarkovTreeNode.NodeSample(self,index,parent_sample=parent_sample)
        if sample_list != None:
            sample_list.append(sample)
        child_samples=[]
        for child in self.children:
            for i in range(int(self.sample_attr_vector(child[0]))):
                #sample child object given parent sample
                #the star is to create a singel list in the recursion
                child_samples.append(child[1].sample(sample,i,sample_list))
            setattr(sample,"children",child_samples)
        return sample


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
    shape_exteriors={"square": [(0, 0), (0, 1), (1, 1),(1,0)]}
    standard_func = {"scale": lambda a, b: a*b,
                     "add": lambda a,b : a+b
                             }
    standard_property_map = {"position":(standard_func["add"],["position"]),
                    "rotation":(standard_func["add"],["rotation"]),
                    "size":((standard_func["scale"],["size"]))
                    }
   
     
    def __init__(self, name, size=(1,1),position=[0,0], rotation=0,shape=[(0,0),(0,1),(1,1),(1,0)], color="b",children=[(0,None)], property_map=standard_property_map):
        super().__init__(name,{"size":size, "position":position,"rotation":rotation,"color":color,"shape":shape},children,property_map)        

               
class Distribution():
    #low and high is ignored if options (enums) is set
    #if nr_of_values is -1 the distribution is continous
    def __init__(self,distr:scipy.stats.rv_continuous=scipy.stats.uniform,low=0, high=1, nr_of_values=-1, options=[]):
        self.low=low
        self.high=high
        self.distr=distr
        self.nr_of_values=nr_of_values
        self.options=options
        #Continous, Ordered Discrete ,Unordered Discrete
        if self.options:
            self.var_type="u"
        elif self.nr_of_values!= -1:
            self.var_type="o"
        else:
            self.var_type="c"
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
    d1 = Distribution(low=-1,high=1)
    d3 = Distribution(low=0.2,high=0.4)
    d2= Distribution(low=4,high=8,nr_of_values=4)
    colors = Distribution(options=["red","green","blue"])
    #test MarkovTreeNode
    #it should be possible to have stochastic shape as well, defined by its own points
    #to allow this each attribute of a markovnode should be possibly a markov node as well
    #shapes1= Distribution(options=[[(0, 0), (0, 1), (1, 1),(1,0)],[(0, 0), (0, 1),(0.5,1),(0.5,0.5),(1, 0.5),(1,0)]])
    shapes1= Distribution(options=[[(0, 0), (0, 1),(0.5,1),(0.5,0.5),(1, 0.5),(1,0)]])

    p1=Distribution(low=0,high=0.5)
    p2=Distribution(low=0.5,high=1)
    shapes2= [(p1,p1),(p1,p2),(p2,p2),(p2,p1)]
    rot = Distribution(distr=scipy.stats.uniform,low=0,high=360)
    
    chair = LayoutMarkovTreeNode(size=(d3,d3),name="chair", position=(d1,d1), rotation=rot,shape=shapes2, color=colors)
    table = LayoutMarkovTreeNode(name="table",position=(1,2),shape=shapes1,children=[(d2,chair)],color="blue")
    list_of_samples=[]    
    root = table.sample(sample_list=list_of_samples)
    return list_of_samples,root
    
