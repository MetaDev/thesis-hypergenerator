# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:30:20 2016

@author: Harald
"""
import scipy.stats
import numpy as np
from util import setting_values
from gmr import GMM
from typing import List
from util import utility as ut
from typing import  TypeVar
from operator import itemgetter


class Variable():
    #name can be either a string or a list of string
    #size is an int that defines the size of the variable
    #the func is a function that is applied on each element
    def __init__(self, name,func=None):
        self.name=name
        self._value=None
        self.parent_vars_name=None
        self.unpack=False
        self.func=func
        if func:
            self.parent_vars_name=func.__code__.co_varnames[1:]
    def sample(self, parent_sample):
        return self._value
    #the last boolean is to indicate whether the returned value needs to be unpacked
    def relative_sample(self,parent_sample):
        #if there is a function and the parent has the required attributes
        #todo check this with exceptions
        if self.func and parent_sample:
            parent_var_values=[parent_sample.relative_vars[var_name] for var_name in self.parent_vars_name]
            return self.func(self._value,*parent_var_values)
        else:
            return self._value

    #return if is stochastic
    def stochastic(self):
        pass



class StochasticVariable(Variable):
    @staticmethod
    def standard_distr(low=0,high=1):
        return scipy.stats.uniform(loc=low,scale=high-low)
    @staticmethod
    def standard_choices(low=0,high=1,num=1):
        return np.linspace(low,high,num)
    #low and high is ignored if options (enums) is set
    #if nr_of_values is -1 the distribution is continous
    def __init__(self,name,size=1,func=None,distr=None,choices=None):
        super().__init__(name,func)
        self.size=size
        #Discrete
        if choices is not None:
            self.choices=choices
        elif distr is not None and hasattr(distr,"rvs"):
            #continuous
            self.distr=distr
        else:
            raise ValueError("Either the distribution or the choices have to be set.")

    def sample(self,parent_sample):
        if hasattr(self,"choices"):
            self._value=np.random.choice(self.choices,size=self.size)
        else:
            self._value=self.distr.rvs(size=self.size)
        return super().sample(parent_sample)

    def stochastic(self):
        return True




class DeterministicVariable(Variable):
    def __init__(self,name,value,func=None):
        super().__init__(name,func)
        self._value=value

    def stochastic(self):
        return False


class VectorVariable(Variable):
    #using the naming convention of a vector variable p1, p2,...
    #i extract a vector variable in its correct order from the flattened dict of variable values
    #in a nodesample
    @staticmethod
    def extract_ordered_list_vars(list_name,sample_var_dict):
        var_list=[]
        for name,value in sample_var_dict.items():
            if ut.remove_numbers(name)==list_name:
                var_list.append((ut.get_trailing_number(name),value))
        return [var[1] for var in sorted(var_list,key=itemgetter(0))]

    @staticmethod
    def from_deterministic_list(list_name:str,list_):
        return [DeterministicVariable(list_name+str(i),list_[i]) for i in range(len(list_))]
    #name indicates the collection of variables
    def __init__(self,name,variables:List[Variable]):
        super().__init__(name,None)
        self.variables=variables
        self.unpack=True
        self.unpack_names=[v.name for v in variables]
        self._stochastic=[v.stochastic() for v in self.variables]
        self._stochastic_vars=[v for v in self.variables if v.stochastic()]
    def sample(self,parent_sample):
        return [v.sample(parent_sample) for v in self.variables]

    def relative_sample(self,parent_sample):
        return [v.relative_sample(parent_sample) for v in self.variables]

    def stochastic(self):
        return self._stochastic
    def get_stochastic_var(self):
        return self._stochastic_vars

class DummyStochasticVariable(StochasticVariable):
    def __init__(variable: StochasticVariable):
        super().__init__(variable.name,variable.size,variable.func)

    def set_value(self,value):
        self._value=value
    def sample(self,parent_sample):
        pass

#maybe add method to generate trained samples if they are turned on
class GMMVariable(VectorVariable):
    #the GMM should be trained on vector [non_cond_vars,cond_vars]
    #for the GMM the names are a list names of each variable it generates when sampled
    #the lengths are an array of ints indicating the non_cond var length
    def __init__(self,non_cond_vars:List[DummyStochasticVariable],gmm: GMM, cond_vars:List[Variable]):
        super().__init__("",non_cond_vars)
        self.gmm=gmm
        self.cond_names=[v.name for v in cond_vars]
        self.non_cond_lengths=[v.size for v in non_cond_vars]
        #this is for calculating the edges of the vector to be return in relative value
        self.non_cond_lengths.insert(0,0)
        self.size=sum([v.size for v in non_cond_vars])
        #create dummy variable list

    #when sampling, sample from a the distribution completely conditioned on the parent independent variable values
    #values on which is trained of course

    #group the values according the vector structure of the search space (given by dummy variables)
    def sample(self, parent_sample):
        #flatten list for calculation of cond distr
        cond_x=np.flatten([parent_sample.independent_vars[name] for name in self.cond_names])
        #the last X are cond attributes
        cond_indices=np.range(self.size,self.gmm.n_components)
        #maybe cache the gmm cond if ithe value of cond_x has already been conditioned
        non_cond_values=self.gmm.cond(cond_indices,cond_x).sample(1)
        self._value=[non_cond_values[l1:l2] for l1,l2 in ut.pairwise(self.non_cond_lengths)]
        map(DummyStochasticVariable.set_value,self.variables,non_cond_values)
        return super().sample()




class MarkovTreeNode:

    class NodeSample:


        #parent_sample: NodeSample
        def __init__(self,node,index:int,parent_sample):
            #necessary to be able to retrieve stochastic vars
            #properties are saved independent from the parent, for later use of its values in machine learning
            self.name=node.name
            self.independent_vars={}
            self.relative_vars={}
            self.stochastic_vars={}
            self.children={}
            for var in node.variables:
                value=var.sample(parent_sample)
                rel_value=var.relative_sample(parent_sample)

                if var.unpack:
                    for v,rel_v,name,stoch in zip(value,rel_value,var.unpack_names,var.stochastic()):
                        self.independent_vars[name]=v
                        self.relative_vars[name]=rel_v
                        if stoch:
                            self.stochastic_vars[name]=v
                else:
                    self.independent_vars[var.name]=value
                    self.relative_vars[var.name]=rel_value
                    if var.stochastic:
                        self.stochastic_vars[var.name]=value

            #save tree structure in 2 ways
            self.parent=parent_sample
            #the index is a path of indices from root to curent sample, for printing
            if parent_sample:
                self.index=str(index) + parent_sample.index
            else:
                self.index=str(index)
        def get_children(self,name):
            return self.children[name]

        def add_children(self,children):
            self.children[children[0].name]=children

        def __str__(self):
            return "\n".join(key + ": " + str(value) for key, value in vars(self).items())

    #TODO find elegant way to swap back to untrained distr
    def remove_vars(self,var_names):
        self.old_vars=[]
        for name in var_names:
            var=self.get_variable(name)
            self.old_vars.append(var)
            self.variables.remove(var)

    def swap_distribution(self, var: Variable):
        if var.unpack:
            self.remove_vars(var.unpack_names)
        else:
            self.remove_vars([var.name])
        self.variables.append(var)

    def get_variable(self,name):
        for var in self.variables:
            if var.name ==name:
                return var

    def get_child_node(self,name):
        for child in self.children:
            if child[1].name==name:
                return child[1]
    #TODO allow tree structure for nodes
    #sample a layout object and its children
    def sample(self,parent_sample=None,index=0,sample_list=None):
        #create  sample
        sample = MarkovTreeNode.NodeSample(self,index,parent_sample=parent_sample)
        if sample_list != None:
            sample_list.append(sample)
        child_samples=[]
        for n_var,child_node in self.children:
            if child_node:
                n_children=n_var.sample(parent_sample)
                for i in range(n_children):
                    #sample child object given parent sample
                    child_samples.append(child_node.sample(sample,i,sample_list))
                sample.add_children(child_samples)
        return sample

    #the structure of the variables is important for later mapping from search space sample to actual
    #generated content
    leaf_children = [(DeterministicVariable("None",0)),None]
    #children: List(Variable,MarkovTreeNode)
    #variables: VectorVariable
    def __init__(self,name,variables ,children=leaf_children):
        self.name=name
        self.variables = variables
        self.children=children


class LayoutMarkovTreeNode(MarkovTreeNode):
    shape_exteriors={"square": VectorVariable("shape",
                                              VectorVariable.from_deterministic_list
                                              ("p",[(0, 0), (0, 1), (1, 1),(1,0)]))}

    position_rel= lambda p , position: [p[0]+position[0],p[1]+position[1]]
    rotation_rel = lambda r, rotation: r + rotation
    size_rel = lambda s, size: s* size

    default_origin=DeterministicVariable("origin",(0.5,0.5))
    #shape should be normalised in the unit cube, same counts for the origin
    #TODO create factory method for layout node with named arguments for the variables in constru
    def __init__(self, name, origin,position, rotation, size,shape,color,children=[(0,None)]):
        super().__init__(name,[origin,position,rotation,size,shape,color],children)








