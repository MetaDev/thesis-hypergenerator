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
from itertools import chain
from copy import deepcopy
import warnings

from abc import ABCMeta, abstractmethod

class Variable(metaclass = ABCMeta):
    #name can be either a string or a list of string
    #size is an int that defines the size of the variable
    #the func is a function that is applied on each element
    def __init__(self, name,func=None):
        self.name=name
        self.parent_vars_name=None
        self.unpack=False
        self.set_func(func)
    #these samples are for conditional relations

    @abstractmethod
    def sample(self, parent_sample,sibling_sample,i,n_samples):
        pass
    def set_func(self,func):
        self.func=func
        if func:
            self.parent_vars_name=func.__code__.co_varnames[1:]
    #the last boolean is to indicate whether the returned value needs to be unpacked
    #this passed sample is a data-flow or functional relation
    def relative_sample(self,sample,parent_sample):
        #if there is a function and the parent has the required attributes
        #todo check this with exceptions
        if self.func and parent_sample:
            parent_var_values=[parent_sample.values["rel"][var_name]
            for var_name in self.parent_vars_name]
            return self.func(sample,*parent_var_values)
        else:
            return sample

    #return if is stochastic
    def stochastic(self):
        pass

class NumberChildrenVariable(Variable):
    def __init__(self,name,low,high,step=1):
        super().__init__(name,None)
        self.size=1
        self.high=high
        self.low=low
        self.choices=np.arange(low,high+1,step)
    def sample(self,parent_sample,sibling_sample,i,n_samples):
        return np.random.choice(self.choices)

class StochasticVariable(Variable):

    def __init__(self,name,low,high,func=None,poisson=True):
        super().__init__(name,func)
        self.size=len(high)
        self.high=np.array(high)
        self.low=np.array(low)
        self.points=None
        if len(low) is not len(high):
            raise ValueError("Dimensions of bounds are not equal.")
        self.poisson=poisson

        if poisson and self.size>3:
            raise ValueError("Poisson disk sampling is only supported up to 3 dimensions")


    def init_poisson(self,n_samples):
        import model.sampling.poisson as poisson
        poisson_generator = poisson.PoissonGenerator(self.size)
        self.points = poisson_generator.find_point_set(n_samples)

    def sample(self,parent_sample,sibling_sample,i,n_samples):
        if self.poisson and n_samples>1:
            if  self.points is None:
                self.init_poisson(n_samples)
            point=self.points[i]
        else:
            point=scipy.stats.uniform.rvs(size=self.size)
        return self.low + point*(self.high-self.low)

    def stochastic(self):
        return True



class DeterministicVariable(Variable):
    def __init__(self,name,value,func=None):
        super().__init__(name,func)
        self._value=value
        self.high=value
        self.low=value
    def stochastic(self):
        return False
    def sample(self, parent_sample,sibling_sample,i,n_samples):
        return self._value

class VectorVariableUtility():
    #using the naming convention of a vector variable p1, p2,...
    #to extract a vector variable in its correct order from the flattened dict of variable values
    #in a Node
    @staticmethod
    def extract_ordered_list_vars_values(list_name,sample_var_dict):
        var_list=[]
        for name,value in sample_var_dict.items():
            if ut.remove_numbers(name)==list_name:
                var_list.append((ut.get_trailing_number(name),value))
        return [var[1] for var in sorted(var_list,key=itemgetter(0))]

    @staticmethod
    def extract_ordered_list_vars(list_name,all_vars):
        var_list=[]
        for var in all_vars:
            if ut.remove_numbers(var.name)==list_name:
                var_list.append((ut.get_trailing_number(var.name),var))
        return [var[1] for var in sorted(var_list,key=itemgetter(0))]
    @staticmethod
    def from_deterministic_list(list_name:str,list_):
        return [DeterministicVariable(list_name+str(i),list_[i]) for i in range(len(list_))]
    @staticmethod
    def from_variable_list(list_name,variables:List[Variable]):
        for var,i in zip(variables,np.arange(len(variables))):
            var.name=list_name + str(i)
        return variables

    #name indicates the collection of variables, the name of each variable will be overwritten


class DummyStochasticVariable(StochasticVariable):
    def __init__(self,variable: StochasticVariable):
        Variable.__init__(self,variable.name,variable.func)
        self.size=variable.size

import util.data_format as dtfr

#P(X|Y)
class GMMVariable(StochasticVariable):
    #for the GMM the names are a list names of each variable it generates when sampled
    #the lengths are an array of ints indicating the non_cond var length
    #TODO check validity of GMM, its dimension against the given X and Y vars
    def __init__(self,name,gmm: GMM,child_vars:List[Variable], parent_vars:List[Variable],sibling_vars,sibling_order):
        #non_cond_vars are save in attribute variables of VectorVariable
        Variable.__init__(self,name)
        self.child_vars=child_vars
        self.unpack_names=[v.name for v in child_vars]
        self.unpack=True
        self.sibling_vars=sibling_vars
        self.gmm=gmm
        self.parent_vars=parent_vars
        self.sibling_order=sibling_order

        #create dummy variable list

    #when sampling, sample from a the distribution completely conditioned on the parent independent variable values
    #values on which is trained of course
    #group the values according the vector structure of the search space (given by dummy variables)
    def sample(self, parent_sample,sibling_samples,i,n_samples):
        values,indices= dtfr.format_data_for_conditional(parent_sample,self.parent_vars,
                                                         sibling_samples,self.sibling_vars,
                                                         self.sibling_order)

        #maybe cache the gmm cond if ithe value of cond_x has already been conditioned
        cond_values=self.gmm.condition(indices,values).sample(1)[0]
        #split the data obtained from the joint distribution back into an array of variables
        return dtfr.split_variables(self.sibling_vars,cond_values)


    def relative_sample(self,samples,parent_sample):
        return [var.relative_sample(value,parent_sample) for var,value in zip(self.child_vars,samples)]

class TreeDefNode:

    class VarDefNode:
        class SampleNode:

            def __init__(self,var_def_node,values):
                self.index=var_def_node.index
                self.var_def_node=var_def_node
                self.values=values
                self.children=None
                self.name=self.var_def_node.name

            def get_flat_list(self,sample_list=None):
                if not sample_list:
                    sample_list=[]
                sample_list.append(self)
                for children in self.children.values():
                    for child in children:
                        child.get_flat_list(sample_list)
                return sample_list

        def set_learned_variable(self, gmmvar: GMMVariable):
            #set variables that will be replaced to unactive
            for name in gmmvar.unpack_names:
                self.variable_assignment[name]=gmmvar
            self.variables.append(gmmvar)
            #update set of unique variables
            self.sample_variables=frozenset(self.variable_assignment.values())

        def get_variable(self,name):
            var=self.variable_assignment[name]
            if var.unpack:
                 warnings.warn("The variable "+ var.name + "assigned to " + name+
                 " is packed, possibly used to sample multiple variables",RuntimeWarning)
            return var


        def sample(self,n_samples):
            sample_roots=[]
            for i in range(n_samples):
                sample_roots.append(self._sample(i,n_samples))
            return sample_roots
        #recursively sample the tree
        def _sample(self,i,n_samples,parent_sample=None,sibling_samples=None):
            values={}
            values["ind"]={}
            values["rel"]={}
            values["stoch"]={}
            n_siblings=1
            if parent_sample:
                n_siblings=parent_sample.values["rel"][self.name]
            #check if index not larger than sampled number of childre
            if self.index>=n_siblings:
                return

             #first sample the packed variables
            for var in self.sample_variables:
                if var.unpack:
                    value=var.sample(parent_sample,sibling_samples,i,n_samples)
                    rel_value=var.relative_sample(value,parent_sample)
                    for v,rel_v,name in zip(value,rel_value,
                                                  var.unpack_names):
                        values["ind"][name]=v
                        values["rel"][name]=rel_v
                        if var.stochastic():
                            values["stoch"][name]=v
            #than iterate unpacked variables, which possibly override previous variables
            #from a packed variable
            for var in self.sample_variables:
                if not var.unpack:
                    value=var.sample(parent_sample,sibling_samples,i,n_samples)
                    rel_value=var.relative_sample(value,parent_sample)
                    values["ind"][var.name]=value
                    values["rel"][var.name]=rel_value
                    if var.stochastic():
                        values["stoch"][var.name]=value
            child_samples={}
            #convert var def node to actual sample node
            sample=self.SampleNode(self,values)
            #TODO check ordering of children
            for child_nodes in self.children.values():
                child_samples[child_nodes[0].name]=[]
                for child in child_nodes:
                    n_children=values["rel"][child.name]
                    #only sample the amount of children as sampled
                    if child.index < n_children:
                        siblings=list(child_samples[child.name])
                        child_sample=child._sample(i,n_samples,sample,siblings)
                        child_samples[child.name].append(child_sample)
            #relative values are calculated based on parent, thus the children have to be instantiated before the parent
            sample.children=child_samples
            return sample

        #parent_sample: Node
        def __init__(self,node,index:int,children):
            #necessary to be able to retrieve stochastic vars
            #properties are saved independent from the parent, for later use of its values in machine learning
            self.name=node.name
            #variables is a list of all variables, like an archive
            #make deep copy of variable as each vardefnode should have unique vars
            self.variables=[deepcopy(v) for v in node.variables.values()]
            #keep track which variable is assigned in the node
            #when constructing the assignment is 1-1
            self.variable_assignment=dict([(v.name,v) for v in self.variables])
            #set from which can be sampled
            self.sample_variables=frozenset(self.variable_assignment.values())
            #children are ordered on index
            self.children=children
            self.index=index



        def __str__(self):
            #TODO
            return "\n".join(key + ": " + str(value) for key, value in vars(self).items())


    def get_child(self,name):
        if name in self.children:
            return self.children[name]
    def build_tree(self,index=0):
        children={}
        for distr_name,child_node in self.children.items():
            max_child=self.variables[child_node.name].high
            children[child_node.name]=[]
            #defines order of children
            for i in range(max_child):
                #give a copy of the sibling list, because it is different for each child
                children[child_node.name].append(child_node.build_tree(i))
        return TreeDefNode.VarDefNode(self,index,children)





    #the structure of the variables is important for later mapping from search space sample to actual
    #generated content
    #children: List(Variable,TreeDefNode)
    #variables: List[Variable]-> be aware that each variable can be a VectorVariable which needs to be unpacked, but it's only allowed level deep, not vector variables of vectorvariables
    #it is not possible to unpack vector variables when initialising because the vector variable that
    #contains learned distributions might not be seperable (has to be sampled simultanously)
    def __init__(self,name,variables:List[Variable] ,children=[]):
        self.node_sample=None
        self.name=name
        #add number of child distributions to variables
        #the name of that distribution needs to be equal
        for c in children:
            c[0].name=c[1].name
            variables.append(c[0])

        self.variables=dict([(var.name,var) for var in variables])
        #the children keep the variable as key
        self.children=dict([(c[1].name,c[1])for c in children])

    def max_children(self,child_name):
        return self.variables[child_name].high



class LayoutTreeDefNode(TreeDefNode):
    shape_exteriors={"square":                                         VectorVariableUtility.from_deterministic_list("shape",[(0, 0), (0, 1), (1, 1),(1,0)])}

    position_rel= lambda p , position: [p[0]+position[0],p[1]+position[1]]
    rotation_rel = lambda r, rotation: r + rotation
    size_rel = lambda s, size: s* size

    default_origin=DeterministicVariable("origin",(0.5,0.5))
    #shape should be normalised in the unit cube, same counts for the origin
    def __init__(self, name,position, rotation, size,shape,color, origin=None,children=[]):
        position.set_func(LayoutTreeDefNode.position_rel)
        rotation.set_func(LayoutTreeDefNode.rotation_rel)
        if origin==None:
            origin=self.default_origin
        #size.func=self.size_rel
        super().__init__(name,[origin,position,rotation,size,*shape,color],children)








