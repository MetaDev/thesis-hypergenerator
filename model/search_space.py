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
import util.utility as ut
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
        self.freeze_value=None
    #these samples are for conditional relations

    @abstractmethod
    def sample(self, parent_sample,sibling_sample,i,n_samples,expressive):
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




class StochasticVariable(Variable):

    def __init__(self,name,low,high,func=None):
        super().__init__(name,func)
        self.freeze_value=None
        self.high=np.array(high)
        self.low=np.array(low)
        self.size=ut.size(self.high)
        self.points=None
        if ut.size(self.high) is not ut.size(self.low):
            raise ValueError("Dimensions of bounds are not equal.")


    def init_poisson(self,n_samples):
        import model.sampling.poisson as poisson
        poisson_generator = poisson.PoissonGenerator(self.size)
        self.points = poisson_generator.find_point_set(n_samples)

    def sample(self,parent_sample,sibling_sample,i,n_samples,expressive):
        if expressive and self.size>3:
            raise ValueError("Poisson disk sampling for more expressive result is only supported up to 3 dimensions")
        if expressive and n_samples>1:
            #init poisson disk at the start of sampling
            if i is 0:
                self.init_poisson(n_samples)
            point=self.points[i]
        else:
            point=scipy.stats.uniform.rvs(size=self.size)
        return self.low + np.array(point)*(self.high-self.low) if not self.frozen() else self.freeze_value

    def stochastic(self):
        return True
    def freeze(self,value):
        self.freeze_value=np.array(value)
    def thaw(self):
        self.freeze(None)
    def frozen(self):
        return self.freeze_value



class DeterministicVariable(Variable):
    def __init__(self,name,value,func=None):
        super().__init__(name,func)
        self._value=np.array(value)
        self.size=ut.size(self._value)
        self.high=value
        self.low=value
    def stochastic(self):
        return False
    def sample(self, parent_sample,sibling_sample,i,n_samples,expressive):
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

import util.data_format as dtfr

#P(X|Y)
class GMMVariable(StochasticVariable):
    max_tries=20


    #for the GMM the names are a list names of each variable it generates when sampled
    #the lengths are an array of ints indicating the non_cond var length
    #the output variables of the GMM variable are the sibling_vars
    #TODO check validity of GMM, its dimension against the given X and Y vars
    #also add sibling name, if parent has multiple named siblings these can also be passed
    def __init__(self,name,gmm: GMM, parent_vars:List[Variable],sibling_vars,sibling_order):
        #non_cond_vars are save in attribute variables of VectorVariable
        Variable.__init__(self,name)
        self.unpack=True
        self.sibling_vars=[deepcopy(var) for var in sibling_vars]
        self.unpack_names=[v.name for v in sibling_vars]
        self.var_bounds=[(var.low,var.high) for var in sibling_vars]

        self.gmm=gmm
        self.parent_vars=parent_vars
        self.sibling_order=sibling_order

        #keep track of how sanpling quality, 0 is worst
        self.sample_quality=1

    #when sampling, sample from a the distribution completely conditioned on the parent independent variable values
    #values on which is trained of course
    #group the values according the vector structure of the search space (given by dummy variables)
    def sample(self, parent_sample,sibling_samples,i,n_samples,expressive):
        attempts=0
        found=False
        indices,values= dtfr.format_data_for_conditional(parent_sample,self.parent_vars,
                                                             sibling_samples,self.sibling_vars,
                                                             self.sibling_order)
        gmm_cond=self.gmm.condition(indices,values)
        for i in range(GMMVariable.max_tries):
            attempts+=1
            cond_values=gmm_cond.sample(1)[0]
            #split the data obtained from the joint distribution back into an array of variables
            var_values=dtfr.split_variables(self.sibling_vars,cond_values)
            #check if the values are within bounds
            if all((all(np.greater_equal(var,bounds[0])) and all(np.less_equal(var,bounds[1]))) for var,bounds in zip(var_values,self.var_bounds)):
                found=True
                break
        self.sample_quality=(self.sample_quality+(1-(1-attempts)/(1-GMMVariable.max_tries)))/2
        if not found:
            var_values=[var.sample(parent_sample,sibling_samples,i,n_samples,False) for var in self.sibling_vars]
        return var_values


    def relative_sample(self,samples,parent_sample):
        return [var.relative_sample(value,parent_sample) for var,value in zip(self.sibling_vars,samples)]

    def freeze(self,var_name,value):
        if var_name not in self.unpack_names:
            raise ValueError("Variable to freeze not packed in GMM variable.")
        for var in self.sibling_vars:
            if var.name is var_name:
                var.freeze(value)
    def thaw(self,var_name):
        if var_name not in self.unpack_names:
            raise ValueError("Variable to thaw not packed in GMM variable.")
        for var in self.sibling_vars:
            if var.name is var_name:
                var.freeze(None)
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
            if self.children:
                for children in self.children.values():
                    for child in children:
                        child.get_flat_list(sample_list)
            return sample_list
        def get_normalised_value(self,name):
            low,high= self.var_def_node.tree_def_node.variable_range(name)
            return (self.values["ind"][name]-low)/(high-low)
    #parent_sample: Node
    def __init__(self,tree_def_node,children=None):
        self.index=0
        #properties are saved independent from the parent, for later use of its values in machine learning
        self.name=tree_def_node.name
        self.tree_def_node=tree_def_node
        #variables is a list of all variables, like an archive, these define the structure of the node sample variables
        #make deep copy of variable as each vardefnode should have unique vars
        self.variables=dict([(v.name,deepcopy(v)) for v in tree_def_node.variables.values()])

        #keep track which variable is assigned in the node
        #when constructing the assignment is 1-1
        self.variable_assignment=dict(self.variables)


        self._update_sample_variables()
        #children are ordered on index
        self.children={}
        if children:
            for n_children, child_list in children:
                self.add_children(child_list,n_children)
    #n children is the variable depicting the number of children
    def add_children(self,child_list,n_children):
        #TODO check if all children have the same name
        self.children[n_children.name]=child_list
        self.add_new_variable(n_children)
        for i,child in enumerate(child_list):
            child.index=i

    def _update_sample_variables(self):
        #set containing all variables from which needs to be samples
        self.sample_variables=frozenset(self.variable_assignment.values())

    def add_new_variable(self,variable):
        self.variables[variable.name]=deepcopy(variable)
        self.variable_assignment[variable.name]=variable
        self._update_sample_variables()
    def __str__(self):
            #TODO
            return "\n".join(key + ": " + str(value) for key, value in vars(self).items())

    def delete_learned_variable(self,gmm_name):
        for var in self.sample_variables:
            if var.name is gmm_name:
                #put all original vars back
                for var_name in var.unpack_names:
                    self.variable_assignment[var_name]=self.variables[var_name]

    def set_learned_variable(self, gmmvar: GMMVariable):
        #set variables that will be replaced to unactive
        for name in gmmvar.unpack_names:
            self.variable_assignment[name]=gmmvar
        #update set of unique variables
        self._update_sample_variables()

    def get_variable(self,name):
        var=self.variable_assignment[name]
        if var.unpack:
             warnings.warn("The variable "+ var.name + "assigned to " + name+
             " is packed, possibly used to sample multiple variables",RuntimeWarning)
        return var
    def freeze_n_children(self,child_name,n_children):
        self.variable_assignment[child_name].freeze(n_children)

    def sample(self,n_samples,expressive=False):
        sample_roots=[]
        for i in range(n_samples):
            sample_roots.append(self._sample(i,n_samples,expressive))
        return sample_roots
    #recursively sample the tree
    def _sample(self,i,n_samples,expressive,parent_sample=None,sibling_samples=None):
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
                value=var.sample(parent_sample,sibling_samples,i,n_samples,expressive)
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
                value=var.sample(parent_sample,sibling_samples,i,n_samples,expressive)
                rel_value=var.relative_sample(value,parent_sample)
                values["ind"][var.name]=value
                values["rel"][var.name]=rel_value
                if var.stochastic():
                    values["stoch"][var.name]=value
        #convert var def node to actual sample node
        sample=self.SampleNode(self,values)
        if self.children:
            child_samples={}
            for child_nodes in self.children.values():
                child_samples[child_nodes[0].name]=[]
                for child in child_nodes:
                    #n_children is a continuous variable
                    n_children=values["rel"][child.name]
                    #only sample the amount of children as sampled
                    if child.index < np.round(n_children):
                        siblings=list(child_samples[child.name])
                        child_sample=child._sample(i,n_samples,expressive,sample,siblings)
                        child_samples[child.name].append(child_sample)
            #relative values are calculated based on parent, thus the children have to be instantiated before the parent
            sample.children=child_samples
        return sample

class TreeDefNode:


    def get_child(self,name):
        if name in self.children:
            return self.children[name]

    def build_child_nodes(self,child_n_list):
        children=[]
        if child_n_list:
            for n_children,child_def in child_n_list:
                max_child=n_children.high
                child_list=[]
                #defines order of children
                for i in range(max_child):
                    #give a copy of the sibling list, because it is different for each child
                    child_list.append(child_def.build_child_nodes(None))
                children.append((n_children,child_list))
        return VarDefNode(self,children)

    #the structure of the variables is important for later mapping from search space sample to actual
    #generated content
    #children: List(Variable,TreeDefNode)
    #variables: List[Variable]-> be aware that each variable can be a VectorVariable which needs to be unpacked, but it's only allowed level deep, not vector variables of vectorvariables
    #it is not possible to unpack vector variables when initialising because the vector variable that
    #contains learned distributions might not be seperable (has to be sampled simultanously)
    def __init__(self,name,variables:List[Variable]):
        self.node_sample=None
        self.name=name

        self.variables=dict([(var.name,var) for var in variables])


    def variable_range(self,var_name):
        return (self.variables[var_name].low,self.variables[var_name].high)



class LayoutTreeDefNode(TreeDefNode):
    shape_exteriors={"square":                                         VectorVariableUtility.from_deterministic_list("shape",[(0, 0), (0, 1), (1, 1),(1,0)])}

    position_rel= lambda p , position: p+position
    rotation_rel = lambda r, rotation: r + rotation
    size_rel = lambda s, size: s* size

    default_origin=DeterministicVariable("origin",(0.5,0.5))
    #shape should be normalised in the unit cube, same counts for the origin
    def __init__(self, name,position, rotation, size,shape, origin=None):
#        position.set_func(LayoutTreeDefNode.position_rel)
#        rotation.set_func(LayoutTreeDefNode.rotation_rel)
        if origin==None:
            origin=self.default_origin
        #size.func=self.size_rel
        super().__init__(name,[origin,position,rotation,size,*shape])








