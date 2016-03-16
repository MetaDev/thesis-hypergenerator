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
import warnings

import rx
from rx import Observable, Observer
from itertools import combinations
from rx.subjects import Subject
from copy import deepcopy

class Variable():
    #name can be either a string or a list of string
    #size is an int that defines the size of the variable
    #the func is a function that is applied on each element
    def __init__(self, name,func=None):
        self.name=name
        self.parent_vars_name=None
        self.unpack=False
        self.set_func(func)
    #these samples are for conditional relations
    def sample(self, parent_sample,sibling_sample):
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
            parent_var_values=[parent_sample.relative_vars[var_name]
            for var_name in self.parent_vars_name]
            return self.func(sample,*parent_var_values)
        else:
            return sample

    #return if is stochastic
    def stochastic(self):
        pass


class StochasticVariable(Variable):

    #low and high is ignored if options (enums) is set
    #if nr_of_values is -1 the distribution is continous
    def __init__(self,name,size=1,func=None,low=0,high=1,step=None):
        super().__init__(name,func)
        self.size=size
        self.high=high
        self.low=low
        #Discrete
        if step is not None:
            self.choices=np.arange(low,high,step)
        else:
            #continuous
            self.distr=scipy.stats.uniform(loc=low,scale=high-low)


    def sample(self,parent_sample,sibling_sample):
        if hasattr(self,"choices"):
            return np.random.choice(self.choices,size=self.size)
        else:
            return self.distr.rvs(size=self.size)

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
    def sample(self, parent_sample,sibling_sample):
        return self._value

class VectorVariableUtility():
    #using the naming convention of a vector variable p1, p2,...
    #to extract a vector variable in its correct order from the flattened dict of variable values
    #in a SampleNode
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


#P(X|Y)
#maybe add method to generate trained samples if they are turned on
class GMMVariable(StochasticVariable):
    #the GMM should be trained on vector [non_cond_vars,cond_vars]
    #for the GMM the names are a list names of each variable it generates when sampled
    #the lengths are an array of ints indicating the non_cond var length
    #TODO check validity of GMM, its dimension against the given X and Y vars
    def __init__(self,name,gmm: GMM,X_vars:List[DummyStochasticVariable], Y_vars_parent:List[Variable],Y_vars_siblings,sibling_order):
        #non_cond_vars are save in attribute variables of VectorVariable
        Variable.__init__(self,name)
        self.unpack_names=[v.name for v in X_vars]
        self.unpack=True
        self.X_vars=X_vars
        self.gmm=gmm
        self.Y_vars_parent=Y_vars_parent
        self.Y_vars_siblings=Y_vars_siblings
        self.X_lengths=[v.size for v in X_vars]
        #this is for calculating the edges of the vector to be return in relative value
        self.X_lengths.insert(0,0)
        self.size=sum(self.X_lengths)
        self.sibling_order=sibling_order
        self.gmm_size=len(gmm._means[0])

        #create dummy variable list

    #when sampling, sample from a the distribution completely conditioned on the parent independent variable values
    #values on which is trained of course

    #group the values according the vector structure of the search space (given by dummy variables)
    def sample(self, parent_sample,sibling_samples):
        #flatten list for calculation of cond distr
        #parent condition variable
        Y_parent_values=np.array([parent_sample.independent_vars[var.name] for var in self.Y_vars_parent]).flatten()
        #sibling condition variable
        #TODO
        Y_sibling_values=[]
        for sibling in sibling_samples:
            Y_sibling_values.extend(np.array([sibling.independent_vars[var.name] for var in self.Y_vars_siblings]).flatten())
        Y_values=np.hstack((Y_sibling_values,Y_parent_values))
        #the last X are cond attributes
        Y_indices=np.arange(self.size,self.gmm_size)
        #maybe cache the gmm cond if ithe value of cond_x has already been conditioned
        print(Y_indices,Y_values,self.sibling_order)
        X_values=self.gmm.condition(Y_indices,Y_values).sample(1)[0]
        return [X_values[l1:l2] for l1,l2 in ut.pairwise(self.X_lengths)]



    def relative_sample(self,samples,parent_sample):
        return [var.relative_sample(value,parent_sample) for var,value in zip(self.X_vars,samples)]

class DefinitionNode:

    class SampleNode:
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


        def get_flat_list(self,sample_list=None):
            if not sample_list:
                sample_list=[]
            sample_list.append(self)
            for child_name in self.children.keys():
                for child in self.get_children(child_name,activity=True):
                    child.get_flat_list(sample_list)
            return sample_list
        def sample(self):
            #check if index not larger than sampled number of childre
            if self.parent_sample and self.index>=self.parent_sample.relative_vars[self.name]:
                print(self.index)
                self.active=False
                return
            self.active=True

            if self.parent_sample:
                self.id=self.parent_sample.id + str(self.index)
            else:
                self.id=str(self.index)
             #first sample the packed variables
            for var in self.sample_variables:
                if var.unpack:
                    value=var.sample(self.parent_sample,self.sibling_samples)
                    rel_value=var.relative_sample(value,self.parent_sample)
                    for v,rel_v,name in zip(value,rel_value,
                                                  var.unpack_names):
                        self.independent_vars[name]=v
                        self.relative_vars[name]=rel_v
                        self.stochastic_vars[name]=v
            #than iterate unpacked variables, which possibly override previous variables
            #from a packed variable
            for var in self.sample_variables:
                if not var.unpack:
                    value=var.sample(self.parent_sample,self.sibling_samples)
                    rel_value=var.relative_sample(value,self.parent_sample)
                    self.independent_vars[var.name]=value
                    self.relative_vars[var.name]=rel_value
                    if var.stochastic():
                        self.stochastic_vars[var.name]=value
            for children in self.children.values():
                for child in children:
                    child.sample()

        #parent_sample: SampleNode
        def __init__(self,node,index:int,parent_sample=None,sibling_samples=None):
            #necessary to be able to retrieve stochastic vars
            #properties are saved independent from the parent, for later use of its values in machine learning
            self.name=node.name
            #variables is a list of all variables, like an archive
            self.variables=[v for v in node.variables.values()]

            #keep track which variable is assigned in the node
            #when constructing the assignment is 1-1
            self.variable_assignment=dict(node.variables)
            #set from which can be sampled
            self.sample_variables=frozenset(self.variable_assignment.values())

            self.children={}
            self.parent_sample=parent_sample
            self.sibling_samples=sibling_samples
            self.index=index
            #variable value dictionaries
            self.independent_vars={}
            self.relative_vars={}
            self.stochastic_vars={}
            self.active=False
        #returns iterator not list
        #activity set to true returns only active children activity to false all children
        def get_children(self,name,activity=False,as_list=False):
            #only return active children
            iterator = filter(lambda child: child.active is True or not activity, self.children[name])
            if as_list:
                return list(iterator)
            return iterator

        def add_children(self,children):
            #add children based on index
            self.children[children[0].name]=[None]*len(children)
            for child in children:
                self.children[children[0].name][child.index]=child

        def __str__(self):
            #TODO
            return "\n".join(key + ": " + str(value) for key, value in vars(self).items())





    def get_child(self,name):
        if name in self.children:
            return self.children[name]
    def build_tree(self,index=0,parent_sample=None,sibling_samples=None):
        node_sample = DefinitionNode.SampleNode(self,index,parent_sample,sibling_samples)
        for distr_name,child_node in self.children.items():
            child_samples=[]
            max_child=self.variables[child_node.name].high
            for i in range(max_child):
                print(i,child_samples)
                #give a copy of the sibling list, because it is different for each child
                child_samples.append(child_node.build_tree(i,
                                                           node_sample,list(child_samples)))
            node_sample.add_children(child_samples)
        return node_sample





    #the structure of the variables is important for later mapping from search space sample to actual
    #generated content
    #children: List(Variable,DefinitionNode)
    #variables: List[Variable]-> be aware that each variable can be a VectorVariable which needs to be unpacked, but it's only allowed level deep, not vector variables of vectorvariables
    #it is not possible to unpack vector variables when initialising because the vector variable that
    #contains learned distributions might not be seperable (has to be sampled simultanously)
    def __init__(self,name,variables:List[Variable] ,children=[]):
        self.node_sample=None
        self.name=name
        self.variables=variables
        #add number of child distributions to variables
        #the name of that distribution needs to be equal
        for c in children:
            c[0].name=c[1].name
            self.variables.append(c[0])
        self.variables=dict([(var.name,var) for var in variables])
        #the children keep the variable as key
        self.children=dict([(c[1].name,c[1])for c in children])





class LayoutDefinitionNode(DefinitionNode):
    shape_exteriors={"square":                                         VectorVariableUtility.from_deterministic_list("shape",[(0, 0), (0, 1), (1, 1),(1,0)])}

    position_rel= lambda p , position: [p[0]+position[0],p[1]+position[1]]
    rotation_rel = lambda r, rotation: r + rotation
    size_rel = lambda s, size: s* size

    default_origin=DeterministicVariable("origin",(0.5,0.5))
    #shape should be normalised in the unit cube, same counts for the origin
    def __init__(self, name,position, rotation, size,shape,color, origin=None,children=[]):
        position.set_func(LayoutDefinitionNode.position_rel)
        rotation.set_func(LayoutDefinitionNode.rotation_rel)
        if origin==None:
            origin=self.default_origin
        #size.func=self.size_rel
        super().__init__(name,[origin,position,rotation,size,*shape,color],children)








