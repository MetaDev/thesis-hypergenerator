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

class Variable():
    #name can be either a string or a list of string
    #size is an int that defines the size of the variable
    #the func is a function that is applied on each element
    def __init__(self, name,func=None):
        self.name=name
        self.parent_vars_name=None
        self.unpack=False
        self.func=func
        if func:
            self.parent_vars_name=func.__code__.co_varnames[1:]
    def sample(self, parent_sample):
        pass

    #the last boolean is to indicate whether the returned value needs to be unpacked
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
    @staticmethod
    def standard_distr(low=0,high=1):
        return scipy.stats.uniform(loc=low,scale=high-low)
    @staticmethod
    def standard_choices(low=0,high=1,step=1):
        return np.arange(low,high,step)
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
            return np.random.choice(self.choices,size=self.size)
        else:
            return self.distr.rvs(size=self.size)

    def stochastic(self):
        return True




class DeterministicVariable(Variable):
    def __init__(self,name,value,func=None):
        super().__init__(name,func)
        self._value=value

    def stochastic(self):
        return False
    def sample(self, parent_sample):
        return self._value

#TODO change name in Variable utility,  add factory method for creating vector vars
class VectorVariableUtility():
    #using the naming convention of a vector variable p1, p2,...
    #to extract a vector variable in its correct order from the flattened dict of variable values
    #in a nodesample
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
    def __init__(self,name,gmm: GMM,X_vars:List[DummyStochasticVariable], Y_vars:List[Variable]):
        #non_cond_vars are save in attribute variables of VectorVariable
        Variable.__init__(self,name)
        self.unpack_names=[v.name for v in X_vars]
        self.unpack=True
        self.X_vars=X_vars
        self.gmm=gmm
        self.Y_vars=Y_vars
        self.X_lengths=[v.size for v in X_vars]
        #this is for calculating the edges of the vector to be return in relative value
        self.X_lengths.insert(0,0)
        self.size=sum(self.X_lengths)
        self.gmm_size=sum([v.size for v in Y_vars])+sum(self.X_lengths)
        #create dummy variable list

    #when sampling, sample from a the distribution completely conditioned on the parent independent variable values
    #values on which is trained of course

    #group the values according the vector structure of the search space (given by dummy variables)
    def sample(self, parent_sample):
        #flatten list for calculation of cond distr
        Y_values=np.array([parent_sample.independent_vars[var.name] for var in self.Y_vars]).flatten()
        #the last X are cond attributes
        Y_indices=np.arange(self.size,self.gmm_size)
        #maybe cache the gmm cond if ithe value of cond_x has already been conditioned
        X_values=self.gmm.condition(Y_indices,Y_values).sample(1)[0]
        return [X_values[l1:l2] for l1,l2 in ut.pairwise(self.X_lengths)]



    def relative_sample(self,samples,parent_sample):
        return [var.relative_sample(value,parent_sample) for var,value in zip(self.X_vars,samples)]

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
            #first sample the packed variables
            for var in node.sample_variables:
                if var.unpack:
                    value=var.sample(parent_sample)
                    rel_value=var.relative_sample(value,parent_sample)
                    for v,rel_v,name in zip(value,rel_value,
                                                  var.unpack_names):
                        self.independent_vars[name]=v
                        self.relative_vars[name]=rel_v
                        self.stochastic_vars[name]=v
            #than iterate unpacked variables, which possibly override previous variables
            #from a packed variable
            for var in node.sample_variables:
                if not var.unpack:
                    value=var.sample(parent_sample)
                    rel_value=var.relative_sample(value,parent_sample)
                    self.independent_vars[var.name]=value
                    self.relative_vars[var.name]=rel_value
                    if var.stochastic():
                        self.stochastic_vars[var.name]=value
            #save tree structure in 2 ways
            self.parent=parent_sample
            #the index is a path of indices from root to curent sample, for printing
            if parent_sample:
                self.index=parent_sample.index +str(index)
            else:
                self.index=str(index)
        def get_children(self,name):
            return self.children[name]

        def add_children(self,children):
            self.children[children[0].name]=children

        def __str__(self):
            #TODO
            return "\n".join(key + ": " + str(value) for key, value in vars(self).items())

    #TODO make it possible to "freeze' a stochastic variable at a certain position
    #TODO add method to request all stochastic variables

    #TODO make it possible to switch back to original prior distribution for a named var

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
             warnings.warn("The variable "+ var.name + "assigned to " + name+ " is packed, possibly used to sample multiple variables",
                           RuntimeWarning)
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
        for n_var,child_node in self.children:
            if child_node:
                child_samples=[]
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
    #variables: List[Variable]-> be aware that each variable can be a VectorVariable which needs to be unpacked, but it's only allowed level deep, not vector variables of vectorvariables
    #it is not possible to unpack vector variables when initialising because the vector variable that
    #contains learned distributions might not be seperable (has to be sampled simultanously)
    def __init__(self,name,variables:List[Variable] ,children=leaf_children):
        self.name=name
        self.variables=variables

        self.variable_assignment={}
        #when constructing the assignment is 1-1
        for var in self.variables:
            self.variable_assignment[var.name]=var
        #set from which can be sampled
        self.sample_variables=frozenset(self.variable_assignment.values())
        self.children=children


class LayoutMarkovTreeNode(MarkovTreeNode):
    shape_exteriors={"square":                                         VectorVariableUtility.from_deterministic_list("shape",[(0, 0), (0, 1), (1, 1),(1,0)])}

    position_rel= lambda p , position: [p[0]+position[0],p[1]+position[1]]
    rotation_rel = lambda r, rotation: r + rotation
    size_rel = lambda s, size: s* size

    default_origin=DeterministicVariable("origin",(0.5,0.5))
    #shape should be normalised in the unit cube, same counts for the origin
    #TODO create factory method for layout node with named arguments for the variables in constru
    def __init__(self, name, origin,position, rotation, size,shape,color,children=[(0,None)]):
        super().__init__(name,[origin,position,rotation,size,*shape,color],children)








