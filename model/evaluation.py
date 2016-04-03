# -*- coding: utf-8 -*-

#to compare performance between learned models the actual fitness should be used not the fitness with order that used in traing
#also compare expressiveness of a GMM variable by estimating it's surface coverage (99%)
import util.data_format as dtfr
import numpy as np
import copy
import learning.data_generation as dg

class ModelEvaluation:
    def __init__(self,n_data,parent_node,parent_var_names,parental_fitness,
                                child_name,sibling_fitness,sibling_var_names):
        self.n_data=n_data
        self.parent_node=parent_node
        self.parent_var_names=parent_var_names

        #copy and set order to 1 for all fitness funcs
        self.parental_fitness=[copy.deepcopy(fn) for fn in parental_fitness]
        self.sibling_fitness=[copy.deepcopy(fn) for fn in sibling_fitness]
        for fn in self.parental_fitness+self.sibling_fitness:
            fn.order=1

        self.child_name=child_name
        self.sibling_var_names=sibling_var_names


#calculate emperical performance and expressiveness
def evaluate_model(self):

    #unfreeze model
    self.parent_node.freeze_n_children(self.child_name,None)
    parent_samples = self.parent_node.sample(self.n_data,expressive=False)
    sibling_samples= [ps.children[self.child_name]for ps in parent_samples]
    #performance

    #calc fitness, without order
    fitness=[dg._fitness_calc(ps,self.parental_fitness,
                              ss,self.sibling_fitness) for ps,ss in zip(parent_samples,sibling_samples)]
    #there are different ways of comparing fitness, for now we choose average of product combined fitness
    fitness_average=dtfr.format_generated_fitness(fitness,(dtfr.FitnessInstanceDim.single,
                                                              dtfr.FitnessFuncDim.single),
                                                              dtfr.FitnessCombination.average)
    #expressiveness

    #calculate weighted variance between sibling vars
    #calculate 1 fitness per sibling as weight


    return fitness_average

