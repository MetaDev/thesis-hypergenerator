# -*- coding: utf-8 -*-

#to compare performance between learned models the actual fitness should be used not the fitness with order that used in traing
#also compare expressiveness of a GMM variable by estimating it's surface coverage (99%)
import util.data_format as dtfr
import numpy as np

#todo
class ModelEvaluation:
    def __init__(n_data,parent_def,parent_var_names,parental_fitness,
                                child_name,sibling_fitness,sibling_var_names):
        pass

#calculate emperical performance and expressiveness
def evaluate_model(self):

    #unfreeze model
    parent_node.freeze_n_children(child_name,None)
    parent_samples = parent_node.sample(n_data,expressive=False)
    subling_samples=[p.children[child_name] for p in parent_samples]
    #performance
    #calc parent fitness
    #calc siblings fitness
    sibling_fitness=[fn.calc(child0,child1) for fn in self.siblings_fitness
    for child0,child1 in combinations(siblings,2)
    for siblings in siblings_samples]
    #expressivenss
    #calculate weighted variance between sibling vars


    return np.average(fitness_funcs,axis=0)

