# -*- coding: utf-8 -*-

#to compare performance between learned models the actual fitness should be used not the fitness with order that used in traing
#also compare expressiveness of a GMM variable by estimating it's surface coverage (99%)
import util.data_format as dtfr
import numpy as np
import copy
import learning.data_generation as dg
import util.utility as ut


class ModelEvaluation:
    def __init__(self,n_data,parent_var_names,parental_fitness,
                                child_name,sibling_fitness,sibling_var_names,fitness_average_threshhold,fitness_func_threshhold):
        self.fitness_average_threshhold=fitness_average_threshhold
        self.fitness_func_threshhold=fitness_func_threshhold
        self.n_data=n_data
        self.parent_var_names=parent_var_names

        #copy and set order to 1 for all fitness funcs
        self.parental_fitness=[copy.deepcopy(fn) for fn in parental_fitness]
        self.sibling_fitness=[copy.deepcopy(fn) for fn in sibling_fitness]
        for fn in self.parental_fitness+self.sibling_fitness:
            fn.order=1

        self.child_name=child_name
        self.sibling_var_names=sibling_var_names


    #calculate emperical performance and expressiveness
    #compare per fitness func for perfomance and per sibling variable for expressiveness
    def evaluate(self,parent_node,parental_fitness,sibling_fitness,expressive_performance_ratio=[1,1]):

        #unfreeze model
        parent_node.freeze_n_children(self.child_name,None)
        parent_samples = parent_node.sample(self.n_data,expressive=False)
        sibling_samples= [ps.children[self.child_name]for ps in parent_samples]
        #performance

        #calc fitness, without order
        fitness=[dg._fitness_calc(ps,self.parental_fitness,
                                  ss,self.sibling_fitness) for ps,ss in zip(parent_samples,sibling_samples)]
        #there are different ways of comparing fitness, for now we choose average of product combined fitness
        fitness_average=dtfr.format_generated_fitness(fitness,
                                                      parental_fitness,sibling_fitness,
                                                      (dtfr.FitnessInstanceDim.single,
                                                                  dtfr.FitnessFuncDim.single),
                                                                  dtfr.FitnessCombination.average)
        #expressiveness

        #calculate weighted variance between sibling vars
        #calculate 1 fitness per sibling as weight

        fitness_siblings=dtfr.format_generated_fitness(fitness,
                                                       parental_fitness,sibling_fitness,
                                                       (dtfr.FitnessInstanceDim.parent_sibling,
                                                                  dtfr.FitnessFuncDim.single),
                                                                  dtfr.FitnessCombination.average)
        #calc per variable it's weighted variance
        sibling_var_variance=[]
        for var_name in self.sibling_var_names:
            for siblings,fitness in zip(sibling_samples,fitness_siblings):
                sibling_var_variance.append(np.var([siblings[i].values["ind"][var_name]/fitness[i] for i in range(len(siblings))]))

        ratio=expressive_performance_ratio[0]*np.average(sibling_var_variance) + expressive_performance_ratio[1]*np.average(fitness_average)
        return ratio


    def converged(self,fitness_values,verbose=True):
        fitness_average=dtfr.format_generated_fitness(fitness_values,
                                                      self.parental_fitness,self.sibling_fitness,
                                                      (dtfr.FitnessInstanceDim.single,
                                                                  dtfr.FitnessFuncDim.single),
                                                                  dtfr.FitnessCombination.average)
        fitness_func_average=dtfr.format_generated_fitness(fitness_values,
                                                           self.parental_fitness,self.sibling_fitness,
                                                           (dtfr.FitnessInstanceDim.parent_children,
                                                                  dtfr.FitnessFuncDim.seperate),
                                                                  dtfr.FitnessCombination.average)
        fitness_average=np.average(fitness_average,None)
        fitness_func_average=np.average(fitness_func_average,0)
        print(fitness_average)
        print(fitness_func_average)
        if fitness_average>self.fitness_average_threshhold or any(fn>self.fitness_func_threshhold
                                                                    for fn in fitness_func_average):
            if verbose:
                print("Convergence reached, further training would lead be numerically unstable.")
            return True
        return False