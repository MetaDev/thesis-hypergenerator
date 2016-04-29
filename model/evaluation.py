# -*- coding: utf-8 -*-

#to compare performance between learned models the actual fitness should be used not the fitness with order that used in traing
#also compare expressiveness of a GMM variable by estimating it's surface coverage (99%)
import util.data_format as dtfr
import numpy as np
import copy
import learning.data_generation as dg
import util.utility as ut
import model.fitness as fn


class ModelEvaluation:
    def __init__(self,n_data,
                                parent_tree_def,parent_node,parent_var_names,
                                child_name,sibling_var_names,
                                fitness_funcs,
                                fitness_average_threshhold,fitness_func_threshhold):
        self.fitness_average_threshhold=fitness_average_threshhold
        self.fitness_func_threshhold=fitness_func_threshhold
        self.n_data=n_data
        self.parent_var_names=parent_var_names
        self.parent_tree_def=parent_tree_def
        self.parent_node=parent_node
        #copy and set order to 1 for all fitness funcs
        self.fitness_funcs=[copy.deepcopy(fn) for fn in fitness_funcs]
        for fitn in self.fitness_funcs:
            fitn.order=1

        self.child_name=child_name
        self.sibling_var_names=sibling_var_names


    #calculate emperical performance and expressiveness
    #compare per fitness func for perfomance and per sibling variable for expressiveness
    def evaluate(self,expressive_performance_ratio=[1,1]):

        #unfreeze model
        self.parent_node.freeze_n_children(self.child_name,None)
        parent_samples = self.parent_node.sample(self.n_data,expressive=False)
        sibling_samples= [ps.children[self.child_name]for ps in parent_samples]
        #performance

        #calc fitness, without order
        fitness_values=[fn.fitness_calc(ps,
                                  ss,self.fitness_funcs) for ps,ss in zip(parent_samples,sibling_samples)]
        #there are different ways of comparing fitness, for now we choose average of product combined fitness
        fitness_average=dtfr.reduce_fitness_dimension(fitness_values,
                                                      (dtfr.FitnessInstanceDim.single,
                                                                  dtfr.FitnessFuncDim.single),
                                                                  dtfr.FitnessCombination.average)


        #expressiveness


        #calculate normalised weighted variance between sibling vars

        #calc per variable it's weighted variance
        sibling_var_variance=[]
        _,max_siblings=self.parent_tree_def.variable_range(self.child_name)

        for var_name in self.sibling_var_names:
            variables=[]
            for siblings,fitness in zip(sibling_samples,fitness_average):
                variables.append([siblings[i].get_normalised_value(var_name)*fitness for i in range(len(siblings))])

            #calculate variance per variable and sibling
            #not every set of siblings has the same length, wich complicates the calculation
            var_length=self.parent_tree_def.children[self.child_name].variables[var_name].size

            var_sum=[0]*(max_siblings*var_length)
            var_count=[0]*(max_siblings*var_length)
            for variable in variables:
                for i,var in enumerate(ut.flatten(variable)):
                    var_sum[i]+=var
                    var_count[i]+=1

            var_variance=[0]*(max_siblings*var_length)
            var_mean = np.array(var_sum)/np.array(var_count)
            for variable in variables:
                for i,var in enumerate(ut.flatten(variable)):
                    var_variance[i]+=(var-var_mean[i])**2
            var_variance=np.array(var_variance)/np.array(var_count)
            #the result is the variance of a variable and a specific object, one of the siblings in this case
            sibling_var_variance.append(np.average(var_variance))

        #the expressiveness get's an additional order,calculated using the fitness average, to be able to compare both
        fitn_avg=np.average(fitness_average)
        expr_order=10*(1/fitn_avg)
        ratio=expressive_performance_ratio[0]*np.average(sibling_var_variance)**expr_order + \
                                        expressive_performance_ratio[1]*np.average(fitness_average)
        return ratio

    def print_evaluation(self,fitness_values,iteration,summary=True):
        print("fitness before training iteration ", iteration)

        from model import fitness as fn
        fitness_func_values=dtfr.reduce_fitness_dimension(fitness_values,
                                                                   (dtfr.FitnessInstanceDim.single,
                                                                    dtfr.FitnessFuncDim.seperate),
                                                                    dtfr.FitnessCombination.average)

        print("seperate fitness")
        for i in range(len(self.fitness_funcs)):
            print(self.fitness_funcs[i].name)
            fn.fitness_statistics(np.array(fitness_func_values)[:,i],summary=summary)



    def converged(self,fitness_values,verbose=True):
        fitness_average=dtfr.reduce_fitness_dimension(fitness_values,
                                                      (dtfr.FitnessInstanceDim.single,
                                                                  dtfr.FitnessFuncDim.single),
                                                                  dtfr.FitnessCombination.average)
        fitness_func_average=dtfr.reduce_fitness_dimension(fitness_values,
                                                           (dtfr.FitnessInstanceDim.single,
                                                                  dtfr.FitnessFuncDim.seperate),
                                                                  dtfr.FitnessCombination.average)
        fitness_average=np.average(fitness_average,None)
        fitness_func_average=np.average(fitness_func_average,0)
        if fitness_average>self.fitness_average_threshhold or any(fn>self.fitness_func_threshhold
                                                                    for fn in fitness_func_average):
            if verbose:
                print("Convergence reached, further training would lead be numerically unstable.")
            return True
        return False