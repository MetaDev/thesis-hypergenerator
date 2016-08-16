# -*- coding: utf-8 -*-

import model.test_models as tm

from learning.gmm import GMM

import util.utility as ut

import learning.data_generation as dg

from model import fitness as fn
import util.data_format as dtfr

import learning.train_gmm as tgmm

import numpy as np

import util.visualisation as vis

import model.evaluation as mev


child_name="child"
#model to train on
parent_node,parent_def=tm.model_var_pos(True,True,True)
child_nodes=parent_node.children[child_name]

sibling_data=dg.SiblingData.first

sibling_var_names=["position"]
parent_var_names=[]

n_model_eval_data=200

sibling_vars=[parent_def.children[child_name].variables[name] for name in sibling_var_names]
parent_vars=[parent_def.variables[name] for name in parent_var_names]
if not all( var.stochastic() for var in sibling_vars+parent_vars):
    non_stoch_var=[var.name for var in sibling_vars+parent_vars if not var.stochastic()]
    raise ValueError("Only the distribution of stochastic variables can be trained on. The variables "+
                                                        str(non_stoch_var)+ "are not stochastic")
#check if none of the vars is deterministic


#this expliicitly also defines the format of the data
#fitness func, order cap and regression target
#fitness_funcs=[fn.Targetting_Fitness("Minimum distances",fn.min_dist_sb,fn.Fitness_Relation.pairwise_siblings,1,0,1,target=1),fn.Fitness("polygon overlap",fn.negative_overlap_pc,fn.Fitness_Relation.pairwise_parent_child,1,0,1)]
fitness_funcs=[fn.Targetting_Fitness("Minimum distances",fn.min_dist_sb,fn.Fitness_Relation.pairwise_siblings,1/4,0,0.8,target=1)]
fitness_dim=(dtfr.FitnessInstanceDim.single,dtfr.FitnessFuncDim.single)
data,fitness_values=dg.training_data_generation(200,parent_def,
                                parent_node,parent_var_names,
                                child_name,sibling_var_names,2,
                                fitness_funcs,
                               sibling_data)

data,fitness_values=dtfr.filter_fitness_and_data_training(data,fitness_values,fitness_funcs)

fitness_average_threshhold=0.95
fitness_func_threshhold=0.98

model_evaluation = mev.ModelEvaluation(n_model_eval_data,parent_def,parent_node,parent_var_names,
                                               child_name,sibling_var_names,
                                               fitness_funcs,
                                               fitness_average_threshhold,fitness_func_threshhold)


model_evaluation.print_evaluation(fitness_values,0)
gmm = GMM(n_components=15)

regression=False
if regression:
    #filter
    data,fitness_values=dtfr.filter_fitness_and_data_training(data,fitness_values,
                                                                                  fitness_funcs)
    #apply order
    fitness_values=dtfr.apply_fitness_order(fitness_values,fitness_funcs)

    #reduce
    fitness_regression=dtfr.reduce_fitness_dimension(fitness_values,fitness_dim,
                                                        dtfr.FitnessCombination.product)
    #renormalise
    fitness_regression=dtfr.normalise_fitness(fitness_regression)
    fitness_regression=[ list(ut.flatten(fn_value_line)) for fn_value_line in fitness_regression]

    #add fitness data
    train_data=np.column_stack((data,fitness_regression))

    gmm = GMM(n_components=5)
    #for regression calculate full joint
    gmm.fit(train_data,infinite=False,min_covar=0.01)

    indices,targets = dtfr.format_target_indices_for_regression_conditioning(data,fitness_values,
                                                                             fitness_funcs,
                                                                             fitness_dim)

    #condition on fitness
    gmm= gmm.condition(indices,targets)

else:
    data,fitness_values=dtfr.filter_fitness_and_data_training(data,fitness_values,
                                                                              fitness_funcs)
    #apply order
    fitness_values=dtfr.apply_fitness_order(fitness_values,fitness_funcs)

    #reduce fitness to a single dimension
    fitness_single=dtfr.reduce_fitness_dimension(fitness_values,(dtfr.FitnessInstanceDim.single,
                                                                 dtfr.FitnessFuncDim.single),
                                                                        dtfr.FitnessCombination.product)
    #renormalise
    fitness_single=dtfr.normalise_fitness(fitness_single)

    gmm.fit(data,np.array(fitness_single)[:,0],infinite=False,min_covar=0.01)


#give a conditional position and visualise the result
gmm=gmm.condition([0,1],[1,2])

vis.visualise_gmm_marg_2D_density(gmm,[0,1])

sibling_order_sequence=[0,1,1,1,1,1,1]
gmms=[None]*2
gmms[1]=gmm

#marginalise gmms, starting from the largest
for child_index in reversed(range(2)):
    if not gmms[child_index]:
        gmms[child_index]=dtfr.marginalise_gmm(gmms,child_index,parent_vars,sibling_vars)
gmm_var_name="test"+str(0)
gmm_vars=tgmm._construct_gmm_vars(gmms,gmm_var_name,parent_def,parent_node,child_name,
             parent_var_names,sibling_var_names)

#the gmms are ordered the same as the children
#use sibling order sequence to assign gmms[i] to the a child with order i
#assign variable child i with gmm min(i,sibling_order)
for k in range(len(child_nodes)):
    child_nodes[k].set_learned_variable(gmm_vars[sibling_order_sequence[k]])

for gmm_var in gmm_vars:
    gmm_var.visualise_sampling(["position"])
#sample once
_,max_children=parent_def.variable_range(child_name)
parent_node.freeze_n_children(child_name,max_children)
parent_node.sample(1)
for gmm_var in gmm_vars:
    gmm_var.visualise_sampling(None)

#give fitness mean
model_evaluation.print_evaluation(fitness_values,0)