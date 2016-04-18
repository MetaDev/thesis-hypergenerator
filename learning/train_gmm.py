
import numpy as np

import learning.evaluation as ev
from util import setting_values
import model.test_models as tm

from model import fitness as fn
from learning.gmm import GMM

import util.data_format as dtfr
import util.utility as ut

import learning.data_generation as dg
import model.evaluation as mev

def training(n_data=100,n_iter=1,n_trial=1,n_components=15,infinite=False,regression=False,verbose=True):


    #experiment hyperparameters:

    fitness_average_threshhold=0.95
    fitness_func_threshhold=0.98

    #this sequence indicates the order of the markov chain between siblings [1,2]-> second child depends on the first
    #the third on the first and second
    #the first child is always independent
    sibling_order_sequence=[0,1,2,3,4,4,4,4]

    sibling_data=dg.SiblingData.combination
    fitness_dim=(dtfr.FitnessInstanceDim.parent_children,dtfr.FitnessFuncDim.single)

    #the sibling order defines the size of the joint distribution that will be trained
    sibling_order=np.max(sibling_order_sequence)
    n_children=sibling_order+1

    #TODO
    #gmm marginalisation [order_1,order_2,..,order_sibling_order]

    #True->train full joint
    #False->derive (marginalise) from closest higher order

    gmm_full=[False,False,False,False]
    #the largest sibling order always has to be calculated
    gmm_full.append(True)


    child_name="child"
    #model to train on
    parent_node,parent_def=tm.test_model_var_child_position_parent_shape()
    child_nodes=parent_node.children[child_name]

    #training variables and fitness functions
    #this expliicitly also defines the format of the data
    sibling_var_names=["position","rotation"]
    parent_var_names=["shape3"]

    sibling_vars=[parent_def.children[child_name].variables[name] for name in sibling_var_names]
    parent_vars=[parent_def.variables[name] for name in parent_var_names]
    if not all( var.stochastic() for var in sibling_vars+parent_vars):
        raise ValueError("Only the distribution of stochastic variables can be trained on.")
    #check if none of the vars is deterministic


    #this expliicitly also defines the format of the data
    #fitness func, order cap and regression target
    sibling_fitness=[fn.Fitness(fn.fitness_min_dist,4,0,1)]
    #only the func order and cap is used for training

    parental_fitness=[fn.Fitness(fn.fitness_polygon_overl,32,0,1)]

    model_evaluation = mev.ModelEvaluation(100,parent_def,parent_var_names,parental_fitness,
                                               child_name,sibling_fitness,sibling_var_names,
                                               fitness_average_threshhold,fitness_func_threshhold)

    score=model_evaluation.evaluate(parent_node,parental_fitness,sibling_fitness)


    #check sibling sequence
    wrong_sequence = any(sibling_order_sequence[i]>i for i in range(len(sibling_order_sequence)))
    if wrong_sequence:
        raise ValueError("Some orders of the sibling order sequence exceed the number of previous siblings.")
    max_children=parent_def.variable_range(child_name)[1]
    if len(sibling_order_sequence) != max_children:
        raise ValueError("The number of siblings implied by the sibling order sequence can not be different than the maximum number of children in the model.")
    #check marginalisation
    if len(gmm_full) != n_children:
        raise ValueError("the array defining which sibling order to train seperately should have the same length as the maximum amount of children for a given sibling order. \n length array: ",len(gmm_full),", expected: ",n_children)
    #do n_iter number of retrainings using previously best model
    for iteration in range(n_iter):
        #find out the performance of the current model

        #TODO change training data to evaluation
        data,fitness=dg.training_data_generation(n_data,parent_def,
                                parent_node,parent_var_names,parental_fitness,
                                child_name,sibling_fitness,sibling_var_names,n_children=n_children,
                                sibling_data=sibling_data)

        if verbose:
            print_result(fitness,parental_fitness,sibling_fitness,iteration)



        if model_evaluation.converged(fitness):
            return

        #combine fitness per func
        #evaluate model at the start of every iteration
        iteration_gmm_score=model_evaluation.evaluate(parent_node,parental_fitness,sibling_fitness)


        gmm_vars_retry_eval=[]
        #do n trials to find a better gmm for the model
        for trial in range(n_trial):
            #calculate all full joints
            gmms=[None]*n_children
            for child_index in np.where(gmm_full)[0]:
                data,fitness=dg.training_data_generation(n_data,parent_def,
                                parent_node,parent_var_names,parental_fitness,
                                child_name,sibling_fitness,sibling_var_names,n_children=child_index+1,
                                sibling_data=sibling_data)
                gmm = GMM(n_components=n_components,random_state=setting_values.random_state)
                if regression:
                    fitness_regression=dtfr.format_generated_fitness(fitness,
                                                                     parental_fitness,sibling_fitness,
                                                                     fitness_dim,
                                                                     dtfr.FitnessCombination.product)

                    gmm=_train_regression(gmm,data,fitness_regression,infinite,parental_fitness,sibling_fitness,child_index,fitness_dim)
                else:
                    fitness_single=dtfr.format_generated_fitness(fitness,
                                                                 parental_fitness,sibling_fitness,
                                                                 (dtfr.FitnessInstanceDim.single,
                                                                         dtfr.FitnessFuncDim.single),
                                                                         dtfr.FitnessCombination.product)
                    gmm=_train_weighted_sampling(gmm,data,fitness_single,infinite)
                gmms[child_index]=gmm

            #marginalise gmms, starting from the largest
            for child_index in reversed(range(n_children)):
                if not gmms[child_index]:
                    gmms[child_index]=dtfr.marginalise_gmm(gmms,child_index,parent_vars,sibling_vars)
            gmm_var_name="test"+str(iteration)
            gmm_vars=_construct_gmm_vars(gmms,gmm_var_name,parent_def,parent_node,child_name,
                         parent_var_names,sibling_var_names,
                         sibling_order)

            #the gmms are ordered the same as the children
            #use sibling order sequence to assign gmms[i] to the a child with order i
            #assign variable child i with gmm min(i,sibling_order)
            for k in range(len(child_nodes)):
                child_nodes[k].set_learned_variable(gmm_vars[sibling_order_sequence[k]])

            #evaluate new model
            score=model_evaluation.evaluate(parent_node,parental_fitness,sibling_fitness)
            gmm_vars_retry_eval.append((gmm_vars,score))
            #put original vars back
            for i in range(len(child_nodes)):
                child_nodes[i].delete_learned_variable(gmm_var_name)
        #check which gmm performed best

        max_gmm_vars=None
        for gmm_vars,gmm_score in gmm_vars_retry_eval:
            if gmm_score>iteration_gmm_score:
                max_gmm_vars=gmm_vars
        #if it is better as the previous iteration-
        #inject new variable
        #else print that training didn't help
        if max_gmm_vars:
            for i in range(len(child_nodes)):
                child_nodes[i].set_learned_variable(max_gmm_vars[sibling_order_sequence[i]])
        else:
            print("The model did not improve over consecutive training iteration.")
            break

    _,fitness=dg.training_data_generation(n_data,parent_def,
                                parent_node,parent_var_names,parental_fitness,
                                child_name,sibling_fitness,sibling_var_names,n_children=n_children,
                                sibling_data=sibling_data)
    #show the result of nth iteration
    if verbose:
        print("final results")
        print_result(fitness,parental_fitness,sibling_fitness,iteration)


def print_result(fitness_values,parental_fitness,sibling_fitness,iteration):
    fitness_parent_child=dtfr.format_generated_fitness(fitness_values,parental_fitness,sibling_fitness,
                                                               (dtfr.FitnessInstanceDim.parent_children,
                                                                dtfr.FitnessFuncDim.single),
                                                                dtfr.FitnessCombination.product)
    fitness_parent_child=np.array(fitness_parent_child)
    print("fitness before training iteration ", iteration)
    #print the first fitness func
    print("parent fitness")
    fn.fitness_statistics(fitness_parent_child[:,0],verbose=True)
    if fitness_parent_child.shape[0]>1:
        print("child fitness")
        fn.fitness_statistics(fitness_parent_child[:,1],verbose=True)

def _train_weighted_sampling(gmm,data,fitness,infinite):
    gmm.fit(data,fitness,infinite=infinite,min_covar=0.01)
    return gmm
def _train_regression(gmm,data,fitness,infinite,parental_fitness,sibling_fitness,n_siblings,fitness_dim):
    #check if regression works better this way
    train_data=np.column_stack((data,fitness))
    #for regression calculate full joint
    gmm.fit(train_data,infinite=False,min_covar=0.01)
    indices,targets = dtfr.format_fitness_for_regression_conditioning(parental_fitness,
                                                                      sibling_fitness,n_siblings,
                                                                      len(data[0]),fitness_dim)
    #condition on fitness
    return gmm.condition(indices,targets)


def _construct_gmm_vars(gmms,gmm_name,parent_def,parent_node,child_name,
                 parent_var_names,sibling_var_names,
                 sibling_order):

    #get size of vars
    sibling_vars=[parent_def.children[child_name].variables[name] for name in sibling_var_names]
    parent_vars=[parent_def.variables[name] for name in parent_var_names]
    #check if none of the vars is deterministic



    #edit model with found variables
    from model.search_space import GMMVariable
    child_def=parent_def.children["child"]

    sibling_vars=[child_def.variables[name] for name in sibling_var_names]
    parent_vars=[parent_def.variables[name] for name in parent_var_names]


    #the sibling order of the gmm is maximum the sibling order of the trained gmm
    gmm_vars=[GMMVariable(gmm_name,gmm,parent_vars,sibling_vars,sibling_order=i)
    for gmm,i in zip(gmms,range(len(gmms)))]
    return gmm_vars


