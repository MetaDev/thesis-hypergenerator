
import numpy as np

import learning.evaluation as ev
from util import setting_values
import model.test_models as tm

from learning.gmm import GMM

from operator import itemgetter

import learning.data_generation_MO as dg

def training(n_data=100,n_iter=1,n_trial=1,n_components=15,infinite=False,regression=False,verbose=True):
    #experiment hyperparameters:
    #this sequence indicates the order of the markov chain between siblings [1,2]-> second child depends on the first
    #the third on the first and second
    #the first child is always independent
    sibling_order_sequence=[0,1,2,3,4,4,4,4]
    #sibling_order_sequence=[0,1,2,3,4]
    #the sibling order defines the size of the joint distribution that will be trained
    sibling_order=np.max(sibling_order_sequence)
    child_name="child"
    sibling_train_order=dg.SiblingTrainReOrder.no_reorder
    order_variable="position"
    respect_sibling_order=False

    #training variables and fitness functions
    #this expliicitly also defines the format of the data
    sibling_var_names=["position"]
    parent_var_names=["shape3"]

    #this expliicitly also defines the format of the data
    #fitness func, order cap and regression target
    sibling_fitness_funcs=[(dg.fitness_min_dist,4,0,1)]
    #only the func order and cap is used for training
    sibling_fitness_funcs_sampling=[f[0:3] for f in sibling_fitness_funcs]

    parent_child_fitness_funcs=[(dg.fitness_polygon_overl,32,0,1)]
    parent_child_fitness_funcs_sampling=[f[0:3] for f in parent_child_fitness_funcs]
    #model to train on
    parent_node,parent_def=tm.test_model_var_child_position_parent_shape()
    child_nodes=parent_node.children[child_name]


    #check sibling sequence
    wrong_sequence = any(sibling_order_sequence[i]>i for i in range(len(sibling_order_sequence)))
    if wrong_sequence:
        raise ValueError("Some orders of the sibling order sequence exceed the number of previous siblings.")
    max_children=parent_def.max_children(child_name)
    if len(sibling_order_sequence) != max_children:
        raise ValueError("The number of siblings implied by the sibling order sequence can not be different than the maximum number of children in the model.")
    #do n_iter number of retrainings using previously best model
    for iteration in range(n_iter):
        #find out the performance of the current model
        data,fitness=dg.data_generation(n_data,parent_node,parent_child_fitness_funcs_sampling,
                                               sibling_fitness_funcs_sampling,
                                               parent_var_names,
                                               sibling_var_names,
                                               child_name=child_name,
                                               sibling_order=sibling_order,
                                               sibling_train_order=sibling_train_order,
                                               order_variable=order_variable,
                                               respect_sibling_order=respect_sibling_order)

        if verbose:
            print("fitness before training iteration ", iteration)
            #print the first fitness func
            print("parent fitness")
            ev.fitness_statistics(fitness[:,0],verbose=verbose)
            print("child fitness")
            ev.fitness_statistics(fitness[:,1],verbose=verbose)

        mean_fitness=np.mean(fitness,axis=0)


        gmm_vars_retry_eval=[]
        #do n trials to find a better gmm for the model
        for trial in range(n_trial):

            gmm = GMM(n_components=n_components,random_state=setting_values.random_state)
            all_fitness=parent_child_fitness_funcs+sibling_fitness_funcs
            fitness_targets=[f[3] for f in all_fitness]
            gmm=_train_regression(gmm,data,fitness,infinite,fitness_targets)


            gmm_vars=_construct_gmm_vars(gmm,"test",parent_def,parent_node,child_name,
                         parent_var_names,sibling_var_names,
                         sibling_order)

            #the gmms are ordered the same as the children
            #use sibling order sequence to assign gmms[i] to the a child with order i
            #assign variable child i with gmm min(i,sibling_order)
            for k in range(len(child_nodes)):
                child_nodes[k].set_learned_variable(gmm_vars[sibling_order_sequence[k]])

            #evaluate new model
            _,eval_fitness=dg.data_generation(n_data,parent_node,parent_child_fitness_funcs_sampling,
                                                   sibling_fitness_funcs_sampling,
                                                   parent_var_names,
                                                   sibling_var_names,
                                                   child_name=child_name,
                                                   sibling_order=sibling_order,
                                                   sibling_train_order=sibling_train_order,
                                                   order_variable=order_variable,
                                                   respect_sibling_order=respect_sibling_order)
            temp_mean_fitness=np.mean(eval_fitness,axis=0)

            gmm_vars_retry_eval.append((gmm_vars,temp_mean_fitness))
            #put original vars back
            for i in range(len(child_nodes)):
                child_nodes[i].delete_learned_variable("test")
        #check which gmm performed best

        #first check which one is larger than previous
        #than choose the "best", we will use the product of all fitness as final value
        gmm_f_prod=0
        max_gmm_vars=None
        for gmm_vars,gmm_mean_fitness in gmm_vars_retry_eval:
            if all((gmm_f > f for f,gmm_f in zip(mean_fitness,gmm_mean_fitness))):
                gmm_f_prod_temp=np.prod(gmm_mean_fitness)
                if gmm_f_prod_temp>gmm_f_prod:
                    max_gmm_vars=gmm_vars
                    gmm_f_prod=gmm_f_prod_temp
        #inject new variable
        if max_gmm_vars:
            for i in range(len(child_nodes)):
                child_nodes[i].set_learned_variable(max_gmm_vars[sibling_order_sequence[i]])
        _,fitness=dg.data_generation(n_data,parent_node,parent_child_fitness_funcs_sampling,
                                               sibling_fitness_funcs_sampling,
                                               parent_var_names,
                                               sibling_var_names,
                                               child_name=child_name,
                                               sibling_order=sibling_order,
                                               sibling_train_order=sibling_train_order,
                                               order_variable=order_variable,
                                               respect_sibling_order=respect_sibling_order)
    #show the result of nth iteration
    if verbose:
        print("final")
        #print the first fitness func
        print("parent fitness")
        ev.fitness_statistics(fitness[:,0],verbose=verbose)
        print("child fitness")
        ev.fitness_statistics(fitness[:,1],verbose=verbose)





def _train_regression(gmm,data,fitness,infinite,fitness_targets):
    #check if regression works better this way
    train_data=np.column_stack((data,fitness))
    #for regression calculate full joint
    gmm.fit(train_data,infinite=infinite,min_covar=0.01)
    #fitness indices and values according to convention
    fitness_indices=np.arange(len(data[0]),len(data[0])+len(fitness[0]))
    #condition on fitness
    return gmm.condition(fitness_indices,fitness_targets)


def _construct_gmm_vars(gmm,gmm_name,parent_def,parent_node,child_name,
                 parent_var_names,sibling_var_names,
                 sibling_order):

    #get size of vars
    sibling_vars=[parent_def.children[child_name].variables[name] for name in sibling_var_names]
    parent_vars=[parent_def.variables[name] for name in parent_var_names]

    import util.data_format as dtfr
    gmms=dtfr.marginalise_gmm(gmm,parent_vars,sibling_vars,sibling_order)

    #edit model with found variables
    from model.search_space import GMMVariable
    child_def=parent_def.children["child"]

    child_vars=[child_def.variables["position"]]
    sibling_vars=[child_def.variables[name] for name in sibling_var_names]
    parent_vars=[parent_def.variables[name] for name in parent_var_names]


    #the sibling order of the gmm is maximum the sibling order of the trained gmm
    gmm_vars=[GMMVariable(gmm_name,gmm,child_vars,parent_vars,sibling_vars,sibling_order=i)
    for gmm,i in zip(gmms,range(len(gmms)))]
    return gmm_vars
