
import numpy as np

from util import setting_values
import model.test_models as tm

from learning.gmm import GMM

import util.utility as ut

import learning.data_generation as dg
import model.evaluation as mev

from model import fitness as fn
import util.data_format as dtfr

from util import visualisation as vis
class print_params:
    verbose_trial=False
    verbose_iter=True
    verbose_final_extra=False
    visual=True
    print_parameters_set=True
    print_fitness_bins=False
class training_params:
    def reset():
        training_params.sibling_order_sequence=[0,1,2,3,4,4,4]
        training_params.gmm_full=[False,False,True,False,True]

        training_params.sibling_data=dg.SiblingData.first
        training_params.fitness_dim=(dtfr.FitnessInstanceDim.seperate,dtfr.FitnessFuncDim.seperate)
        training_params.n_data=500
        training_params.poisson=False

        training_params.n_iter=1
        training_params.n_trial=5
        training_params.n_model_eval_data=500

        training_params.n_components=30
        training_params.min_covar=0.02

        training_params.regression=False
        training_params.extra_info=""
        training_params.fitness_regr_cond=1
        training_params.fitness_cap=0
        training_params.model_4_target=0.6


    sibling_order_sequence=[0,1,2,3,4,4,4]
    gmm_full=[False,False,True,False,True]
    fitness_regr_cond=1
    fitness_cap=0

    sibling_data=dg.SiblingData.first
    fitness_dim=(dtfr.FitnessInstanceDim.seperate,dtfr.FitnessFuncDim.seperate)
    n_data=500
    poisson=False

    n_iter=1
    n_trial=3
    n_model_eval_data=200

    n_components=30
    min_covar=0.02
    model_4_target=0.6

    regression=False

    title="test"
    extra_info=""

import timeit


def time_score_all_models():
    scores=[]
    times=[]
    time_score_training(training_model_0,times,scores)
    time_score_training(training_model_1,times,scores)
    time_score_training(training_model_2,times,scores)
    time_score_training(training_model_3,times,scores)
    time_score_training(training_model_4,times,scores)
    print("model average computation time", np.average(times))
    print("model average score gain", np.average(scores))
    print()

def time_score_training(training_model_function,times,scores):
    try:
        extra_info = training_params.extra_info
        training_params.extra_info= extra_info +  " ," + training_model_function.__name__
        start_time = timeit.default_timer()
        score = training_model_function()
        elapsed = timeit.default_timer() - start_time
        scores.append(score)
        times.append(elapsed)
        #the title is different for each model, revert to old string
        training_params.extra_info=extra_info
    except KeyboardInterrupt:
        raise
    except Exception as e:
        training_params.reset()
        print(e)
def test_n_model_eval():
    training_params.title="test for number of number of model evaluations between trials "
    print(training_params.title)
    print()
    for n_model_eval_data in np.arange(200,1000,200):
        training_params.n_model_eval_data=n_model_eval_data
        training_params.extra_info="number of model evaluation: " + str(n_model_eval_data)
        print("number of model evaluation: ",n_model_eval_data)
        time_score_all_models()

    print_train_parameters()
    training_params.reset()
def test_model_var():
    training_params.title="test for model with additional variable "
    print(training_params.title)
    print("single stochastic variable")
    methods=[training_model_0,training_model_1,training_model_2,training_model_3,training_model_4]
    for i,method in enumerate(methods):
        training_params.extra_info="single stochastic variable, model " + str(i)
        scores=[]
        times=[]
        try:
            start_time = timeit.default_timer()
            score = method(False)
            elapsed = timeit.default_timer() - start_time
            scores.append(score)
            times.append(elapsed)
        except KeyboardInterrupt:
            raise
        except:
            print("error")
    print("model average computation time", np.average(times))
    print("model average score gain", np.average(scores))
    print()
    print("double stochastic variable")
    for i,method in enumerate(methods):
        training_params.extra_info="double stochastic variable, model " + str(i)
        scores=[]
        times=[]
        try:
            start_time = timeit.default_timer()
            score = method(True)
            elapsed = timeit.default_timer() - start_time
            scores.append(score)
            times.append(elapsed)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(e)
    print("model average computation time", np.average(times))
    print("model average score gain", np.average(scores))
    print()
    print_train_parameters()
    training_params.reset()

def test_fitness_dim_regression():
    #regression true and different dims
    training_params.title="test for fitness dimensionality regression sampling "
    print(training_params.title)
    print()
    print("with regression")
    training_params.regression=True
    fitness_dims=[]
    fitness_dims.append((dtfr.FitnessInstanceDim.seperate,dtfr.FitnessFuncDim.seperate))
    fitness_dims.append((dtfr.FitnessInstanceDim.single,dtfr.FitnessFuncDim.seperate))
    fitness_dims.append((dtfr.FitnessInstanceDim.single,dtfr.FitnessFuncDim.single))
    for fitness_dim in fitness_dims:
        training_params.extra_info="fitness dimensionality, " + str(fitness_dim)

        print("fitness dimensionality, ",fitness_dim)
        training_params.fitness_dim=fitness_dim

        time_score_all_models()



    #regresion false
    print("without regression")
    training_params.extra_info="without regression"
    training_params.regression=False

    time_score_all_models()


    print_train_parameters()
    training_params.reset()
def test_poisson():
    training_params.title="test for poisson disk sampling "
    print(training_params.title)
    print()
    print("with poison")
    training_params.extra_info="with poison"
    training_params.poisson=True
    time_score_all_models()


    print("without poisson")
    training_params.extra_info="without poison"

    training_params.poisson=False

    time_score_all_models()


    print_train_parameters()
    training_params.reset()
def test_n_iter_n_trial():
    print_params.verbose_trial=True
    training_params.title="test for retraining number of iterations and number of trials "
    print(training_params.title)
    print()
    for n_iter in range(1,4):
        for n_trial in range(1,6):
            training_params.n_iter=n_iter
            training_params.n_trial=n_trial
            print("number of iterations, trials: ",n_iter,n_trial)
            training_params.extra_info="number of iterations, trials: " + str(n_iter) + str(n_trial)

            time_score_all_models()

    print_train_parameters()
    training_params.reset()
    print_params.verbose_trial=False
def test_n_component():
    training_params.title="test for number of components in gmm "
    print(training_params.title)
    print()
    for n_components in np.arange(10,50,10):
        training_params.n_components=n_components
        training_params.extra_info="GMM number of components: " + str(n_components)
        print("GMM number of components: ",n_components)
        time_score_all_models()

    print_train_parameters()
    training_params.reset()
def test_min_covar():
    training_params.title="test for min covar of gmm "
    print(training_params.title)
    print()
    for covar_fact in np.arange(1,7,1):
        training_params.min_covar=0.1**covar_fact
        training_params.extra_info="GMM min covar: "+ str(training_params.min_covar)
        print("GMM min covar: ",training_params.min_covar)
        time_score_all_models()

    print_train_parameters()
    training_params.reset()
def test_n_data():
    training_params.title="test for number of training samples "
    print(training_params.title)
    for n_data in range(200,1000,100):
        training_params.n_data=n_data
        training_params.extra_info="number of training samples: "+ str(n_data)
        print("number of training sample: ",n_data)
        time_score_all_models()

    print_train_parameters()
    training_params.reset()

def test_marginal_gmm():
    import itertools
    training_params.title="test for gmm full training, marginalisation, "
    print(training_params.title)


    sibling_order_sequence=[0,1,2,3,4,4,4]



    for gmm_full_part in itertools.product([True,False], repeat=3):

        gmm_full=[False]+list(gmm_full_part)+[True]
        training_params.extra_info="GMM full training: "+ str(gmm_full)
        training_params.gmm_full=gmm_full
        training_params.sibling_order_sequence=sibling_order_sequence
        print("gmm full: ",gmm_full)
        time_score_all_models()

    print_train_parameters()
    training_params.reset()


def test_sibling_order_seq():
    training_params.title="test for sibling order seq"
    print(training_params.title)

    sibling_order_sequences=[]
    gmm_fulls=[]

    sibling_order_sequences.append([0,1,1,1,1,1,1])
    gmm_fulls.append([False,True])
    sibling_order_sequences.append([0,1,2,2,2,2,2])
    gmm_fulls.append([False,True,True])
    sibling_order_sequences.append([0,1,2,3,3,3,3])
    gmm_fulls.append([False,True,True,True])
    sibling_order_sequences.append([0,1,2,3,4,4,4])
    gmm_fulls.append([False,True,True,True,True])

    for sibling_order_sequence,gmm_full in zip(sibling_order_sequences,gmm_fulls):
        training_params.extra_info="sibling order sequence: "+ str(sibling_order_sequence)
        training_params.gmm_full=gmm_full
        training_params.sibling_order_sequence=sibling_order_sequence
        print("gmm full: ",gmm_full)
        print("sibling_order_sequence:",sibling_order_sequence)
        time_score_all_models()

    print_train_parameters()
    training_params.reset()

def fitness_regression_condition_test():
    training_params.title="test for regression condition, model fitness target distance"
    print(training_params.title)

    training_params.regression=True
    for cond in np.arange(0.6,1.3,0.1):
        training_params.extra_info="regression with condition: "+ str(cond)
        print("condition: "+ str(cond))
        time_score_all_models()

    print_train_parameters()
    training_params.reset()

def fitness_cap_test():
    training_params.title="test for fitness caps, model fitness target distance"
    print(training_params.title)
    for cap in np.arange(0,0.4,0.05):
        print("cap: ",cap)
        training_params.extra_info="cap: "+ str(cap)

        time_score_all_models()

    print_train_parameters()
    training_params.reset()

def fitness_order_MO_test():
    training_params.title="test for fitness order, fitness target distance, polgygon overlay"
    print_params.verbose_iter=True
    print_params.verbose_final_extra=True

    for fo0 in range(4,32,4):
        for fo1 in range(4,32,4):
            print("fitness orders TD PO:",fo0,fo1)
            training_params.extra_info="TD order: " + str(fo0) + "  PO order: " + str(fo1)
            fitness_funcs=[fn.Targetting_Fitness("Target distances",fn.min_dist_sb,fn.Fitness_Relation.pairwise_siblings,fo0,0,1,target=2),fn.Fitness("polygon overlay",fn.norm_overlap_pc,fn.Fitness_Relation.pairwise_parent_child,fo1,0,1)]
            #hierarchy,var_rot,var_nr_children)
            model=tm.model_var_pos(True,False,True)
            sibling_var_names=["position"]

            parent_var_names=["shape0"]


            try:
                training(model=model,fitness_funcs=fitness_funcs,
                 sibling_var_names=sibling_var_names,parent_var_names=parent_var_names)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(e)
    print_train_parameters()
    training_params.reset()

def test_model_4_target():
    training_params.title="test for model 4 optimal fitness target"
    print(training_params.title)
    for target in np.arange(0.05,1,0.1):
        print("target: ",target)
        training_params.model_4_target=target
        training_params.extra_info="target: "+ str(target)
        print("score gain: ",training_model_4())

    print_train_parameters()
    training_params.reset()
def training_model_0(var_rot=False):
    print("model 0")
    fitness_funcs=[fn.Targetting_Fitness("Target distances",fn.min_dist_sb,fn.Fitness_Relation.pairwise_siblings,4,training_params.fitness_cap,training_params.fitness_regr_cond,target=2)]
    #hierarchy,var_rot,var_nr_children)
    model=tm.model_var_pos(True,var_rot,True)
    if var_rot:
        sibling_var_names=["position","rotation"]
    else:
        sibling_var_names=["position"]

    parent_var_names=["shape0"]


    return training(model=model,fitness_funcs=fitness_funcs,
             sibling_var_names=sibling_var_names,parent_var_names=parent_var_names)

def training_model_1(var_rot=False):
    print("model 1")
    fitness_funcs=[fn.Fitness("polygon overlay",fn.norm_overlap_pc,fn.Fitness_Relation.pairwise_parent_child,32,training_params.fitness_cap,training_params.fitness_regr_cond)]
    model=tm.model_var_pos(True,var_rot,True)
    if var_rot:
        sibling_var_names=["position","rotation"]
    else:
        sibling_var_names=["position"]
    parent_var_names=["shape0"]


    return training(model=model,fitness_funcs=fitness_funcs,
             sibling_var_names=sibling_var_names,parent_var_names=parent_var_names)

def training_model_2(var_pos=True):
    print("model 2")
    fitness_funcs=[fn.Fitness("side alignment",fn.closest_side_alignment_pc,fn.Fitness_Relation.pairwise_parent_child,1,training_params.fitness_cap,training_params.fitness_regr_cond)]
    #hierarchy,var_rot,var_nr_children)
    model=tm.model_var_rot(True,var_pos,True)
    if var_pos:
        sibling_var_names=["rotation","position"]
    else:
        sibling_var_names=["rotation"]
    parent_var_names=["shape0","shape3","shape6","shape9"]


    return training(model=model,fitness_funcs=fitness_funcs,
             sibling_var_names=sibling_var_names,parent_var_names=parent_var_names)

def training_model_3(var_pos=True):
    print("model 3")

    fitness_funcs=[
                   fn.Fitness("centroid difference",fn.centroid_dist_absolute,fn.Fitness_Relation.absolute,1,training_params.fitness_cap,training_params.fitness_regr_cond)]
    #hierarchy,var_rot,var_nr_children)
    model=tm.model_var_size(True,var_pos,True)
    if var_pos:
        sibling_var_names=["size","position"]
    else:
        sibling_var_names=["size"]
    parent_var_names=["shape2","shape4"]



    return training(model=model,fitness_funcs=fitness_funcs,
             sibling_var_names=sibling_var_names,parent_var_names=parent_var_names)
def training_model_4(var_pos=True):
    print("model 4")
    fitness_funcs=[
                   fn.Targetting_Fitness("surface ration",fn.combinatory_surface_ratio_absolute,fn.Fitness_Relation.absolute,8,training_params.fitness_cap,training_params.fitness_regr_cond,target=training_params.model_4_target)]
    #hierarchy,var_rot,var_nr_children)
    model=tm.model_var_size(True,var_pos,True)
    if var_pos:
        sibling_var_names=["size","position"]
    else:
        sibling_var_names=["size"]
    parent_var_names=["shape2","shape4"]

    return training(model=model,fitness_funcs=fitness_funcs,
             sibling_var_names=sibling_var_names,parent_var_names=parent_var_names)


def training(model,fitness_funcs,sibling_var_names,parent_var_names):

    sibling_order_sequence=training_params.sibling_order_sequence
    gmm_full=training_params.gmm_full

    sibling_data=training_params.sibling_data
    fitness_dim=training_params.sibling_data
    n_data=training_params.n_data
    poisson=training_params.poisson

    n_iter=training_params.n_iter
    n_trial=training_params.n_trial
    n_model_eval_data=training_params.n_model_eval_data

    n_components=training_params.n_components
    min_covar=training_params.min_covar

    regression=training_params.regression

    #experiment hyperparameters:

    fitness_average_threshhold=0.95
    fitness_func_threshhold=0.98

    #this sequence indicates the order of the markov chain between siblings [1,2]-> second child depends on the first
    #the third on the first and second
    #the first child is always independent

    sibling_data=dg.SiblingData.first
    fitness_dim=(dtfr.FitnessInstanceDim.seperate,dtfr.FitnessFuncDim.seperate)

    #the sibling order defines the size of the joint distribution that will be trained
    sibling_order=np.max(sibling_order_sequence)
    n_children=sibling_order+1

    #gmm marginalisation [order_1,order_2,..,order_sibling_order]

    #True->train full joint
    #False->derive (marginalise) from closest higher order



    child_name="child"
    #model to train on
    parent_node,parent_def=model
    child_nodes=parent_node.children[child_name]

    #training variables and fitness functions
    #this expliicitly also defines the format of the data

    n_model_eval_data=400

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
    #only the func order and cap is used for training

    model_evaluation = mev.ModelEvaluation(n_model_eval_data,parent_def,parent_node,parent_var_names,
                                               child_name,sibling_var_names,
                                               fitness_funcs,
                                               fitness_average_threshhold,fitness_func_threshhold)

    score=model_evaluation.evaluate()
    startscore=score
    delta_score=0
    print("score before training: ", score)

    #check sibling sequence
    wrong_sequence = any(sibling_order_sequence[i]>i for i in range(len(sibling_order_sequence)))
    if wrong_sequence:
        print(sibling_order_sequence)
        raise ValueError("Some orders of the sibling order sequence exceed the number of previous siblings.")
    max_children=parent_def.variable_range(child_name)[1]
    if len(sibling_order_sequence) != max_children:
        raise ValueError("The number of siblings implied by the sibling order sequence can not be different than the maximum number of children in the model.")
    #check marginalisation
    if len(gmm_full) != n_children:
        raise ValueError("the array defining which sibling order to train seperately should have the same length as the maximum amount of children for a given sibling order. \n length array: ",len(gmm_full),", expected: ",n_children)
    #do n_iter number of retrainings using previously best model
    #before iterating set the variable that will control whether a new model is an improvement
    iteration_gmm_score=score
    for iteration in range(n_iter):

        #find out the performance of the current model

        data,fitness_values=dg.training_data_generation(n_data,parent_def,
                                parent_node,parent_var_names,
                                child_name,sibling_var_names,n_children,
                                fitness_funcs,
                                sibling_data=sibling_data,poisson=poisson)

        if print_params.verbose_iter:
            model_evaluation.print_evaluation(fitness_values,iteration,summary=not print_params.print_fitness_bins)

        if model_evaluation.converged(fitness_values):
            return

        #combine fitness per func
        #evaluate model at the start of every iteration

        gmm_vars_retry_eval=[]
        #do n trials to find a better gmm for the model
        for trial in range(n_trial):
            #calculate all full joints
            gmms=[None]*n_children
            for child_index in np.where(gmm_full)[0]:
                #generate data for each number of children
                data,fitness_values=dg.training_data_generation(n_data,parent_def,
                                parent_node,parent_var_names,
                                child_name,sibling_var_names,child_index+1,
                                fitness_funcs,
                               sibling_data,poisson)
                gmm = GMM(n_components=n_components,random_state=setting_values.random_state)
                data,fitness_values=dtfr.filter_fitness_and_data_training(data,fitness_values,
                                                                                  fitness_funcs)
                if regression:

                    fitness_values=dtfr.apply_fitness_order(fitness_values,fitness_funcs)

                    fitness_regression=dtfr.reduce_fitness_dimension(fitness_values,fitness_dim,
                                                                        dtfr.FitnessCombination.product)
                    #renormalise
                    fitness_regression=dtfr.normalise_fitness(fitness_regression)
                    fitness_regression=[ list(ut.flatten(fn_value_line)) for fn_value_line in fitness_regression]

                    #add fitness data
                    train_data=np.column_stack((data,fitness_regression))


                    #for regression calculate full joint
                    gmm.fit(train_data,infinite=False,min_covar=min_covar)

                    indices,targets = dtfr.format_target_indices_for_regression_conditioning(data,fitness_values,
                                                                                             fitness_funcs,
                                                                                             fitness_dim)

                    #condition on fitness
                    gmm= gmm.condition(indices,targets)

                else:

                    fitness_values=dtfr.apply_fitness_order(fitness_values,fitness_funcs)
                    #reduce fitness to a single dimension
                    fitness_single=dtfr.reduce_fitness_dimension(fitness_values,(dtfr.FitnessInstanceDim.single,
                                                                         dtfr.FitnessFuncDim.single),
                                                                        dtfr.FitnessCombination.product)
                    #renormalise
                    fitness_single=dtfr.normalise_fitness(fitness_single)

                    gmm.fit(data,np.array(fitness_single)[:,0],infinite=False,min_covar=min_covar)
                gmms[child_index]=gmm

            #marginalise gmms, starting from the largest
            for child_index in reversed(range(n_children)):
                if not gmms[child_index]:
                    gmms[child_index]=dtfr.marginalise_gmm(gmms,child_index,parent_vars,sibling_vars)
            gmm_var_name="test"+str(iteration)
            gmm_vars=_construct_gmm_vars(gmms,gmm_var_name,parent_def,parent_node,child_name,
                         parent_var_names,sibling_var_names)

            #the gmms are ordered the same as the children
            #use sibling order sequence to assign gmms[i] to the a child with order i
            #assign variable child i with gmm min(i,sibling_order)
            for k in range(len(child_nodes)):
                child_nodes[k].set_learned_variable(gmm_vars[sibling_order_sequence[k]])

            #evaluate new model

            score=model_evaluation.evaluate()

            gmm_vars_retry_eval.append((gmm_vars,score))
            if print_params.verbose_trial:
                print()
                print("trial: ", trial," score: ",score)

            #put original vars back
            for i in range(len(child_nodes)):
                child_nodes[i].delete_learned_variable(gmm_var_name)
        #check which gmm performed best
        max_gmm_vars=None
        for gmm_vars,gmm_score in gmm_vars_retry_eval:
            if gmm_score>iteration_gmm_score:
                max_gmm_vars=gmm_vars
                iteration_gmm_score=gmm_score
        #if it is better as the previous iteration-
        #inject new variable
        #else print that training didn't help
        gmm_scores=[gmm_score for gmm_vars,gmm_score in gmm_vars_retry_eval ]
        print("iteration ",iteration, " trial score mean: ", np.mean(gmm_scores)," variance: ",np.var(gmm_scores))
        if max_gmm_vars:
            print("improved selected with score: ",iteration_gmm_score )
            delta_score=iteration_gmm_score-startscore
            for i in range(len(child_nodes)):
                child_nodes[i].set_learned_variable(max_gmm_vars[sibling_order_sequence[i]])
        else:
            print("The did not improve over consecutive training iteration.")
            break
    if print_params.verbose_final_extra:
        print()
        print("final evaluation of fitness" )
        data,fitness_values=dg.training_data_generation(n_model_eval_data,parent_def,
                                parent_node,parent_var_names,
                                child_name,sibling_var_names,n_children,
                                fitness_funcs,
                                sibling_data=sibling_data,poisson=False)
        model_evaluation.print_evaluation(fitness_values,-1,summary=not print_params.print_fitness_bins)
        print("score gain: ", str(delta_score))
    if print_params.visual and max_gmm_vars:
        for gmm_var in max_gmm_vars:
           vis.draw_1D_2D_GMM_variable_sampling(gmm_var,training_params.title,training_params.extra_info)
    if print_params.print_parameters_set:

        print("fitness parameters,")
        for fitn in fitness_funcs:
            print(str(fitn))
            print(",")
        print()

        print("model parameters")
        print("parent variables,",str(parent_var_names))

        print("sibling variables,", str(sibling_var_names))

        print()
    return delta_score




def print_train_parameters():
    print("parameter configuration")
    print(training_params.title)
    print("model hierarchy parameters")
    print("sibling order list,", training_params.sibling_order_sequence)
    print("non marginalisation list,",training_params.gmm_full)

    print("data generation parameters")
    print("number of data samples, ", training_params.n_data)
    print("sibling data,",training_params.sibling_data)
    print("fitness dimension,",training_params.fitness_dim)
    print("poisson sampling,", training_params.poisson)
    print()

    print("GMM parameters")
    print("minimum covariance, ",training_params.min_covar)
    print("number of components, ", training_params.n_components)

    print("retraining")
    print("number of trials,", training_params.n_trial)
    print("number of iterations,", training_params.n_iter)
    print("number of samples for evaluations,", training_params.n_model_eval_data)
    print()




def _construct_gmm_vars(gmms,gmm_name,parent_def,parent_node,child_name,
                 parent_var_names,sibling_var_names):

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


