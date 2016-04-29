#Centralise the ordering of variables and children for traing and sampling (marginalising, conditional)
#and the order of the data as well (child and parent variables)
#each variable definition and child definition (not sample) should have an id indicating it's order

#in this class the order can be checked of both variables and samples
import util.utility as ut
import numpy as np
from enum import Enum
import model.fitness as fn

#the dimensionality of the fitness per instance in the data
#single means all fitness value are combined into a single fitness function
#parent children means that there the fitness between parent-child and siblings is combined
#parent siblings means that each sibling has a seperate fitness value, calculated relative to the previously placed siblings

class FitnessInstanceDim(Enum):
    single=1
    seperate=2

#the difference between parent_children and parent_sibling dimensionality can be seen semanticalliy as
#how you look at the child placement concept, if each child is placed after the other the fitness of the next only depends on the previous
#however if you see it as if all children are placed simultanously than their fitness has to be calculated likewise
#->in relation to all children
#the dimensionality of the fitness per fitness function

class FitnessFuncDim(Enum):
    single=1
    seperate=2
class FitnessCombination(Enum):
    product=1
    average=2


#conditioning format for trained GMM: P(S_i|P,S_0,...,S_i-1) given parent and all siblings
def format_data_for_conditional(parent_sample,parent_vars,sibling_samples,sibling_vars,sibling_order):
    #flatten list for calculation of cond distr
    #parent condition variable
    parent_values=list(ut.flatten([parent_sample.values["ind"][var.name] for var in parent_vars]))
    #sibling condition variable
    #only retrieve values of necessary siblings,for sibling order i take last i siblings
    sibling_values=list(ut.flatten([sibling.values["ind"][var.name] for var in sibling_vars
    for sibling in sibling_samples[-sibling_order:]]))
    #the order of values is by convention, also enforced in the sampling and learning procedure
    values=np.concatenate((parent_values,sibling_values))
    #the first X are cond attributes
    indices=np.arange(0,len(values))
    return indices,values

def marginalise_gmm(gmms,child_index,parent_vars,sibling_vars):
    #a model trained on 4 children can also be used for 5 children of the fifth no longer conditions on the first
    #These models can be reused because there's no difference in the markov chain of order n between the n+1 and n+2 state

    #find next full gmm
    full_gmm=next(gmm for gmm in gmms[child_index:] if gmm is not None)
    #calculate indices
    #the order of the data is parent,sibling0,sibling1,..
    indices=np.arange(0,variables_length(parent_vars)+(child_index+1)*variables_length(sibling_vars))
    gmm=full_gmm.marginalise(indices)
    return gmm

def variables_length(variables):
    return np.sum([var.size for var in variables])

def check_normalised(flat_fitness_value_line):
    return all (fn_func_sibl_value<=1 and fn_func_sibl_value>=0 for fn_func_sibl_value in flat_fitness_value_line)

#this needs to be a line because the number of elements in a line can vary and can thus not be cast to a numpy array
def _combine_fitness(flat_fitness_value_line,fitness_comb):
    if fitness_comb is FitnessCombination.product:
        if not check_normalised(flat_fitness_value_line):
            raise ValueError("only normalised fitness values can be combined as a product. Unnormalised line: ",flat_fitness_value_line)
        #this only works if the fitness is normalised
        return np.prod(flat_fitness_value_line,None)
    else:
        return np.average(flat_fitness_value_line,None)

def reduce_fitness_dimension(fitness_values,
                             fitness_dim,fitness_comb):

    return [_reduce_fitness_dimension(fn_value_line, fitness_dim,fitness_comb) for fn_value_line in fitness_values]
def _reduce_fitness_dimension(fitness_value_line,
                             fitness_dim,fitness_comb):

    if fitness_dim[0] is FitnessInstanceDim.seperate and  fitness_dim[1] is FitnessFuncDim.seperate:
        return fitness_value_line
    if fitness_dim[0] is FitnessInstanceDim.single and  fitness_dim[1] is FitnessFuncDim.seperate:
        return [_combine_fitness(fn_func_value,fitness_comb) for fn_func_value in fitness_value_line]
    if fitness_dim[0] is FitnessInstanceDim.single and  fitness_dim[1] is FitnessFuncDim.single:
        #turn back in array because the fitness value array should be 2D
        return [_combine_fitness(list(ut.flatten(fitness_value_line)),fitness_comb)]
    if fitness_dim[0] is FitnessInstanceDim.seperate and  fitness_dim[1] is FitnessFuncDim.single:
        #group the fitness values per siblings is not possible because some fitness functions can be filtered for certain siblings
        raise ValueError("Seperate siblings and single fitness function values is not supported.")

def normalise_fitness(fitness_values):
    fn_bounds=np.array(fitness_value_bounds(fitness_values))
    normalised_fitness_values=[]
    for fn_value_lines in fitness_values:
        normalised_fitness_values.append([(np.array(fn_values)-fn_func_bounds[0])/(fn_func_bounds[1]-fn_func_bounds[0]) for fn_values,fn_func_bounds in zip(fn_value_lines,fn_bounds)])

    return np.array(normalised_fitness_values)

#works only on 2D fitness value arrays
def fitness_value_bounds(fitness_values):

    return [(np.min(list(ut.flatten(np.array(fitness_values)[:,i]))),np.max(list(ut.flatten(np.array(fitness_values)[:,i])))) for i in range(len(fitness_values[0]))]

def format_fitness_and_data_training(data,fitness_values,fitness_funcs):
    #normalise fitness
    fitness_values=normalise_fitness(fitness_values)

    #TODO test order
    #apply order
    fitness_values=[[np.array(fn_func_values)**fn_func.order for fn_func_values,fn_func in zip(fn_value_line,fitness_funcs)] for fn_value_line in fitness_values ]

    filtered_data=[]
    filtered_fitness_values=[]


    #filter fitness_values and data based on fitness_funcs cap
    for l,fn_value_line in enumerate(fitness_values):
        if all(fitness_funcs[i].threshhold<=fn_func_sibling_value for i,fn_func_values in enumerate(fn_value_line) for fn_func_sibling_value in fn_func_values):
            filtered_data.append(data[l])
            filtered_fitness_values.append(fitness_values[l])

    #filter fitness which is always 1 or close too it (no training necessary and numerically unstable)

    non_max_fn_indices=[[] for i in range(len(fitness_funcs))]

    #for each fitness func
    for i in range(len(fitness_funcs)):
        #for each fitness func value per sibling
        for j in range(len(filtered_fitness_values[0][i])):
            if not all(fn_value_line[i][j]>0.99 for fn_value_line in filtered_fitness_values ):
                #save index of column
                non_max_fn_indices[i].append(j)
    #TODO

    #if the fitness is always 1 for each sibling it should be warned to the user to remove it from the learning
    #filter rows that contain outliers, because there is not much information in outliers, or is their?

    #use indices to build new array
    if any(len(func_ind)>0 for func_ind in non_max_fn_indices):
        non_max_fn_values=[]
        for l,fn_value_line in enumerate(filtered_fitness_values):
            non_max_fn_value_line=[]
            for i in range(len(non_max_fn_indices)):
                non_max_fn_value_line.append([filtered_fitness_values[l][i][j] for j in non_max_fn_indices[i]])
            non_max_fn_values.append(non_max_fn_value_line)
    else:
        non_max_fn_values=filtered_fitness_values

    return filtered_data,non_max_fn_values


def _format_fitness_targets_regression(fitness_funcs,unreduced_fitness_values,fitness_dim):
    #first create one line of fitness targets, with the same structure as unreduced fitness
    fitness_targets=[]
    for i in range(len(unreduced_fitness_values[0])):
        #the i th block in the fitness value line corresponds with i th fitness function
        fitness_targets.append([fitness_funcs[i].regression_target]*len(unreduced_fitness_values[0][i]))

    #reduce the fitness values in the same way as the original fitness, targets are already normalised
    return _reduce_fitness_dimension(fitness_targets,fitness_dim,FitnessCombination.average)

def format_fitness_values_columnwise(fitness_values):
    return [np.array(column) for column in map(list,zip(*fitness_values))]

def format_target_indices_for_regression_conditioning(data,unreduced_fitness_values,fitness_funcs,
                                               fitness_dim):
    #create fitness targets
    fitness_targets= _format_fitness_targets_regression(fitness_funcs,unreduced_fitness_values,
                                                        fitness_dim)

    #create fitness indices
    fitness_size= len(list(ut.flatten(fitness_targets)))
    data_size=len(list(ut.flatten(data[0])))
    indices=np.arange(data_size,data_size+fitness_size)
    return indices,fitness_targets

#the order of the data is parent,sibling0,sibling1,..
#the order of the variable of each instance in the data is defined by the variable lists
def format_data_for_training(parent,parent_var_names,siblings,sibling_var_names):
    data=list(ut.flatten([parent.values["ind"][name] for name in parent_var_names]+[[child.values["ind"][name] for name in sibling_var_names] for child in siblings]))
    return data


#all variables need to be of type numpy array
def split_variables(variables,joint_data):
    sizes=[v.size for v in variables]
    sizes.insert(0,0)
    lengths=np.cumsum(sizes)
    #this is for calculating the edges of the vector to be return in relative value
    return [np.array(joint_data[l1:l2]) if not var.frozen() else var.freeze_value
    for (l1,l2),var in zip(ut.pairwise(lengths),variables)]

