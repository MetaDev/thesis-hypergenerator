#Centralise the ordering of variables and children for traing and sampling (marginalising, conditional)
#and the order of the data as well (child and parent variables)
#each variable definition and child definition (not sample) should have an id indicating it's order

#in this class the order can be checked of both variables and samples
import util.utility as ut
import numpy as np
from enum import Enum

#the dimensionality of the fitness per instance in the data
#single means all fitness value are combined into a single fitness function
#parent children means that there the fitness between parent-child and siblings is combined
#parent siblings means that each sibling has a seperate fitness value, calculated relative to the previously placed siblings

class FitnessInstanceDim(Enum):
    single=1
    parent_children=2
    parent_sibling=3

#the difference between parent_children and parent_sibling dimensionality can be seen semanticalliy as
#how you look at the child placement concept, if each child is placed after the other the fitness of the next only depends on the previous
#however if you see it as if all children are placed simultanously than their fitness has to be calculated likewise
#->in relation to all children
#the dimensionality of the fitness per fitness function

class FitnessFuncDim(Enum):
    single=1
    seperate=2
#TODO add weighted average
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



def combine_fitness(fitness_values,fitness_axis,fitness_comb):
    if fitness_comb is FitnessCombination.product:
        return np.prod(fitness_values,fitness_axis)
    else:
        return np.average(fitness_values,fitness_axis)

def format_fitness_dimension(parental_fitness_values,sibling_fitness_values,fitness_dim,fitness_comb):

    if fitness_dim[0] is FitnessInstanceDim.parent_children and fitness_dim[1] is FitnessFuncDim.seperate:
        fitness_axis=0
    if fitness_dim[0] is FitnessInstanceDim.parent_sibling and fitness_dim[1] is FitnessFuncDim.single:
        fitness_axis=1
    if fitness_dim[0] is FitnessInstanceDim.single or (fitness_dim[0] is FitnessInstanceDim.parent_children
    and fitness_dim[1] is FitnessFuncDim.single):
        fitness_axis=None

    if not sibling_fitness_values:
        #if only a single child , there are no sibling fitness values
        fitness_axis=None if FitnessFuncDim.single else 0
        fitness_values=parental_fitness_values
        return combine_fitness(fitness_values,fitness_axis,fitness_comb)
    if fitness_dim[0] is FitnessInstanceDim.single:
        fitness_values=list(ut.flatten([parental_fitness_values,sibling_fitness_values]))
        return combine_fitness(fitness_values,fitness_axis,fitness_comb)

    if not (fitness_dim[0] is FitnessInstanceDim.parent_sibling and fitness_dim[1] is FitnessFuncDim.seperate):
        parental_fitness_values=combine_fitness(parental_fitness_values,fitness_axis,fitness_comb)
        sibling_fitness_values=combine_fitness(sibling_fitness_values,fitness_axis,fitness_comb)
    return list(ut.flatten([parental_fitness_values,sibling_fitness_values]))
#this method is used to combine the result from the data generation process
def format_generated_fitness(fitness,fitness_dim,fitness_comb):
    if ut.size(fitness[0])>1:
        return np.array([format_fitness_dimension(vals[0],vals[1],fitness_dim,fitness_comb) for vals in fitness])
    else:
        return np.array([format_fitness_dimension(vals,None,fitness_dim,fitness_comb) for vals in fitness])


def format_fitness_values_training(fitness_value_parent_child,fitness_value_sibling,siblings):
    if len(siblings)<2:
        return fitness_value_parent_child[siblings[0]]
    parental_fitness_values=[fitness_value_parent_child[child] for child in siblings]
    sibling_fitness_values=[fitness_value_sibling[child] for child in siblings[1:]]
    return parental_fitness_values,sibling_fitness_values

def format_fitness_targets_regression(parental_fitness,sibling_fitness,n_siblings):
    if n_siblings<2:
        return [parental_fitness[i].regression_target for i in range(len(parental_fitness))],None
    parental_fitness_targets=[parental_fitness[i].regression_target for i in range(len(parental_fitness))]
    parental_fitness_values=[parental_fitness_targets for _ in range(n_siblings)]
    sibling_fitness_targets=[sibling_fitness[i].regression_target for i in range(len(sibling_fitness))]
    sibling_fitness_values=[sibling_fitness_targets for _ in range(n_siblings-1)]

    return parental_fitness_values,sibling_fitness_values

def format_fitness_for_regression_conditioning(parental_fitness,sibling_fitness,n_siblings,data_size,fitness_dim):
    fitness_value_parental,fitness_value_sibling=format_fitness_targets_regression(parental_fitness,sibling_fitness,n_siblings)
    fitness=format_fitness_dimension(fitness_value_parental,fitness_value_sibling,
                                   fitness_dim,FitnessCombination.average)
    fitness_size= ut.size(fitness)
    indices=np.arange(data_size,data_size+fitness_size)
    return indices,fitness

#the order of the data is parent,sibling0,sibling1,..
#the order of the variable of each instance in the data is defined by the variable lists
def format_data_for_training(parent,parent_var_names,siblings,sibling_var_names):
    data=list(ut.flatten([parent.values["ind"][name] for name in parent_var_names]+[[child.values["ind"][name] for name in sibling_var_names] for child in siblings]))
    return data

#here is where the order of variables will be enforced
def concat_variables():
    pass
#all variables need to be of type numpy array
def split_variables(variables,joint_data):
    sizes=[v.size for v in variables]
    sizes.insert(0,0)
    lengths=np.cumsum(sizes)
    #this is for calculating the edges of the vector to be return in relative value
    return [np.array(joint_data[l1:l2]) if not var.frozen() else var.freeze_value
    for (l1,l2),var in zip(ut.pairwise(lengths),variables)]

