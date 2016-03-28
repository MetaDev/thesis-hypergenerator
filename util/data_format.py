#Centralise the ordering of variables and children for traing and sampling (marginalising, conditional)
#and the order of the data as well (child and parent variables)
#each variable definition and child definition (not sample) should have an id indicating it's order

#in this class the order can be checked of both variables and samples
import util.utility as ut
import numpy as np


#conditioning format for trained GMM: P(S_i|P,S_0,...,S_i-1) given parent and all siblings
def format_data_for_conditional(parent_sample,parent_vars,sibling_samples,sibling_vars,sibling_order):
    #flatten list for calculation of cond distr
    #parent condition variable
    parent_values=np.array([parent_sample.values["ind"][var.name] for var in parent_vars]).flatten()
    #sibling condition variable
    sibling_values=[]
    #only retrieve values of necessary siblings,for sibling order i take last i siblings
    for sibling in sibling_samples[-sibling_order:]:
        sibling_values.extend(np.array([sibling.values["ind"][var.name] for var in sibling_vars]).flatten())
    #the order of values is by convention, also enforced in the sampling and learning procedure
    values=np.concatenate((parent_values,sibling_values))
    #the first X are cond attributes
    indices=np.arange(0,len(values))
    return values,indices

def marginalise_gmm(gmm_full,parent_vars,sibling_vars,sibling_order):
    #a model trained on 4 children can also be used for 5 children of the fifth no longer conditions on the first
    #These models can be reused because there's no difference in the markov chain of order n between the n+1 and n+2 state

    #if sibling order is 0 you have only a single gmm
    gmms=[None]*(sibling_order+1)
    gmms[sibling_order]=gmm_full
    #marginalise for each child
    #the gmm is trained on data of the form parent,sibling0,sibling1,..
    #thus to find GMM[i] we need to marginalise out the sibling data beyond sibling i
    #GMM[i]=P(p,c0,c1,c2,..,ci)
    for i in range(sibling_order):
        indices=np.arange(0,(sibling_order+1)*variables_length(sibling_vars)+variables_length(parent_vars))
        #gmms[i]=markov chain order i
        gmms[i]=gmm_full.marginalise(indices)
    return gmms

def variables_length(variables):
    return np.sum([var.size for var in variables])

def format_data_for_training():
    pass
def extract_data_for_traing(children,parent):
    pass
#here is where the order of variables will be enforced
def concat_variables():
    pass
def split_variables(variables,joint_data):
    lengths=[v.size for v in variables]
    #this is for calculating the edges of the vector to be return in relative value
    lengths.insert(0,0)
    return [joint_data[l1:l2] for l1,l2 in ut.pairwise(lengths)]

