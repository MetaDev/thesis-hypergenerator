# -*- coding: utf-8 -*-
import numpy as np
import model.mapping as mp
import model.fitness as fn
import util.data_format as dtfr

import numpy as np
from itertools import combinations,permutations
import util.utility as ut

#the order variable should be a string matching either a parent fitness func method or a string matching a variable name
def data_generation(n_samples,root,parental_fitness,sibling_fitness,
                                parent_vars,sibling_vars,child_name="child",
                                sibling_order=0,sibling_window=None,
                                sibling_train_order=SiblingTrainReOrder.no_reorder,
                                order_variable=None,respect_sibling_order=False,
                                multi_objective=False):
     #the order of the data is parent,sibling0,sibling1,..
    #the order of the variable of each instance in the data is defined by the variable lists
    data=dict([(i,[]) for i in np.arange(root.variables[child_name].low,root.variables[child_name].high)])
    #fitness order parent_funcs,sibling_funcs
    fitness=[]
    root_samples=root.sample(n_samples)

    #only get children of a certain name
    for i in range(n_samples):
        root_sample=root_samples[i]
        #these children are ordered, the sibling window is respected
        children=root_sample.children[child_name]
        parent_child_data()


def parent_sibling_data(data,fitness,parent,children,parental_fitness,sibling_fitness,
                                parent_vars,sibling_vars,child_name="child",
                                sibling_order=1,
                                multi_objective=False):


        #the number of siblings trained on cannot be bigger than the number of children
        n_siblings=sibling_order+1
        if n_siblings>len(children):
            raise ValueError("Sibling order + 1 cannot be larger than the number of siblingss")

        for consecutive_siblings in ut.window(children,n_siblings):
            #first check if all children are uncapped
            #if the one of the children is capped don't learn combination
            if any(len(fitness_value_parent_child[child])!=len(parental_fitness) for child in consecutive_siblings):
                break
            #save the parent child fitness per child
            fitness_values_siblings=dict([(child,[fitness_value_parent_child[child]])for child in consecutive_siblings])
            #calculate the sibling fitness depending on whether the ordering of the children is respected or not
            #the ordering is important if there is a possibility to learn on a seperate fitness for each sibling, this is however not the case for single objective
            #if single objecti

            #the parent fitness of siblings is their respective product, has already been capped
            fitness_value_sibling_parents=np.prod([fitness_value_parent_child[child]
                                                            for child in childcombinations],0)
            #the sibling fitness of siblings is the pairwise fitness procuct
            for func,order,cap in sibling_fitness:
                #calcualte fitness with previous siblings
                for child_index in range(len(childcombinations)):
                    fitness_func_sibling=[]
                    for prev_sibling_index in range(child_index):
                        temp_sibling_fitness=func(child0,child1)**order
                        capped[child0]=temp_sibling_fitness<cap
                        capped[child1]=temp_sibling_fitness<cap
                        if not temp_sibling_fitness < cap:
                            fitness_func_sibling.append(temp_sibling_fitness)
                        else:
                            break
                #a set of siblings is invalid if one of its fitness funcs is below the cap
                if any(capped[child] is True for child in childcombinations):
                    break
                #if each sibling past the cap add product to final fitness
                fitness_values_siblings.append(fitness_func_sibling)
            if any(capped[child] is True for child in consecutive_siblings):
                break
            #add both the data and the fitness values to the final result if not capped
            #this can be either for a single specific ordering or all possible orderings
            #the fitness values are products of the fitness values of each sibling
            #this is because it is the ensemble of siblings that needs to be fit
            fitness_values=np.concatenate((fitness_value_sibling_parents,fitness_values_siblings))


            data.append(dtfr.format_data_for_traing(root_sample,parent_vars,
                                                            consecutive_siblings,sibling_vars))
            #TODO
            #if it is not multi-objective, combine all fitness
            #reorder fitness in the same way
            fitness.append(fitness_values)

    return np.array(data),np.array(fitness)




#order the list of children
def sort_children_by_variable(children,var_name,fitness_values):
    #variable can be a vector
    values = tuple(zip(*[c.values["ind"][var_name] for c in children]))
    indices = np.lexsort(values)
    return [(children[i],fitness_values[i]) for i in indices]
def sort_children_by_fitness(children,fitness_order_index,fitness_values):
    return [(c,f) for (c,f) in sorted(zip(children,fitness_values),
             key=lambda pair: pair[1][fitness_order_index])]

#TODO random order
def fitness_polygon_overl(sample0,sample1):
    polygons=mp.map_layoutsamples_to_geometricobjects([sample0,sample1],shape_name="shape")
    return fn.pairwise_overlap(polygons[0],polygons[1])

def fitness_polygon_alignment(sample0,sample1):
    polygons=mp.map_layoutsamples_to_geometricobjects([sample0,sample1],shape_name="shape")
    return fn.pairwise_closest_line_alignment(polygons[0],polygons[1],threshold=30)
def fitness_min_dist(sample0,sample1):
    pos0=sample0.values["ind"]["position"]
    pos1=sample1.values["ind"]["position"]
    return fn.pairwise_min_dist(pos0,pos1,threshold=1)