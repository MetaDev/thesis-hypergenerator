import model.mapping as mp
import model.fitness as fn
import util.data_format as dtfr

import numpy as np
from itertools import combinations,permutations
import util.utility as ut

#todo each fitness should return the name of the variables it depends upon
#this is the minimal set for learning, more variables can be added
#also each variable should be checked wether it's stochastic
#when re-learning the order of the siblings will be set and should be respected->don't use combinations
#the use of combinations instead of more data still needs to be validated
#sibling order 1 is child-parent relation only
#the fitness funcs should include their order
from enum import Enum
class SiblingTrainReOrder(Enum):
    reorder_fitness = 1
    reorder_value = 2
    no_reorder = 3
    random = 4
    any_order = 5


#this data generation is tailored for weighted sampling
def data_generation(n_samples,root,parental_fitness,sibling_fitness,
                                parent_var_names,sibling_var_names,child_name="child",
                                sibling_order=1,
                                sibling_train_order=SiblingTrainReOrder.no_reorder,
                                order_variable=None,respect_sibling_order=False):
    #the order of the data is parent,sibling0,sibling1,..
    data=[]
    #fitness order parent_funcs,sibling_funcs
    fitness=[]
    root_samples=root.sample(n_samples)

    #only get children of a certain name
    for i in range(n_samples):
        root_sample=root_samples[i]
        children=root_sample.children[child_name]
        data_parent=np.array([root_sample.values["ind"][name] for name in parent_var_names]).flatten()

        #first calculate all parent->child fitness funcs
        fitness_value_parent_child={}
        for child in children:
            fitness_value_parent_child[child]=[]
            for func,order,cap in parental_fitness:
                temp_fitness_parent_child=func(child,root_sample)**order
                capped =  temp_fitness_parent_child < cap
                if not temp_fitness_parent_child < cap:
                    fitness_value_parent_child[child].append(temp_fitness_parent_child)
                else:
                    break

        #the number of siblings trained on cannot be bigger than the number of children
        n_siblings=sibling_order+1

        #if the order of the sibling needs to be respected we can not reuse that data by training
        #on all possible combinations of siblings, because the siblings are no longer independent from their order
        #for example after training the n order markov chain between siblings
        if respect_sibling_order:
            sibling_sequences=[children]
        else:
            sibling_sequences=combinations(children,n_siblings)
        for childcombinations in sibling_sequences:
            fitness_values_siblings=[]
            #calculate complete pairwise fitness

            #first check if all children are uncapped
            #if the one of the children is capped don't learn combination
            if any(len(fitness_value_parent_child[child])!=len(parental_fitness) for child in childcombinations):
                break
            #the parent fitness of siblings is their respective product, has already been capped
            fitness_value_sibling_parents=np.prod([fitness_value_parent_child[child]
                                                            for child in childcombinations],0)
            #the sibling fitness of siblings is the pairwise fitness procuct
            for func,order,cap in sibling_fitness:
                fitness_func_sibling=[]
                for child0,child1 in combinations(childcombinations,2):
                    temp_sibling_fitness=func(child0,child1)**order
                    capped=temp_sibling_fitness < cap
                    if not capped:
                        fitness_func_sibling.append(temp_sibling_fitness)
                    else:
                        break
                #a set of siblings is invalid if one of its fitness funcs is below the cap
                if capped:
                    break
                #if each sibling past the cap add product to final fitness
                fitness_values_siblings.append(np.prod(fitness_func_sibling))
            #a set of siblings is invalid if one of its fitness funcs is below the cap
            if capped:
                break
            #add both the data and the fitness values to the final result if not capped
            #this can be either for a single specific ordering or all possible orderings
            fitness_values=np.concatenate((fitness_value_sibling_parents,fitness_values_siblings))

            if sibling_train_order is SiblingTrainReOrder.any_order:
                for childrenperm in permutations(childcombinations):
                    data_sibling=np.array([np.array([child.values["ind"][name]
                            for name in sibling_var_names]).flatten() for child in childrenperm]).flatten()
                    data.append(np.concatenate((data_parent,data_sibling)))
                    fitness.append(fitness_values)
            else:
                if sibling_train_order is SiblingTrainReOrder.reorder_fitness:
                    fitness_index=next((i for i, fitness in enumerate(parental_fitness)
                                                    if fitness[0].__name__ == order_variable), -1)
                    if fitness_index is not -1:
                        #fitnes value of each child in child combinations
                        fitness_order_values=[fitness_value_parent_child[child][fitness_index]
                                                                for child in childcombinations]

                        childcombinations=sort_children_by_fitness(childcombinations,fitness_order_values)
                    else:
                        raise Exception("Fitness function to sort on not found")
                elif sibling_train_order is SiblingTrainReOrder.reorder_value:
                    childcombinations=sort_children_by_variable(childcombinations,order_variable)
                elif sibling_train_order is SiblingTrainReOrder.random:
                    import random
                    random.shuffle(list(childcombinations))
                data_siblings=np.array([np.array([child.values["ind"][name]
                            for name in sibling_var_names]).flatten() for child in childcombinations ]).flatten()
                data.append(np.concatenate((data_parent,data_siblings)))
                fitness.append(fitness_values)

    return np.array(data),np.array(fitness)




#order the list of children
def sort_children_by_variable(children,var_name):
    #variable can be a vector
    values = tuple(zip(*[c.values["ind"][var_name] for c in children]))
    indices = np.lexsort(values)
    return [children[i] for i in indices]
def sort_children_by_fitness(children,fitness_values):
    return [(c,f) for (c,f) in sorted(zip(children,fitness_values),
             key=lambda pair: fitness_values[1])]

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
