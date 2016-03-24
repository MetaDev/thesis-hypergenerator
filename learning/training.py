import model.mapping as mp
import model.fitness as fn

import numpy as np
from itertools import combinations
import util.utility as ut

#todo each fitness should return the name of the variables it depends upon
#this is the minimal set for learning, more variables can be added
#also each variable should be checked wether it's stochastic
#when re-learning the order of the siblings will be set and should be respected->don't use combinations
#the use of combinations instead of more data still needs to be validated
#sibling order 1 is child-parent relation only
#the fitness funcs should include their order

def parent_child_variable_training(n_samples,root,parental_fitness,sibling_fitness,
                                vars_parent,vars_children,child_name="child",sibling_order=0):
    #the order of the data is child,parent,sibling0,sibling1,..
    data=[]
    #fitness order parent_funcs,sibling_funcs
    fitness=[]
    root_samples=root.sample(n_samples)

    #only get children of a certain name
    for i in range(n_samples):
        root_sample=root_samples[i]
        children=root_sample.children[child_name]
        parent_vars=np.array([root_sample.values["ind"][name] for name in vars_parent]).flatten()

        capped=dict([(child,False) for child in children])
        data_child={}
        #first calculate all parent->child fitness funcs
        parent_child_fitness={}
        for child in children:
            #the order of the data is child,parent,sibling0,sibling1,..
            parent_child_fitness[child]=[]
            for func,order,cap in parental_fitness:
                parent_fitness=func(child,root_sample)**order
                capped[child] = capped[child] and parent_fitness < cap
                if not capped[child]:
                    parent_child_fitness[child].append(parent_fitness)
                else:
                    break
            #data -> child,parent
            if not capped[child]:
                child_vars=np.array([child.values["ind"][name] for name in vars_children]).flatten()
                data_child[child]=np.concatenate((child_vars,parent_vars))

        # if the sibling order is 0, only the child parent fitness is learned
        if sibling_order is 0:
            #only non capped data is saved
            data.extend([data_child[child] for child in children])
            fitness.extend([parent_child_fitness[child] for child in children])
         #sibling fitness only necessary if the order of the siblings in the model is > 0
        else:
            #TODO add combinations iteration where the order is respected, eg window combinations
            #todo allow traing with and without combinations, and if sibling order is > nr of children skip

            #calculate sibling funcs
            #for all possible combinations of lenght sibling order of existing children
            for childcombinations in combinations(children,sibling_order+1):
                child_comb_data={}
                child_comb_fitness={}
                for child in childcombinations:
                    #if the one of the children is capped don't learn combination
                    if capped[child]:
                        break
                    else:
                        #the data that corresponds with this fitness child,parent
                        child_comb_data[child]=np.array(data_child[child]).tolist()
                        #sibling fitness for each child in the combination
                        child_comb_fitness[child]=[]
                        for func,order,cap in sibling_fitness:
                            fitness_siblings=[]
                            #calculate pairwise fitness between children
                            for sibling in (sibling for sibling in childcombinations if sibling is not child):
                                #calc fitness multiplied by all sublings
                                capped[child]=func(child,sibling)**order < cap
                                if not capped[child]:
                                    #fitness f1,f2,.. for each child combination
                                    fitness_siblings.append(func(child,sibling)**order)
                                    #add sibling to data
                                    #data-> child,parent,sibling0,...
                                    child_comb_data[child].extend(np.array([sibling.values["ind"][name]
                                    for name in vars_children]).flatten())
                                else:
                                    break
                                if capped[child]:
                                    break
                            if not capped[child]:
                                #append for each child it's sibling func
                                child_comb_fitness[child].append(ut.prod(fitness_siblings))
                            else:
                                break
                #if none of the children in the combination are capped, add data and fitness
                if all(cap is False for cap in capped.values()):
                    #add uncapped data
                    data.extend([child_comb_data[child] for child in childcombinations])
                    fitness.extend([parent_child_fitness[child]+child_comb_fitness[child] for child in childcombinations])

    return np.array(data),np.array(fitness)



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
