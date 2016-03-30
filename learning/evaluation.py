# -*- coding: utf-8 -*-
from scipy.stats import binned_statistic
import numpy as np
import util.utility as ut

#calculate MSE of GMM

#calculate fitness statistics
def fitness_statistics(fitness_values,verbose=False,summary=True):
    fitness_statistics_count, fitness_bin_edges,_ = binned_statistic(fitness_values, fitness_values,
                                                                     statistic='count', bins=20)
    if not summary:
        fitness_statistics_mean = binned_statistic(fitness_values, fitness_values,  statistic='mean'
        , bins=20)[0]
        all_edges=[]
        counts=[]
        means=[]
        for edges,count,mean,i in zip(ut.pairwise(fitness_bin_edges),fitness_statistics_count,
            fitness_statistics_mean,range(len(fitness_statistics_count))):

            edges ="bin nr. "+ str(i) + ": "+format(edges[0], '.5g') + " - "+format(edges[1], '.5g')
            count = ": count= " + str(count)
            mean=": mean= "+ format(mean, '.5g')
            if verbose:
                print(edges + count + mean)
            all_edges.append(edges)
            counts.append(count)
            means.append(mean)
        return means,all_edges,counts
    else:
        if verbose:
            print("total mean= {0:f}: variance= {1:f}".format(np.mean(fitness_values),np.var(fitness_values),'.5g'))
        return np.mean(fitness_values),np.var(fitness_values)
