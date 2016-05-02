5# -*- coding: utf-8 -*-

#here comes the get to try out the performance of different training hyperparameters


import learning.train_gmm as tgmm

tgmm.training(n_data=100,n_iter=5,n_trial=5,regression=False)