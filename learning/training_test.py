# -*- coding: utf-8 -*-

#here comes the get to try out the performance of different training hyperparameters


import learning.train_gmm as tgmm

#save all prints
import sys

def save_test_out(test_method,test_id):
    test_name="test"+str(test_id)
    sys.stdout = open(test_name+".txt", "w")


    test_method()
    import matplotlib.backends.backend_pdf
    import matplotlib.pyplot as plt

    pdf = matplotlib.backends.backend_pdf.PdfPages(test_name+".pdf")

    for i in plt.get_fignums():
        plt.figure(i)
        pdf.savefig( plt.gcf() )

    pdf.close()
    plt.close("all")

#save_test_out(tgmm.fitness_order_MO_test,2)

#save_test_out(tgmm.fitness_cap_test,3)
#
#save_test_out(tgmm.fitness_regression_condition_test,4)
#
#save_test_out(tgmm.test_sibling_order_seq,5)
#
#save_test_out(tgmm.test_model_4_target,7)
#
#
#save_test_out(tgmm.test_marginal_gmm,8)
#
#save_test_out(tgmm.test_n_data,9)





#print back to normal
sys.stdout = sys.__stdout__
