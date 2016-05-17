5# -*- coding: utf-8 -*-

#here comes the get to try out the performance of different training hyperparameters


import learning.train_gmm as tgmm

save_output=True

test_id=2
test_name="test"+str(test_id)


#save output to txt
if save_output:
    import sys
    sys.stdout = open(test_name+".txt", "w")
else:
    sys.stdout = sys.__stdout__


#execute tests

tgmm.fitness_order_MO_test()
#print("model_1")
tgmm.training_model_1()

#print("model_2")
#
#tgmm.training_model_2()
#print("model_3")
#
#tgmm.training_model_3()
#print("model_4")
#
#tgmm.training_model_4()



#save all figures to pdf
if save_output:
    import matplotlib.backends.backend_pdf
    import matplotlib.pyplot as plt

    pdf = matplotlib.backends.backend_pdf.PdfPages(test_name+".pdf")
    for i in plt.get_fignums():
        plt.figure(i)
        pdf.savefig( plt.gcf() )
    pdf.close()


