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



save_test_out(tgmm.test_min_covar,10)

save_test_out(tgmm.test_n_component,11)

save_test_out(tgmm.test_n_iter_n_trial,12)

save_test_out(tgmm.test_poisson,13)

save_test_out(tgmm.test_fitness_dim_regression,14)

save_test_out(tgmm.test_model_var,15)

save_test_out(tgmm.test_n_model_eval,16)





#print back to normal
sys.stdout = sys.__stdout__

