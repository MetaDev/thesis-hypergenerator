import learning.train_gmm as tgmm


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

save_test_out(tgmm.fitness_cap_test,3)


