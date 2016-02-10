import matplotlib
matplotlib.use('TKAgg')

from matplotlib import pyplot

pyplot.close('all')
if(pyplot.gcf()==0):
    fig = pyplot.figure("main")
fig=pyplot.gcf()
pyplot.show()
pyplot.close('all')