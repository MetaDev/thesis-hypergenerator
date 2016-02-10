# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:53:27 2015

@author: Harald
"""
#polygon imports

from matplotlib import pyplot
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely import affinity

from descartes.patch import PolygonPatch

#learn imports
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture


#polygon stuff
COLOR = {
    True:  '#6699cc',
    False: '#ff3333'
    }

def v_color(ob):
    return COLOR[ob.is_valid]


def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, 'o', color='#999999', zorder=1)
def make_ellipses(gmm, fig):
    for n in range(len(gmm._get_covars())):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color='#999999')
        ell.set_clip_box(fig.gca().bbox)
        ell.set_alpha(0.5)
        fig.gca().add_artist(ell)
#doesnt work
#pyplot.close('all')
#pyplot.clf()
def test():

    if(pyplot.gcf()==0):
        fig = pyplot.figure("main")
    fig=pyplot.gcf()
    fig.clear()
    # 1: valid polygon

    ax = fig.add_subplot(111)

    ext = [(0, 0), (0, 2), (1, 2),(1,1),(2,1),(2,0)]
    polygon = Polygon(ext)
    polygon1 = affinity.translate(polygon,-2,-2,0)
    plot_coords(ax, polygon.exterior)

    patch1 = PolygonPatch(polygon, facecolor=v_color(polygon), edgecolor=v_color(polygon), alpha=0.5, zorder=1)
   
    ax.add_patch(patch1)
    patch2 = PolygonPatch(polygon1, facecolor=v_color(polygon1), edgecolor=v_color(polygon1), alpha=0.5, zorder=1)
   
    ax.add_patch(patch2)

    ax.set_title('a) valid')

    xrange = [-3, 3]
    yrange = [-3, 3]
    ax.set_xlim(*xrange)
    ax.set_xticks(list(range(*xrange)) + [xrange[-1]])
    ax.set_ylim(*yrange)
    ax.set_yticks(list(range(*yrange)) + [yrange[-1]])
    ax.set_aspect(1)



    #learn stuff

    # Number of samples per component
    n_samples = 1000
    max_point=-3
    min_point=3

    # Generate random sample, two components
    np.random.seed(0)
    points= [np.random.uniform(min_point,max_point,2) for _ in range(n_samples)]
    X= [np.append(p,polygon.contains(Point(p))) for p in points]
    #X=[np.append(p,polygon.distance(Point(p))) for p in points]


    # Fit a mixture of Gaussians with EM using five components
    gmm = mixture.GMM(n_components=3, covariance_type='full')


    insideX = [x[:2] for x in X if x[2] ==1]

    gmm.fit(insideX)
    make_ellipses(gmm,fig)
    #plot the points
    #plt.scatter(*zip(*points), c=list(zip(*X))[2])
    plt.scatter(*zip(*insideX))
    plt.scatter(*zip(*gmm.sample(100)),color='r')
    plt.show()
    plt.pause(1) # <-------

    input("<Hit Enter To Show generated samples>")
    plt.close(fig)
    
    
    plt.show()