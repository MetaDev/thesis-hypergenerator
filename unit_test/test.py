import model.search_space as sp
import model.mapping as mp
import util.visualisation as vis
import util.utility as ut
import model.fitness as fn
import numpy as np
import model.test_models as tm
from shapely.geometry import *
import numpy as np
import numpy.linalg as la
from itertools import combinations
import model.sampling.poisson as poisson


n_children=CV("child",low=5,high=8)
root_node,root_def = tm.model_reachabillity(True)
n=5
root_samples = root_node.sample(n)

for i in range(n):
    root_sample=root_samples[i]
    vis.init()
    ax=vis.get_ax(None)
    sample_list=root_sample.get_flat_list()


    polygons = mp.map_layoutsamples_to_geometricobjects(sample_list,"shape")
    (xrange,yrange) = ut.range_from_polygons(polygons,size=1.3)
    step=ut.nr_of_steps_from_range(xrange,yrange,step_size=0.5)
    vis.draw_node_sample_tree(root_sample,color="b")


    children =[v for v in sample_list if v.name.startswith("child")]
    chairs = [polygons[i]  for i in range(len(sample_list)) if sample_list[i].name.startswith("child")]
    table = [polygons[i]  for i in range(len(sample_list)) if sample_list[i].name.startswith("parent")][0]

    centroids=[np.array(p.centroid) for p in polygons]
    points=ut.extract_samples_vars(sample_list,var_name="position",independent=False)
    x,y=zip(*points)
    ax.scatter(x,y,color="g")
    ax.scatter(*zip(*centroids),color="r")
    vis.plt.show()
