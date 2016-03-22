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


root = tm.test_model_var_child_position_parent_shape()
n=5
root_samples = root.sample(n)

for i in range(n):
    root_sample=root_samples[i]
    vis.init()
    ax=vis.get_ax(None)
    sample_list=root_sample.get_flat_list()


    polygons = mp.map_layoutsamples_to_geometricobjects(sample_list,"shape")
    #colors = [s.relative_vars["color"] for s in sample_list]
    (xrange,yrange) = ut.range_from_polygons(polygons,size=1.3)
    step=ut.nr_of_steps_from_range(xrange,yrange,step_size=0.5)
    vis.draw_node_sample_tree(root_sample)

    #test visualisation of search space mapping
    graph = mp.map_polygons_to_neighbourhoud_graph(polygons,[xrange,yrange],step=step)
    vis.draw_graph(ax,graph)
    children =[v for v in sample_list if v.name.startswith("child")]
    chairs = [polygons[i]  for i in range(len(sample_list)) if sample_list[i].name.startswith("child")]
    table = [polygons[i]  for i in range(len(sample_list)) if sample_list[i].name.startswith("parent")][0]
    chair_paths = fn.polygon_path_sequence(graph,chairs)
    vis.draw_graph_path(ax,graph,chair_paths)
    #for child in chairs:
    #    print(child)
    #    print("alignment angle:",fn.pairwise_closest_line_alignment(child,table))
#    for childpair in combinations(children,2):
#        print("dist",fn.pairwise_min_dist(childpair[0].values["ind"]["position"],childpair[1].values["ind"]["position"],2))

    centroids=[np.array(p.centroid) for p in polygons]
    points=ut.extract_samples_vars(sample_list,var_name="position",independent=False)
    x,y=zip(*points)
    ax.scatter(x,y,color="g")
    ax.scatter(*zip(*centroids),color="r")
    vis.plt.show()
