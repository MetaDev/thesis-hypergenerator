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


upper=np.array([2,3,4])
lower=np.array([1,2,3])



 # user defined options
# this parameter defines if we look for Poisson-like distribution on a disk/sphere (center at 0, radius 1) or in a square/box (0-1 on x and y)
disk = False
# this parameter defines if we look for "repeating" pattern so if we should maximize distances also with pattern repetitions
repeatPattern = False
num_points = 8              # number of points we are looking for
num_iterations = 4          # number of iterations in which we take average minimum squared distances between points and try to maximize them
first_point_zero = disk     # should be first point zero (useful if we already have such sample) or random
iterations_per_point = 128  # iterations per point trying to look for a new point with larger distance
sorting_buckets = 0         # if this option is > 0, then sequence will be optimized for tiled cache locality in n x n tiles (x followed by y)
num_dim = 3               # 1, 2, 3 dimensional version
num_rotations = 1           # number of rotations of pattern to check against

poisson_generator = poisson.PoissonGenerator(num_dim,lower,upper,disk, repeatPattern, first_point_zero)
points = poisson_generator.find_point_set(num_points, num_iterations, iterations_per_point, num_rotations)
print(points)
print(poisson_generator.format_points_string(points))


#
#root = tm.test_model_var_child_position_parent_shape()
#root.sample()
#
#vis.init()
#ax=vis.get_ax(None)
#sample_list=root.get_flat_list()
#print(len(sample_list))
#
#
#polygons = mp.map_layoutsamples_to_geometricobjects(sample_list,"shape")
##colors = [s.relative_vars["color"] for s in sample_list]
#(xrange,yrange) = ut.range_from_polygons(polygons,size=1.3)
#step=ut.nr_of_steps_from_range(xrange,yrange,step_size=0.5)
#vis.draw_node_sample_tree(root)
#
##test visualisation of search space mapping
#graph = mp.map_polygons_to_neighbourhoud_graph(polygons,[xrange,yrange],step=step)
#vis.draw_graph(ax,graph)
#children =[v for v in sample_list if v.name.startswith("child")]
#chairs = [polygons[i]  for i in range(len(sample_list)) if sample_list[i].name.startswith("child")]
#table = [polygons[i]  for i in range(len(sample_list)) if sample_list[i].name.startswith("parent")][0]
#chair_paths = fn.polygon_path_sequence(graph,chairs)
#vis.draw_graph_path(ax,graph,chair_paths)
##for child in chairs:
##    print(child)
##    print("alignment angle:",fn.pairwise_closest_line_alignment(child,table))
#for childpair in combinations(children,2):
#    print("dist",fn.pairwise_min_dist(childpair[0].independent_vars["position"],childpair[1].independent_vars["position"],2))
#
#centroids=[np.array(p.centroid) for p in polygons]
#points=ut.extract_samples_vars(sample_list,var_name="position",independent=False)
#x,y=zip(*points)
#ax.scatter(x,y,color="g")
#ax.scatter(*zip(*centroids),color="r")
#vis.plt.show()
