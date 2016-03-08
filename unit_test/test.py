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



root = tm.test_model_var_child_position_parent_shape()
#test poylygon mapping
sample_list=[]
sample_root=root.sample(sample_list=sample_list)
polygons = mp.map_layoutsamples_to_geometricobjects(sample_list,"shape")
#colors = ut.extract_samples_attributes(result,attr_name="color")
vis.init()
ax=vis.plt.gca()
(xrange,yrange) = ut.range_from_polygons(polygons,size=1.3)
step=ut.nr_of_steps_from_range(xrange,yrange,step_size=0.5)
#
vis.draw_polygons(ax,polygons,set_range=False)
#
#test visualisation of search space mapping
graph = mp.map_polygons_to_neighbourhoud_graph(polygons,[xrange,yrange],step=step)
#vis.draw_graph(ax,graph)

chairs = [polygons[i]  for i in range(len(sample_list)) if sample_list[i].name.startswith("child")]
table = [polygons[i]  for i in range(len(sample_list)) if sample_list[i].name.startswith("parent")][0]
chair_paths = fn.polygon_path_sequence(graph,chairs)
#vis.draw_graph_path(ax,graph,chair_paths)
for child in chairs:
    print(fn.pairwise_closest_line_alignment(child,table))
    print(fn.pairwise_overlap(child,table))

centroids=[np.array(p.centroid) for p in polygons]
points=ut.extract_samples_vars(sample_list,var_name="position",independent=False)
x,y=zip(*points)
ax.scatter(x,y,color="g")
ax.scatter(*zip(*centroids),color="r")
vis.plt.show()
