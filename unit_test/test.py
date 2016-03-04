import model.search_space as sp
import model.mapping as mp
import util.visualisation as vis
import util.utility as ut
import model.fitness as fn
import numpy as np
import model.test_models as tm
from model.search_space import VectorVariable as VV

from model.search_space import StochasticVariable as SV
from model.search_space import DeterministicVariable as DV

print("a"+ "\n"+b)
result=[]
result,root = tm.test_samples_var_child_position_parent_shape()
#test poylygon mapping
polygons = mp.map_layoutsamples_to_geometricobjects(result)
#colors = ut.extract_samples_attributes(result,attr_name="color")
vis.init()
ax=vis.plt.gca()
(xrange,yrange) = ut.range_from_polygons(polygons,size=1.3)
step=ut.nr_of_steps_from_range(xrange,yrange,step_size=0.5)
#
vis.draw_polygons(ax,polygons,set_range=True)
#
#test visualisation of search space mapping
graph = mp.map_polygons_to_neighbourhoud_graph(polygons,[xrange,yrange],step=step)
vis.draw_graph(ax,graph)

chairs = [polygons[i]  for i in range(len(result)) if result[i].name.startswith("child")]
chair_paths = fn.polygon_path_sequence(graph,chairs)
vis.draw_graph_path(ax,graph,chair_paths)
print( str(fn.pairwise_dist(ut.extract_samples_vars(result,var_name="position",sample_name="child"))))

centroids=[np.array(p.centroid) for p in polygons]
points=ut.extract_samples_vars(result,var_name="position",independent=False)
print(points)
x,y=zip(*points)
ax.scatter(x,y,color="g")
ax.scatter(*zip(*centroids),color="r")
