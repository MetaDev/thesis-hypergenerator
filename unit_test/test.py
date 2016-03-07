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



def angle(p1, p2):
    return np.rad2deg(np.arctan2((p2[1]-p1[1]),(p2[0]-p1[0])))%180

p1=Polygon([(0,0),(0,1),(1,1),(1,0)])
print(p1.exterior)
p2=Polygon([(1,1),(2,1),(2,3),(1,2)])
for l0,l1 in ut.pairwise(p2.exterior.coords):
    print(angle(l0,l1),l0,l1)

#        print(angle(l1,l0))
#closest_lines=(None,None)
#min_dist=infinity
#for line0 in polygon_pair[0].exterior:
#    for line1 in polygon_pair[0].exterior:
#        if line1.distance1(line0) > min_dist:
#            pass

#root = tm.test_model_var_child_position_parent_shape()
##test poylygon mapping
#sample_list=[]
#sample_root=root.sample(sample_list=sample_list)
#polygons = mp.map_layoutsamples_to_geometricobjects(sample_list,"shape")
##colors = ut.extract_samples_attributes(result,attr_name="color")
#vis.init()
#ax=vis.plt.gca()
#(xrange,yrange) = ut.range_from_polygons(polygons,size=1.3)
#step=ut.nr_of_steps_from_range(xrange,yrange,step_size=0.5)
##
#vis.draw_polygons(ax,polygons,set_range=True)
##
##test visualisation of search space mapping
#graph = mp.map_polygons_to_neighbourhoud_graph(polygons,[xrange,yrange],step=step)
#vis.draw_graph(ax,graph)
#
#chairs = [polygons[i]  for i in range(len(sample_list)) if sample_list[i].name.startswith("child")]
#chair_paths = fn.polygon_path_sequence(graph,chairs)
#vis.draw_graph_path(ax,graph,chair_paths)
#print( str(fn.pairwise_dist(ut.extract_samples_vars(sample_list,
#                                                    var_name="position",sample_name="child"))))
#
#centroids=[np.array(p.centroid) for p in polygons]
#points=ut.extract_samples_vars(sample_list,var_name="position",independent=False)
#print(points)
#x,y=zip(*points)
#ax.scatter(x,y,color="g")
#ax.scatter(*zip(*centroids),color="r")
