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
import rx
from rx import Observable, Observer
from itertools import combinations
from rx.subjects import Subject

#create class that generates 100 random values and X subscribers that print it
#create on "larger" observer that collects all emitted data

#root is a subject

root = Subject()
class Root_collector(Observer):
    def __init__(self):
        self.roots=[]
    def on_next(self,x):
        self.roots.append(x)
    def on_error(self, e):
        print("Got error: %s" % e)

    def on_completed(self):
        pass
rc=Root_collector()
n_children=5
n_samples=20
#first subscribe all it's children, filter out the amount of children

#merge all leaf children(no children of their own) in a single observer to check when completed

leaves=Observable.empty()
for i in range(7):
    child= root.filter(
        lambda x, i: i<n_children
    ).map(
        lambda x, i: x * 2
    )
    leaves=leaves.merge(child)
class End_Of_Sampling(Observer):
    def __init__(self,root_collector):
        self.rc=root_collector
    def on_error(self, e):
        print("Got error: %s" % e)

    def on_completed(self):
        print("start training")
        for i in self.rc.roots:
            print(i)
#wait for oncomplete or error
root.subscribe(rc)
test = root.to_blocking().subscribe(print)
#leaves.ignore_elements().subscribe(End_Of_Sampling(rc))
#when sampling is called the values are calculated
for i in range(n_samples):
    #pass parent sample in onnext
    root.on_next(i)

#do the training in this subscription
#leaves.ignore_elements().subscribe(print(rc.roots))


#
#root = tm.test_model_var_child_position_parent_shape()
#
#
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
#vis.draw_polygons(ax,polygons,set_range=False)
##
##test visualisation of search space mapping
#graph = mp.map_polygons_to_neighbourhoud_graph(polygons,[xrange,yrange],step=step)
##vis.draw_graph(ax,graph)
#children =[v for v in sample_list if v.name.startswith("child")]
#chairs = [polygons[i]  for i in range(len(sample_list)) if sample_list[i].name.startswith("child")]
#table = [polygons[i]  for i in range(len(sample_list)) if sample_list[i].name.startswith("parent")][0]
#chair_paths = fn.polygon_path_sequence(graph,chairs)
##vis.draw_graph_path(ax,graph,chair_paths)
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
