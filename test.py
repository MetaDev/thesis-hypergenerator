import search_space
import mapping
import visualisation
import utility
import fitness
import numpy as np


#
result=[]
result,root = search_space.test()
##print([str(r) for r in result])
#test poylygon mapping
polygons = mapping.map_layoutsamples_to_geometricobjects(result)
colors = utility.extract_samples_attributes(result,attr_name="color")
ax=visualisation.init()
(xrange,yrange) = utility.range_from_polygons(polygons,size=1.3)
step=utility.nr_of_steps_from_range(xrange,yrange,step_size=0.5)

visualisation.draw_polygons(ax,polygons,colors=colors,set_range=False)

#test visualisation of search space mapping
graph = mapping.map_polygons_to_neighbourhoud_graph(polygons,[xrange,yrange],step=step)
visualisation.draw_graph(ax,graph)

chairs = [polygons[i]  for i in range(len(result)) if result[i].name.startswith("chair")]
chair_paths = fitness.polygon_path_sequence(graph,chairs)
visualisation.draw_graph_path(ax,graph,chair_paths)
print("Total overlapping surface: "+ str(fitness.surface_overlap(polygons)))
print("total distance between chairs: "+ str(fitness.pairwise_dist(utility.extract_samples_attributes(result,attr_name="position",sample_name="chair"))))

points=[np.array(p.centroid) for p in polygons]
print(points)
x,y=zip(*points)

print(x,y)
print(utility.extract_samples_attributes(result,attr_name="position"))
ax.scatter(x,y,color="g")


visualisation.finish()