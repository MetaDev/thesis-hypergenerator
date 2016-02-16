import search_space
import mapping

import visualisation
import utility
import fitness

#test search space
result = search_space.test()
#test poylygon mapping
polygons,colors = mapping.map_samples_to_polygons(result)
ax=visualisation.init()
(xrange,yrange) = utility.range_from_polygons(polygons,size=1.3)
step=utility.nr_of_steps_from_range(xrange,yrange,step_size=0.5)

visualisation.draw_polygons(ax,polygons)
#test visualisation of search space mapping
graph = mapping.map_polygons_to_neighbourhoud_graph(polygons,[xrange,yrange],step=step)
visualisation.draw_graph(ax,graph)
chairs = [polygons[i]  for i in range(len(result)) if result[i].name.startswith("chair")]
chair_paths = fitness.polygon_path_sequence(graph,chairs)
visualisation.draw_graph_path(ax,graph,chair_paths)
visualisation.finish()