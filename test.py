import search_space
import mapping
import scipy.stats
import shapely
from shapely.geometry import Polygon

#test Distribution
d1 = search_space.Distribution(distr=scipy.stats.uniform,low=-4,high=4)
print (d1.sample())
d2 = search_space.Distribution(distr=scipy.stats.norm,low=0,high=4)
print (d2.sample())
d3 = search_space.Distribution(distr=scipy.stats.uniform,options=["red","green","blue"])
print (d3.sample())
d4 = search_space.Distribution(distr=scipy.stats.norm,nr_of_values=8)
print (d4.sample())


#test LayoutDefinition
rot = search_space.Distribution(distr=scipy.stats.uniform,low=0,high=360)

chair = search_space.LayoutDefinition(size=0.2,name="chair", position_x=d1,position_y=d1, rotation=rot,shape_exterior=search_space.LayoutDefinition.shape_exteriors["chair"], color="green")
table = search_space.LayoutDefinition(size=1,name="table",position_x=3,position_y=2,shape_exterior=search_space.LayoutDefinition.shape_exteriors["table"],children=[(4,chair)],color="blue")
table_and_chairs = table.create_sample()

#test visualisation
(xrange,yrange,polygons) = search_space.LayoutDefinition.visualise_samples(table_and_chairs)

#test fitness
graph = mapping.convert_to_neighbourhoud_graph(polygons,[xrange,yrange],step=(xrange[1]-xrange[0]+1,yrange[1]-yrange[0]+1))
mapping.draw_graph(graph)
print(xrange,yrange)
p=Polygon(search_space.LayoutDefinition.shape_exteriors["chair"])
