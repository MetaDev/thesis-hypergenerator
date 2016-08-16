import model.search_space as sp
import util.utility as ut
from model.search_space import StochasticVariable as SV
from model.search_space import VectorVariableUtility as VVU
from model.search_space import DeterministicVariable as DV

from model.search_space import LayoutTreeDefNode as LN
from model.search_space import VarDefNode

#the hierarchy argument is to evaluate the ability to train conditionally on parent values

#model to train polygon overlap
def model_var_pos():
    position = SV("position",low=(0,0),high=[5,3])
    rotation=SV("rotation",low=0,high=1)

    size=DV("size",(0.2,0.4))


    child = LN(name="child",size=size,position=position,
                                    rotation=rotation,shape=LN.shape_exteriors["square"])

    #irregular variable polygon parent form "<=-,"
    var_p0= SV("p0",low=(0,1),high=(2,2))

    parent_shape = VVU.from_variable_list("shape",
                       [var_p0, DV("p1",(1, 3)),DV("p2",(3, 3)),DV("p3",(3, 2)),DV("p4",(5, 2)),
                        DV("p5",(5, 0)),DV("p6",(4, 0)),DV("p7",(4, 1)),DV("p8",(3, 1)),DV("p9",(3, 0)),DV("p10",(1, 0))])
    n_children=SV("child",low=3,high=7)

    parent_room =  LN(name="parent",origin=DV("origin",(0,0)),size=DV("size",(1,1)),rotation=DV("rotation",[0]),
                                    position=DV("position",(0,0)),
                                    shape=parent_shape)
    children=[(n_children,child)]
    return parent_room.build_child_nodes(children),parent_room

#model to train closest side alignment (both between siblings and parent)
#variable rotation

def model_var_rot():


    #irregular variable polygon parent, in the form of a "<>"
    points=[None]*12
    points[0]= SV("",low=(0,1),high=(2,2))
    points[3]= SV("",low=(1,2),high=(3,4))
    points[6]= SV("",low=(2,1),high=(4,2))
    points[9]= SV("",low=(1,0),high=(3,1))

    points[1]= DV("",(1,2))
    points[2]= DV("",(1,2.5))
    points[4]= DV("",(3,2.5))
    points[5]= DV("",(3,2))
    points[7]= DV("",(3,1))
    points[8]= DV("",(3,0.5))
    points[10]= DV("",(1,.5))
    points[11]= DV("",(1,1))
    parent_shape = VVU.from_variable_list("shape",points)
    n_children=SV("child",low=3,high=7)


    rotation=SV("rotation",low=0,high=1)
    size=DV("size",(0.2,0.4))
    position = SV("position",low=(0,0),high=[4,4])
    child = LN(name="child",size=size,position=position,
                                    rotation=rotation,shape=LN.shape_exteriors["square"])
    parent_room =  LN(name="parent",origin=DV("origin",(0,0)),size=DV("size",(1,1)),rotation=DV("rotation",[0]),
                                position=DV("position",(0,0)),
                                shape=parent_shape)
    children=[(n_children,child)]
    parent_node=parent_room.build_child_nodes(children)




    return parent_node,parent_room

#model to train surface ratio
#model to train balance

#variable size
def model_var_size():

    #irregular variable polygon parent
    points=[None]*5
    points[2]= SV("",low=(2,2),high=(4,4))
    points[4]= SV("",low=(2,0),high=(4,-2))


    points[0]= DV("",(0,0))
    points[1]= DV("",(0,2))
    points[3]= DV("",(2,1))
    parent_shape = VVU.from_variable_list("shape",points)

    position = SV("position",low=(-1,-1),high=[4,3])
    rotation=DV("rotation",0)
    size=SV("size",(0.3,0.5),(0.6,1))
    n_children=SV("child",low=3,high=7)

    child = LN(name="child",size=size,position=position,
                                    rotation=rotation,shape=LN.shape_exteriors["square"])
    parent_room =  LN(name="parent",origin=DV("origin",(0,0)),size=DV("size",(1,1)),rotation=DV("rotation",[0]),
                                position=DV("position",(0,0)),
                                shape=parent_shape)
    children=[(n_children,child)]
    parent_node=parent_room.build_child_nodes(children)



    return parent_node,parent_room

#only position of the child changes
#fitness is parent overlap and min sibling distance
def simple_model():
    #test Distribution
    position = SV("position",low=(-1.5,-1.5),high=(1.5,1.5))
    rotation=DV("rotation",0)


    size = DV("size",(0.2,0.2))
    child = LN(name="child",size=size,position=position,
                                        rotation=rotation,shape=LN.shape_exteriors["square"])

    #var_p3= SV("p3",size=2,choices=SV.standard_choices(0.5,2,4))
    var_p3= SV("p3",low=(0.5,0.5),high=(2,2))
    parent_shape = VVU.from_variable_list("shape",
                       [DV("p0",(0, 0)), DV("p1",(0, 1)),DV("p2",(0.5,1)),
                        var_p3,DV("p4",(1, 0.5)),DV("p5",(1,0))])
    n_children=SV("child",low=3,high=7)
    parent_room = LN(size=DV("size",(1,1)),name="parent",rotation=DV("rotation",0),
                                    position=DV("position",(0,0)),
                                    shape=parent_shape)
    children=[(n_children,child)]
    parent_node=parent_room.build_child_nodes(children)
    return parent_node,parent_room

def model_methodology():
    position = SV("position",low=(2,3),high=[6,7])

    rotation=DV("rotation",0)
    size=SV("size",(0.2,0.2),(1,2))

    child = LN(name="child",size=size,position=position,
                                    rotation=rotation,shape=LN.shape_exteriors["square"])


    n_children=SV("child",low=2,high=4)
    parent_room =  LN(name="parent",size=DV("size",(3,5)),rotation=SV("rotation",low=0,high=180),
                                    position=SV("position",low=(1,1),high=[2,2]),
                                    shape=LN.shape_exteriors["square"])
    children=[(n_children,child)]
    return parent_room.build_child_nodes(children),parent_room