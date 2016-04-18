import model.search_space as sp
import util.utility as ut
from model.search_space import StochasticVariable as SV
from model.search_space import VectorVariableUtility as VVU
from model.search_space import DeterministicVariable as DV

from model.search_space import NumberChildrenVariable as CV
from model.search_space import LayoutTreeDefNode as LN
def test_model_var_child_position_parent_shape():
    #test Distribution
    child_position = SV("position",low=[-2]*2,high=[2]*2)

    colors = DV("color",(0,1,0))


    child_size = DV("size",(0.2,0.4))
    rotation=SV("rotation",low=0,high=359)
    chair = LN(size=child_size,name="child" ,position=child_position,
                                    rotation=rotation,shape=LN.shape_exteriors["square"], color=colors)

    var_p3= SV("p3",low=[2]*2,high=[5]*2)
    parent_shapes = VVU.from_variable_list("shape",
                       [DV("p0",(0, 0)), DV("p1",(0, 1)),DV("p2",(0.5,1)),
                        var_p3,DV("p4",(1, 0.5)),DV("p5",(1,0))])
    parent_pos=DV("position",(0,0))
    n_children=CV("child",low=5,high=8)
    parent_size=DV("size",(1,1))
    colors = DV("color",(0,0,1))
    table_def = LN(size=parent_size,name="parent",rotation=DV("rotation",[0]),
                                    position=parent_pos,
                                    shape=parent_shapes,children=[(n_children,chair)],color=colors)
    table_node = table_def.build_tree()
    return table_node,table_def
def model_1():
    position = SV("position",low=(0,0),high=[5,3])
    rotation=SV("rotation",low=0,high=359)
    size=SV("size",(0.1,0.2),(0.2,0.4))


    child = LN(name="child",size=size,position=position,
                                    rotation=rotation,shape=LN.shape_exteriors["square"])

    #irregular variable polygon parent
    var_p0= SV("p0",low=(0,1),high=(2,2))
    parent_shape = VVU.from_variable_list("shape",
                       [var_p0, DV("p1",(1, 3)),DV("p2",(3, 3)),DV("p3",(3, 2)),DV("p4",(5, 2)),
                        DV("p5",(5, 0)),DV("p6",(4, 0)),DV("p7",(4, 1)),DV("p8",(3, 1)),DV("p9",(3, 0)),DV("p10",(1, 0))])
    n_children=CV("child",low=5,high=8)
    parent_room =  LN(name="parent",origin=DV("origin",(0,0)),size=DV("size",(1,1)),rotation=DV("rotation",[0]),
                                    position=DV("position",(0,0)),
                                    shape=parent_shape,children=[(n_children,child)])
    return parent_room.build_tree(),parent_room


