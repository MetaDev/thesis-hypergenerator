import model.search_space as sp
import util.utility as ut
from model.search_space import StochasticVariable as SV
from model.search_space import VectorVariableUtility as VVU
from model.search_space import DeterministicVariable as DV

from model.search_space import LayoutMarkovTreeNode as LN
def test_model_var_child_position_parent_shape():
    #test Distribution
    child_position = SV("position",size=2,distr=SV.standard_distr(-1.5,1.5))
    colors = DV("color",(0,0,0))


    child_size = DV("size",(0.2,0.2))
    chair = LN(size=child_size,name="child" ,position=child_position,
                                    rotation=DV("rotation",0),shape=LN.shape_exteriors["square"], color=colors)

    #var_p3= SV("p3",size=2,choices=SV.standard_choices(0.5,2,4))
    var_p3= SV("p3",size=2,distr=SV.standard_distr(0.5,2))
    parent_shapes = VVU.from_variable_list("shape",
                       [DV("p0",(0, 0)), DV("p1",(0, 1)),DV("p2",(0.5,1)),
                        var_p3,DV("p4",(1, 0.5)),DV("p5",(1,0))])
    parent_pos=DV("position",(1,2))
    n_children=DV("children",5)
    parent_size=DV("size",(1,1))
    table = LN(size=parent_size,name="parent",rotation=DV("rotation",0),
                                    position=parent_pos,
                                    shape=parent_shapes,children=[(n_children,chair)],color=colors)
    return table
def test_model_var_child_position_parent_shape_reduced():
    #test Distribution
    child_position = SV("position",size=2,choices=SV.standard_choices(-1.5,1.5,0.05),func=LN.position_rel)
    colors = DV("color",(0,0,0))


    child_size = DV("size",(0.2,0.2))
    origin= LN.default_origin
    chair = LN(size=child_size,name="child",origin=origin ,position=child_position,
                                    rotation=DV("rotation",0),shape=LN.shape_exteriors["square"], color=colors)

    #var_p3= SV("p3",size=2,choices=SV.standard_choices(0.5,2,4))
    var_p3= SV("p3",size=2,distr=SV.standard_distr(0.5,2))
    parent_shapes = VVU.from_variable_list("shape",
                       [DV("p0",(0, 0)), DV("p1",(0, 1)),DV("p2",(0.5,1)),
                        var_p3,DV("p4",(1, 0.5)),DV("p5",(1,0))])
    parent_pos=DV("position",(1,2))
    n_children=DV("children",5)
    parent_size=DV("size",(1,1))
    table = LN(size=parent_size,name="parent",rotation=DV("rotation",0),
                                    origin=origin,position=parent_pos,
                                    shape=parent_shapes,children=[(n_children,chair)],color=colors)
    return table

def test_samples_var_child_pos_size_rot_parent_shape():
    #test Distribution
    child_position = SV("position",size=2,distr=SV.standard_distr(-1.5,1.5),func=LN.position_rel)
    colors = SV("color",size=3,choices=SV.standard_choices(num=3))
    child_size = SV("size",size=2,distr=SV.standard_distr(0.1,0.4))
    child_rotation=SV("rotation",distr=SV.standard_distr(0,359))
    origin= LN.default_origin

    chair = LN(size=child_size,name="child",origin=origin ,position=child_position,
                                    rotation=child_rotation,shape=LN.shape_exteriors["square"], color=colors)

    var_p3= SV("p3",size=2,distr=SV.standard_distr(0.5,2))
    var_p0= SV("p0",size=2,distr=SV.standard_distr(-2,0.5))
    parent_shapes = VVU.from_variable_list("shape",
                       [var_p0, DV("p1",(0, 1)),DV("p2",(0.5,1)),
                        var_p3,DV("p4",(1, 0.5)),DV("p5",(1,0))])
    parent_pos=DV("position",(1,2))
    n_children=DV("",5)
    parent_size=DV("size",(1,1))
    table = LN(size=parent_size,name="parent",rotation=DV("rotation",0),
                                    origin=origin,position=parent_pos,
                                    shape=parent_shapes,children=[(n_children,chair)],color=colors)

    return table


def test_samples_var_parent_nchildren():
    pass