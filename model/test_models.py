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


    child_size = DV("size",(0.2,0.2))
    chair = LN(size=child_size,name="child" ,position=child_position,
                                    rotation=DV("rotation",0),shape=LN.shape_exteriors["square"], color=colors)

    #var_p3= SV("p3",size=2,choices=SV.standard_choices(0.5,2,4))
    var_p3= SV("p3",low=[2]*2,high=[5]*2)
    parent_shapes = VVU.from_variable_list("shape",
                       [DV("p0",(0, 0)), DV("p1",(0, 1)),DV("p2",(0.5,1)),
                        var_p3,DV("p4",(1, 0.5)),DV("p5",(1,0))])
    parent_pos=DV("position",(0,0))
    n_children=CV("child",low=5,high=8)
    #n_children=DV("child",5)
    parent_size=DV("size",(1,1))
    colors = DV("color",(0,0,1))
    table_def = LN(size=parent_size,name="parent",rotation=DV("rotation",0),
                                    position=parent_pos,
                                    shape=parent_shapes,children=[(n_children,chair)],color=colors)
    table_node = table_def.build_tree()
    return table_node,table_def
