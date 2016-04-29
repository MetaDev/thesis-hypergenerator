import model.search_space as sp
import util.utility as ut
from model.search_space import StochasticVariable as SV
from model.search_space import VectorVariableUtility as VVU
from model.search_space import DeterministicVariable as DV

from model.search_space import LayoutTreeDefNode as LN
from model.search_space import VarDefNode

#the hierarchy argument is to evaluate the ability to train conditionally on parent values

#model to train polygon overlap
def model_var_pos(hierarchy,var_rot,var_nr_children):
    position = SV("position",low=(0,0),high=[5,3])
    if var_rot:
        rotation=SV("rotation",low=0,high=359)
    else:
        rotation=DV("rotation",0)
    size=DV("size",(0.2,0.4))


    child = LN(name="child",size=size,position=position,
                                    rotation=rotation,shape=LN.shape_exteriors["square"])

    #irregular variable polygon parent form "<=-,"
    if hierarchy:
        var_p0= SV("p0",low=(0,1),high=(2,2))
    else:
        var_p0= DV("p0",(0,1))
    parent_shape = VVU.from_variable_list("shape",
                       [var_p0, DV("p1",(1, 3)),DV("p2",(3, 3)),DV("p3",(3, 2)),DV("p4",(5, 2)),
                        DV("p5",(5, 0)),DV("p6",(4, 0)),DV("p7",(4, 1)),DV("p8",(3, 1)),DV("p9",(3, 0)),DV("p10",(1, 0))])
    if var_nr_children:
        n_children=SV("child",low=3,high=7)
    else:
        n_children=DV("child",5)
    parent_room =  LN(name="parent",origin=DV("origin",(0,0)),size=DV("size",(1,1)),rotation=DV("rotation",[0]),
                                    position=DV("position",(0,0)),
                                    shape=parent_shape)
    children=[(n_children,child)]
    return parent_room.build_child_nodes(children),parent_room

#model to train closest side alignment (both between siblings and parent)
#variable rotation

def model_var_rot(var_parent,var_pos,var_nr_children):


    #irregular variable polygon parent, in the form of a "<>"
    points=[None]*12
    if var_parent:
        points[0]= SV("",low=(0,1),high=(2,2))
        points[3]= SV("",low=(1,2),high=(3,4))
        points[6]= SV("",low=(2,1),high=(4,2))
        points[9]= SV("",low=(1,0),high=(3,1))
    else:
        points[0]= DV("",(1,1.5))
        points[3]= DV("",(2,3))
        points[6]= DV("",(3,1.5))
        points[9]= DV("",(2,1.5))
    points[1]= DV("",(1,2))
    points[2]= DV("",(1,2.5))
    points[4]= DV("",(3,2.5))
    points[5]= DV("",(3,2))
    points[7]= DV("",(3,1))
    points[8]= DV("",(3,0.5))
    points[10]= DV("",(1,.5))
    points[11]= DV("",(1,1))
    parent_shape = VVU.from_variable_list("shape",points)
    if var_nr_children:
        n_children=SV("child",low=3,high=7)
    else:
        n_children=DV("child",5)

    rotation=SV("rotation",low=0,high=359)
    size=DV("size",(0.2,0.4))
    if var_pos:
        position = SV("position",low=(0,0),high=[4,4])
        child = LN(name="child",size=size,position=position,
                                        rotation=rotation,shape=LN.shape_exteriors["square"])
        parent_room =  LN(name="parent",origin=DV("origin",(0,0)),size=DV("size",(1,1)),rotation=DV("rotation",[0]),
                                    position=DV("position",(0,0)),
                                    shape=parent_shape)
        children=[(n_children,child)]
        parent_node=parent_room.build_child_nodes(children)
    else:
        parent_room =  LN(name="parent",origin=DV("origin",(0,0)),size=DV("size",(1,1)),rotation=DV("rotation",[0]),
                                    position=DV("position",(0,0)),
                                    shape=parent_shape)

        child_list=[None]* (4)
        child= LN(name="child",size=size,position=DV("position",(1.5,2.5)),
                                        rotation=rotation,shape=LN.shape_exteriors["square"])
        child_list[0]=VarDefNode(child)
        child= LN(name="child",size=size,position=DV("position",(2.5,2.5)),
                                        rotation=rotation,shape=LN.shape_exteriors["square"])
        child_list[1]=VarDefNode(child)
        child= LN(name="child",size=size,position=DV("position",(1.5,1.5)),
                                        rotation=rotation,shape=LN.shape_exteriors["square"])
        child_list[2]=VarDefNode(child)
        child= LN(name="child",size=size,position=DV("position",(2.5,1.5)),
                                        rotation=rotation,shape=LN.shape_exteriors["square"])
        child_list[3]=VarDefNode(child)
        parent_node = VarDefNode(parent_room)

        n_children= (SV("child",low=2,high=4) if var_nr_children else DV("child",4))
        parent_node.add_children(child_list,n_children)



    return parent_node,parent_room

#model to train surface ratio
#model to train balance

#variable size
def model_var_size(var_parent,var_pos,var_nr_children):

    #irregular variable polygon parent
    points=[None]*5
    if var_parent:
        points[2]= SV("",low=(2,2),high=(4,4))
        points[4]= SV("",low=(2,0),high=(4,-2))
    else:
        points[2]= DV("",(3,3))
        points[4]= DV("",(3,-1))

    points[0]= DV("",(0,0))
    points[1]= DV("",(0,2))
    points[3]= DV("",(2,1))
    parent_shape = VVU.from_variable_list("shape",points)

    position = SV("position",low=(0,0),high=[4,3])
    rotation=DV("rotation",0)
    size=SV("size",(0.3,0.5),(0.6,1))
    if var_nr_children:
        n_children=SV("child",low=3,high=7)
    else:
        n_children=DV("child",5)
    if var_pos:
        position = SV("position",low=(0,0),high=[3,2])
        child = LN(name="child",size=size,position=position,
                                        rotation=rotation,shape=LN.shape_exteriors["square"])
        parent_room =  LN(name="parent",origin=DV("origin",(0,0)),size=DV("size",(1,1)),rotation=DV("rotation",[0]),
                                    position=DV("position",(0,0)),
                                    shape=parent_shape)
        children=[(n_children,child)]
        parent_node=parent_room.build_child_nodes(children)
    else:
        parent_room =  LN(name="parent",origin=DV("origin",(0,0)),size=DV("size",(1,1)),rotation=DV("rotation",[0]),
                                    position=DV("position",(0,0)),
                                    shape=parent_shape)

        child_list=[None]* (4)
        child= LN(name="child",size=size,position=DV("position",(1,2)),
                                        rotation=rotation,shape=LN.shape_exteriors["square"])
        child_list[0]=VarDefNode(child)
        child= LN(name="child",size=size,position=DV("position",(2,2)),
                                        rotation=rotation,shape=LN.shape_exteriors["square"])
        child_list[1]=VarDefNode(child)
        child= LN(name="child",size=size,position=DV("position",(1,0)),
                                        rotation=rotation,shape=LN.shape_exteriors["square"])
        child_list[2]=VarDefNode(child)
        child= LN(name="child",size=size,position=DV("position",(2,0)),
                                        rotation=rotation,shape=LN.shape_exteriors["square"])
        child_list[3]=VarDefNode(child)
        parent_node = VarDefNode(parent_room)
        n_children= (SV("child",low=2,high=4) if var_nr_children else DV("child",4))
        parent_node.add_children(child_list,n_children)


    return parent_node,parent_room

#model to train polygon convexity
#variable parent only
def model_var_parent():

    #irregular variable polygon parent
    points=[None]*12
    points[0]= SV("",low=(0,1),high=(2,2))
    points[1]= DV("",(1,2))
    points[2]= DV("",(1,2.5))
    points[3]= SV("",low=(1,2),high=(3,4))
    points[4]= DV("",(3,2.5))
    points[5]= DV("",(3,2))
    points[6]= SV("",low=(2,1),high=(4,2))
    points[7]= DV("",(3,1))
    points[8]= DV("",(3,0.5))
    points[9]= SV("",low=(1,0),high=(3,1))
    points[10]= DV("",(1,.5))
    points[11]= DV("",(1,1))

    parent_shape = VVU.from_variable_list("shape",points)


    parent =  LN(name="parent",origin=DV("origin",(0,0)),size=DV("size",(1,1)),rotation=DV("rotation",[0]),
                                    position=DV("position",(0,0)),
                                    shape=parent_shape)
    parent_node = VarDefNode(parent)
    return parent_node,parent

#model for reachability
#long parent, var rot and/or var pos

def model_reachabillity(var_pos):

    #irregular variable polygon parent
    points=[None]*4
    points[0]= DV("",(0,0))
    points[1]= DV("",(0,8))
    points[2]= DV("",(2,8))
    points[3]= DV("",(2,0))
    parent_shape = VVU.from_variable_list("shape",points)


    position = SV("position",low=(0,0),high=[4,3])
    rotation=SV("rotation",low=0,high=359)
    size=DV("size",(0.4,1))

    n_children=DV("child",8)
    if var_pos:
        position = SV("position",low=(0,0),high=[2,8])
        child = LN(name="child",size=size,position=position,
                                        rotation=rotation,shape=LN.shape_exteriors["square"])

        parent =  LN(name="parent",origin=DV("origin",(0,0)),size=DV("size",(1,1)),rotation=DV("rotation",[0]),
                                    position=DV("position",(0,0)),
                                    shape=parent_shape)
        children=[(n_children,child)]
        parent_node=parent.build_child_nodes(children)
    else:
        parent =  LN(name="parent",origin=DV("origin",(0,0)),size=DV("size",(1,1)),rotation=DV("rotation",[0]),
                                    position=DV("position",(0,0)),
                                    shape=parent_shape)

        child_list=[None]* (8)
        for i in range(4):
            child= LN(name="child",size=size,position=DV("position",(0.5,1+2*i)),
                                        rotation=rotation,shape=LN.shape_exteriors["square"])
            child_list[2*i]=VarDefNode(child)
            child= LN(name="child",size=size,position=DV("position",(1.5,1+2*i)),
                                        rotation=rotation,shape=LN.shape_exteriors["square"])
            child_list[2*i+1]=VarDefNode(child)

        parent_node = VarDefNode(parent)
        parent_node.add_children(child_list,n_children)

    return parent_node,parent
#TODO
#to show the scalability in number of children of the appraoach
def model_var_children(nr_of_children):
    pass