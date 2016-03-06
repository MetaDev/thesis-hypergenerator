import model.search_space as sp
import util.utility as ut
from model.search_space import StochasticVariable as SV
from model.search_space import VectorVariableUtility as VVU
from model.search_space import DeterministicVariable as DV

from model.search_space import LayoutMarkovTreeNode as LN
def test_model_var_child_position_parent_shape():
    #test Distribution
    child_position = SV("position",size=2,distr=SV.standard_distr(-1.5,1.5),func=LN.position_rel)
    colors = SV("color",size=3,choices=SV.standard_choices(num=3))


    child_size = DV("size",(0.2,0.2))
    origin= LN.default_origin
    chair = LN(size=child_size,name="child",origin=origin ,position=child_position,
                                    rotation=DV("rotation",0),shape=LN.shape_exteriors["square"], color=colors)

    var_p3= SV("p3",size=2,choices=SV.standard_choices(0.5,2,4))
    parent_shapes = VVU.from_variable_list("shape",
                       [DV("p0",(0, 0)), DV("p1",(0, 1)),DV("p2",(0.5,1)),
                        var_p3,DV("p4",(1, 0.5)),DV("p5",(1,0))])
    parent_pos=DV("position",(1,2))
    n_children=DV("",5)
    parent_size=DV("size",(1,1))
    table = LN(size=parent_size,name="parent",rotation=DV("rotation",0),
                                    origin=origin,position=parent_pos,
                                    shape=parent_shapes,children=[(n_children,chair)],color=colors)
    return table

def test_samples_var_child_pos_size_rot():
    #test Distribution
    child_position = sp.Distribution(low=-1.5,high=1.5)
    child_size = sp.Distribution(low=0.1,high=0.4)
    child_rotation = sp.Distribution(low=0,high=360)

    chair = sp.LayoutMarkovTreeNode(size=(child_size,child_size),name="child",origin=[0.5,0.5] ,position=(child_position,child_position), rotation=child_rotation,shape=sp.LayoutMarkovTreeNode.shape_exteriors["square"], color="green")
    table = sp.LayoutMarkovTreeNode(size=[1,1],name="parent",rotation=0,origin=[0.5,0.5],position=(0,0),shape=sp.LayoutMarkovTreeNode.shape_exteriors["square"],children=[(5,chair)],color="blue")
    list_of_samples=[]
    root = table.sample(sample_list=list_of_samples)
    return list_of_samples,root
#todo make it work with larger hierachy-> multiple parents, use the root (recursion)
#attrs is a tuple of tuples (parent_name,attr_names),(child_name,attr_names) each tuple is of the form (string, [string])
#def sample_attributes_parent_child(sample_method,n_samples,attrs):
#    nodes_attr=[]
#    for i in range(n_samples):
#        node_sample_list,root=sample_method()
#        node=root
#        #move hierarchical structure
#        #no while user recursion
##        while node.children[0] != 0:
##            for child in node.children[1]:
#
#        #iterate all desired attributes from nodes to extract
#        for node in attrs:
#            (node_name,node_attr_names)= *node
#            #TODO group nodes per parent
#            #extract each node of that type
#            nodes=ut.extract_samples_attributes(node_sample_list,sample_name=node_name)
#            for node in nodes:
#                node_attr=[]
#                for attr_name in node_attr_names:
#                    node_attr.append(node.attr_name)
#            #can return multiple attr, if the node_sample_list contains multiple nodes with that name
#            #make list of attribute vector per node
#            for a in attr:

def test_samples_var_parent_nchildren():
    pass