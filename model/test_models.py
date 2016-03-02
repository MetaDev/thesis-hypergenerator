import model.search_space as sp
import util.utility as ut

def test_samples_var_child_position_parent_shape():
    #test Distribution
    child_position = sp.Distribution(low=-1.5,high=1.5)
    parent_shape=sp.Distribution(low=0.5,high=2,nr_of_values=4)
    colors = sp.Distribution(options=["red","green","blue"])

    shapes1= [(0, 0), (0, 1),(0.5,1),(parent_shape,parent_shape),(1, 0.5),(1,0)]

    chair = sp.LayoutMarkovTreeNode(size=(0.2,0.2),name="child",origin=[0.5,0.5] ,position=(child_position,child_position), rotation=0,shape=sp.LayoutMarkovTreeNode.shape_exteriors["square"], color=colors)
    table = sp.LayoutMarkovTreeNode(size=[1,1],name="parent",rotation=0,origin=[0.5,0.5],position=(1,2),shape=shapes1,children=[(5,chair)],color="blue")
    list_of_samples=[]
    root = table.sample(sample_list=list_of_samples)
    return list_of_samples,root
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