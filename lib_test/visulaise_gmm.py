# -*- coding: utf-8 -*-

#TODO
#visualisation code needs cleaning up
#(xrange,yrange)=((-2,2),(-2,2))
#
#parent_position=np.array([1,2])
#
#(xrange,yrange)=((-2,4),(-1,5))
#p_shape_points=[0.5,1,1.5]
#for p_shape in zip(p_shape_points,p_shape_points):
#    shape=[(0, 0), (0, 1),(0.5,1),p_shape,(1, 0.5),(1,0)]
#
#    polygon = mp.map_to_polygon(shape,[0.5,0.5],parent_position,0,(1,1))
#
#    previous_gmm_cond=None
#    from copy import deepcopy
#    #visualise the conditional distribution of children
#    #P(c_i,c_i-1,..,c_0,p) -> P(c_i|c_i-1,..,c_0,p)
#    values=p_shape
#    points=[]
#    for i in range(sibling_order+1):
#
#        ax = plt.gca()
#        ax.set_aspect(1)
#        ax.set_xlim(*xrange)
#        ax.set_ylim(*yrange)
#        print("child: ",i)
#        indices=np.arange(X_var_length,(i+1)*X_var_length+Y_var_length)
#
#        if previous_gmm_cond:
#            sample=np.array(previous_gmm_cond.sample(1)).flatten()
#            values=np.hstack((np.array(sample),np.array(values)))
#            point= np.array(sample) + np.array(parent_position)
#            points.append(point)
#            x,y=zip(*points)
#            ax.scatter(x,y,color="r")
#
#        values=np.array(values).flatten()
#        gmm_cond=gmms[i].condition(indices,values)
#        previous_gmm_cond=gmm_cond
#
#        gmm_show=deepcopy(gmm_cond)
#        gmm_show._means=[list(map(add, c, np.array(parent_position))) for c in gmm_show.means_]
#        gmm_show._set_params_gmr()
#        vis.visualise_gmm_marg_2D_density(ax,gmm=gmm_show,colors=["g"])
#
#        vis.draw_polygons([polygon],ax)
#        plt.show()