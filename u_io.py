# Feedback linearizing control law for a relative degree 2 system
def u_IO(xst, ydes, yddes, f_x, g_x, h_x, hd_x, dLie, poles):
    
    # computing Lie derivatives
    l2fh  = dLie*f_x
    lglfh = dLie*g_x
    
    # output errors 
    e_y = h_x  - ydes
    e_yd =hd_x - yddes
    
    # constrolled system dynamics definition
    nu   = e_y.multiply_elementwise(poles[:,0]) + \
        e_yd.multiply_elementwise(poles[:,1])
    
    u = inverse(lglfh)*(l2fh - nu)

    return u