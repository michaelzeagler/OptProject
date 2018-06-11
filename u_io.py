# Feedback linearizing control law for a relative degree 2 system
def u_IO(xst, ydes, yddes, f, g, h_x, hd_x, dLie, poles):
    
    
    l2fh  = dLie*f
    lglfh = dLie*g

    e_y = h_x  - ydes
    e_yd =hd_x - yddes
    
    
    nu   = e_y.multiply_elementwise(poles[:,0]) + \
        e_yd.multiply_elementwise(poles[:,1])

    u = inverse(lglfh)*(l2fh - nu)

    return u