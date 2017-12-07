
# coding: utf-8

# In[97]:

# defines a 3D, 4DOF, human configured arm from the shoulder to wrist


# In[98]:

from sympy import init_printing
init_printing()

from sympy import symbols, cse, Function, ccode, trigsimp

#from sympy import cxxcode

from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point,kinetic_energy, potential_energy, inertia, RigidBody, Lagrangian, LagrangesMethod, init_vprinting, msubs
init_vprinting()

from sympy.utilities.iterables import flatten

from sympy.physics.matrices import Matrix

from sympy.utilities.lambdify import lambdify, implemented_function

import numpy


# In[99]:

# reference frames

GroundFrame = ReferenceFrame('F_g')

link01Frame    = ReferenceFrame('F_{1}')
link02Frame    = ReferenceFrame('F_{2}')
link03Frame    = ReferenceFrame('F_{3}')
link04Frame    = ReferenceFrame('F_{4}')



# In[100]:

# coordinates and states
th1,th2,th3,th4         = dynamicsymbols('theta1,theta2,theta3,theta4')
th1d,th2d,th3d,th4d     = dynamicsymbols('theta1,theta2,theta3,theta4',1)
th1dd,th2dd,th3dd,th4dd = dynamicsymbols('theta1,theta2,theta3,theta4',2)

om1,om2,om3,om4         = dynamicsymbols('omega1,omega2,omega3,omega4')
omd1,omd2,omd3,omd4     = dynamicsymbols('omega1,omega2,omega3,omega4',1)


q   = Matrix([th1,  th2,  th3,  th4])

qd  = Matrix([th1d, th2d, th3d, th4d])
#qd  = Matrix([om1, om2, om3, om4])

qdd = Matrix([th1dd,th2dd,th3dd,th4dd])
#qdd = Matrix([omd1, omd2, omd3, omd4])

x   = Matrix([*q, *qd])
xd  = Matrix([*qd,*qdd])


# In[101]:

# aligning frames

link01Frame.orient(GroundFrame, 'Axis',(th1,GroundFrame.z))
link02Frame.orient(link01Frame, 'Axis',(th2,link01Frame.x))
link03Frame.orient(link02Frame, 'Axis',(th3,link02Frame.z))
link04Frame.orient(link03Frame, 'Axis',(th4,link03Frame.x))

link01Frame.set_ang_vel(GroundFrame, th1d*GroundFrame.z)
link02Frame.set_ang_vel(link01Frame, th2d*link01Frame.x)
link03Frame.set_ang_vel(link02Frame, th3d*link02Frame.z)
link04Frame.set_ang_vel(link03Frame, th4d*link03Frame.x)


# In[102]:

# defining joint positions

shoulder = Point('shoulder') 
shoulder.set_vel(GroundFrame,0)

elbow = Point('elbow')
#UpArmLen = .2
UpArmLen = Symbol('L_{ua}')
elbow.set_pos(shoulder , UpArmLen * link02Frame.z)
elbow.v2pt_theory(shoulder,GroundFrame,link02Frame);

wrist = Point('wrist')
wrist.set_pos(elbow , UpArmLen*link04Frame.z);
wrist.v2pt_theory(elbow,GroundFrame,link04Frame);

#wrist.pos_from(shoulder)
#V_w = wrist.vel(GroundFrame)
#V_w


# In[103]:

# setting positions of centers of mass

l1com = Point('l1com')
l2com = Point('l2com')
l3com = Point('l3com')
l4com = Point('l4com')

l1com.set_pos(shoulder, 0)
l2com.set_pos(shoulder, .45*UpArmLen*link02Frame.z)
l3com.set_pos(shoulder, .55*UpArmLen*link03Frame.z)
l4com.set_pos(elbow   , .5 *UpArmLen*link04Frame.z)


l1com.v2pt_theory(shoulder,GroundFrame,link01Frame);
l2com.v2pt_theory(shoulder,GroundFrame,link02Frame);
l3com.v2pt_theory(shoulder,GroundFrame,link03Frame);
l4com.v2pt_theory(elbow   ,GroundFrame,link04Frame);


# In[104]:

# link masses and inertias

# base/shoulder
m_b = symbols('m_b')
h = symbols('h_b')
d = symbols('d_b')
#jzshoulder = 1/2*m_b*(d/2)**2 # z inertia
jzshoulder = Symbol('J_{z,shoulder}')

# rod and tube
m_rod = symbols('m_r')
l = symbols('l')
r = symbols('r')

#Jx = 1/12*m_rod*l**2 # length inertia
#Jz = 1/2*m_rod*r**2 # solid rod - radial
Jx = Symbol('J_x')
Jz = Symbol('J_z')

#Jzhollow = m_rod*r**2 # hollow tube - radial

# inertia dyadics
JShoulder = inertia(link01Frame,0,0,jzshoulder,ixy=0,iyz=0,izx=0)
JUpper    = inertia(link02Frame,Jx,Jx,Jz,      ixy=0,iyz=0,izx=0)
JLower    = inertia(link03Frame,Jx,Jx,Jz,      ixy=0,iyz=0,izx=0)
JFore     = inertia(link04Frame,Jx,Jx,Jz,      ixy=0,iyz=0,izx=0)

# JFore #
# JFore.to_matrix(link04Frame) # produces a tensor from the dyadic


# In[105]:

# Defining rigid body classes

link01 = RigidBody('Shoulder',l1com, link01Frame,m_b  , (JShoulder,l1com))
link02 = RigidBody('UpperArm',l2com, link02Frame,m_rod, (JUpper   ,l2com))
link03 = RigidBody('LowerArm',l3com, link03Frame,m_rod, (JLower   ,l3com))
link04 = RigidBody('ForeArm' ,l4com, link04Frame,m_rod, (JFore    ,l4com))


# In[106]:

# Lagrangian

g = Symbol('g')

link01.potential_energy = link01.mass * GroundFrame.y.dot(l1com.pos_from(shoulder)) * g
link02.potential_energy = link02.mass * GroundFrame.y.dot(l2com.pos_from(shoulder)) * g
link03.potential_energy = link03.mass * GroundFrame.y.dot(l3com.pos_from(shoulder)) * g
link04.potential_energy = link04.mass * GroundFrame.y.dot(l4com.pos_from(shoulder)) * g


# In[107]:

L = Lagrangian(GroundFrame,link01,link02,link03,link04)


# In[108]:

sys = LagrangesMethod(L,q)
#sys = LagrangesMethod(L,q,forcelist=u_list,frame=GroundFrame)


# In[114]:

# defining nonlinear system

sys.form_lagranges_equations();

D = sys.mass_matrix;
H = sys.forcing;


#f_b = sys.forcing;
#B = f_b.jacobian(u)
#H = f_b - B*u


#f_x = D.inv()*H;
#g_x = D.inv()*B;


# In[110]:

#print(cse(H))


# In[111]:

# defining observer outputs

h1 = trigsimp(wrist.pos_from(shoulder).dot(GroundFrame.x))
h2 = trigsimp(wrist.pos_from(shoulder).dot(GroundFrame.y))
h3 = trigsimp(wrist.pos_from(shoulder).dot(GroundFrame.z))

h = Matrix([h1,h2,h3])

dhdx = h.jacobian(x)
Lfh  = dhdx*xd
dLfhdx = Lfh.jacobian(x)


# In[112]:

# defining actuator torques & B matrix

tau1,tau2,tau3,tau4 = dynamicsymbols('tau1,tau2,tau3,tau4')

link01torques = tau1*link01Frame.z
link02torques = tau2*link02Frame.x - tau3*link02Frame.z
link03torques = tau3*link03Frame.z - tau4*link03Frame.x
link04torques = tau4*link04Frame.x

u = Matrix([tau1,tau2,tau3,tau4])
u_list = [(link01Frame,link01torques),(link02Frame,link02torques),
(link03Frame,link03torques),(link04Frame,link04torques)]

E_nc = u.T*q
B = (E_nc.jacobian(u)).jacobian(q)


# In[122]:

# functions for ODE solutions

#x0,x1,x2,x3,x4,x5,x6,x7 = symbols('x[0] x[1] x[2] x[3]    x[4] x[5] x[6] x[7]')
#_x = [x0,x1,x2,x3,x4,x5,x6,x7]
#_x
#subsdict = {th1:_x[0],th2:_x[1],th3:_x[2],th4:_x[3] , om1:_x[4],om2:_x[5],om3:_x[6],om4:_x[7]}
#h_x = lambda x : msubs(h , subsdict)

#h_ofx = implemented_function(Function('h_ofx'), lambda q: h)
#h_x = lambdify(q,h_ofx,modules = 'numexpr')

#h_x = lambdify([*x],h)

h_x = lambda x : cse(msubs(h, {th1:x[0],th2:x[1],th3:x[2],th4:x[3],       th1d:x[4],th2d:x[5],th3d:x[6],th4d:x[7]}))

Dmat = lambda x : cse(msubs(D , {th1:x[0],th2:x[1],th3:x[2],th4:x[3],       th1d:x[4],th2d:x[5],th3d:x[6],th4d:x[7]}))
    
Hvec = lambda x : cse(msubs(H ,{th1:x[0],th2:x[1],th3:x[2],th4:x[3],       th1d:x[4],th2d:x[5],th3d:x[6],th4d:x[7]}))

Bmat = lambda x : cse(msubs(B ,{th1:x[0],th2:x[1],th3:x[2],th4:x[3],       th1d:x[4],th2d:x[5],th3d:x[6],th4d:x[7]}))

d_Lfhdx = lambda x : cse(msubs(dLfhdx ,{th1:x[0],th2:x[1],th3:x[2],th4:x[3],       th1d:x[4],th2d:x[5],th3d:x[6],th4d:x[7]}))


# In[123]:

x_= (1,2,3,4,.1,.2,.3,.4)
Hvec(x_)

