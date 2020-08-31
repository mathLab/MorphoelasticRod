'''This code implements a 3D rod model for growing plant shoots. Growth is localized at the organ tip, in a growing region of constant length. 
This code includes lignification (by rod stiffening), plant responses to gravity (sensed by means of statoliths), to bending (proprioception) 
and an endogenous oscillator.'''

from __future__ import print_function
from __future__ import unicode_literals
from fenics import *
import time
import numpy as np
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#######################################   PLOT SETTINGS AND FUNCTIONS   #########################################    
#mpl.rcParams['text.usetex']=True # For displaying in LaTex font
height = 15       # figure height
width  = 15       # figure width
ArrowL = 0.01     # lenght of arrows for directors and statoliths direction
MarkerSize = 0.65 # Marker size for tip projections
TickSize  = 20
LabelSize = 24
TitleSize = 28 
LabelSpace= 28
TickSpace = 12
TipPts = 12000    # Number of displayed points for tip projections
RodLineWidth = 3  # Line width of rod
OldRodColor = '#8d5524'
RodColor    = '#4f8f00'
xr = 0.05         # right axis limit
xl = -xr          # left axis limit
Background = 'w'  # facecolor

def save_plot():  
    # Plot
    fig = plt.figure(figsize=(height,width))
    ax  = fig.add_subplot(111, projection='3d',facecolor=Background)
    plt.title('Shoot length: %.5f cm, Shoot time: %.4f min' % (Lt*100,t*Tau_s/60),fontsize=TitleSize)
    ax.quiver(x[ArrowIndex],y[ArrowIndex],z[ArrowIndex],D1[2],D1[0],D1[1], length=ArrowL, normalize=True, color='b')
    ax.quiver(x[ArrowIndex],y[ArrowIndex],z[ArrowIndex],D2[2],D2[0],D2[1], length=ArrowL, normalize=True)
    ax.quiver(x[ArrowIndex],y[ArrowIndex],z[ArrowIndex],D3[2],D3[0],D3[1], length=ArrowL, normalize=True, color='r')
    ax.quiver(x[ArrowIndex],y[ArrowIndex],z[ArrowIndex],H[2], H[0], H[1],  length=ArrowL, normalize=True, color='k')
    ax.auto_scale_xyz([xl,xr], [xl,xr], [0, (xr-xl)])
    X1, X2 = ax.get_xlim3d()
    Y1, Y2 = ax.get_ylim3d()
    Z1, Z2 = ax.get_zlim3d()
    ax.plot(Xtip[-TipPts:],Ytip[-TipPts:],[Z1]*np.size(Ztip[-TipPts:]),'.',markersize=MarkerSize)
    ax.plot(Xtip[-TipPts:],[Y2]*np.size(Ytip[-TipPts:]),Ztip[-TipPts:],'.',markersize=MarkerSize)
    ax.plot([X1]*np.size(Xtip[-TipPts:]),Ytip[-TipPts:],Ztip[-TipPts:],'.',markersize=MarkerSize)
    ax.plot(x[young],y[young],z[young],color=RodColor,linewidth=RodLineWidth) 
    ax.plot(x[old],y[old],z[old],color=OldRodColor,linewidth=RodLineWidth)
    # Set labels
    ax.set_xlabel("$\mathbf{e}_3$ [m]",fontsize=LabelSize, labelpad=LabelSpace)
    ax.set_ylabel("$\mathbf{e}_1$ [m]",fontsize=LabelSize, labelpad=LabelSpace)
    ax.set_zlabel("$\mathbf{e}_2$ [m]",fontsize=LabelSize, labelpad=LabelSpace)
    # Set ticks
    ax.xaxis.set_tick_params(pad=TickSpace, labelsize=TickSize)
    ax.yaxis.set_tick_params(pad=TickSpace, labelsize=TickSize)
    ax.zaxis.set_tick_params(pad=TickSpace, labelsize=TickSize)
    # Save and close
    fig.savefig('Movie/movie%d.png' % frame_num )
    plt.close(fig)

def save_plot2():  
    fig2 = plt.figure(figsize=(height,width))
    plt.title('TOP VIEW - Shoot length: %.2f cm, Shoot time: %.4f min' % (Lt*100,t*Tau_s/60),fontsize=TitleSize)
    plt.plot(Xtip[-TipPts:],Ytip[-TipPts:],'.')
    plt.xlabel("$\mathbf{e}_3$ [m]",fontsize=LabelSize)
    plt.ylabel("$\mathbf{e}_1$ [m]",fontsize=LabelSize)
    plt.tick_params(axis='both', labelsize=TickSize)
    fig2.savefig('Movie2/Tip%d.png' % frame_num )
    plt.close(fig2)

############################################# POST-PROCESSING FUNCTIONS ##########################################
def reconstruct():
    # Function to reconstruct the shape of the rod, starting from the computed angles 
    global x,y,z,Xtip,Ytip,Ztip,D1,D2,D3,H,old,young
    ds = np.array([j-i for i, j in zip(space_ref[:-1], space_ref[1:])]) # reference space-steps
    # Evaluate fields at the mesh points
    space_cur, stretch_num, psi_num, phi_num, chi_num, thetaH_num, alphaH_num = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    for point in space_ref:
        space_cur   = np.append(space_cur,st(point)) # corresponding mesh in the current configuration
        stretch_num = np.append(stretch_num,stretch(point))
        psi_num     = np.append(psi_num,psi(point))
        phi_num     = np.append(phi_num,phi(point))
        chi_num     = np.append(chi_num,chi(point))
        thetaH_num  = np.append(thetaH_num,thetaH(point))
        alphaH_num  = np.append(alphaH_num,alphaH(point))
    h1_num = np.cos(thetaH_num)
    h2_num = np.sin(thetaH_num)*np.cos(alphaH_num)
    h3_num = np.sin(thetaH_num)*np.sin(alphaH_num)
    # Integrate the d3 director
    D31 = stretch_num*np.sin(psi_num)*np.cos(phi_num)
    D32 = stretch_num*np.sin(psi_num)*np.sin(phi_num)
    D33 = stretch_num*np.cos(psi_num)
    for n in range(1,2*nx+1):
        x[n] = x[n-1]+ds[n-1]*D33[n] # integration along e3
        y[n] = y[n-1]+ds[n-1]*D31[n] # integration along e1  
        z[n] = z[n-1]+ds[n-1]*D32[n] # integration along e2  
    # Update tip coordinates
    Xtip = np.append(Xtip,x[-1])
    Ytip = np.append(Ytip,y[-1])
    Ztip = np.append(Ztip,z[-1])
    # Update directors and statoliths direction, at selected points
    CHI = chi_num[ArrowIndex]
    PSI = psi_num[ArrowIndex]
    PHI = phi_num[ArrowIndex]
    D1 = (np.cos(CHI)*np.cos(PSI)*np.cos(PHI)-np.sin(CHI)*np.sin(PHI),np.cos(CHI)*np.cos(PSI)*np.sin(PHI)+np.sin(CHI)*np.cos(PHI),-np.cos(CHI)*np.sin(PSI))
    D2 = (-np.sin(CHI)*np.cos(PSI)*np.cos(PHI)-np.cos(CHI)*np.sin(PHI),-np.sin(CHI)*np.cos(PSI)*np.sin(PHI)+np.cos(CHI)*np.cos(PHI),np.sin(CHI)*np.sin(PSI))
    D3 = (np.sin(PSI)*np.cos(PHI),np.sin(PSI)*np.sin(PHI),np.cos(PSI))
    H = h1_num[ArrowIndex]*D1+h2_num[ArrowIndex]*D2+h3_num[ArrowIndex]*D3
    # Update old and young
    old = space_cur < (Lt-lg)
    young = [not(x) for x in old]

def save_data():
    # Save solutions
    for i in range(max(Nr,Nrp)):
        U.assign(SolutionArray[i])
        output_file = HDF5File(mesh.mpi_comm(), 'Data/Frame%d/solution%d.h5' % (frame_num,i), "w")
        output_file.write(U, 'Data/Frame%d/solution%d' % (frame_num,i))
        output_file.close()    

################################################# SOLVER FUNCTIONS ################################################    
def Picard(maxiter=250,tol=1.0E-11):
    # maxiter is the max no of iterations allowed
    global U_k, U, a, L, bcs, chi, phi, psi, u1S, u2S, u3S, w1, w2, wp1, wp2, alphaH, thetaH, st, stretch
    eps = 1.0           # error measure ||u-u_k||
    Nit = 0             # iteration counter
    while eps > tol and Nit < maxiter:
        Nit += 1
        solve(a == L, U, bcs)
        chi, phi, psi, u1S, u2S, u3S, w1, w2, wp1, wp2, alphaH, thetaH, st, stretch = U.split()
        diff = U.vector().get_local() - U_k.vector().get_local()
        eps = np.linalg.norm(diff, ord=np.Inf)
        print('iter=%d: norm=%g' % (Nit, eps))
        U_k.assign(U)   # update for next iteration    

def Newton(maxiter=50,tol=1E-14):
    global U, F, bcs, J
    problem = NonlinearVariationalProblem(F, U, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = tol
    prm["newton_solver"]["relative_tolerance"] = tol
    prm["newton_solver"]["maximum_iterations"] = maxiter
    solver.solve()     

###########################################    GLOBAL CONSTANTS  ############################################## 
g     = 9.81  # m/s^2 - gravity acceleration

###########################################    MODEL PARAMETERS  ##############################################    
# Time-scales
Tau_a  =  2*60        # s - time-scale for statoliths avalanche dynamics
Tau_s  = 12*60        # s - time-scale for nondimensional equations
Tau_m  = 12*60        # s - Memory time for gravitropism
Tau_mp = 12*60        # s - Memory time for proprioception
Tau_r  = 12*60        # s - Reaction time for gravitropism
Tau_rp = 12*60        # s - Reaction time for proprioception
Tau_g  = 40*60*60     # s - Growth time for the upper part
Tau_e  = 24*60        # s - Time period for endogenous oscillator
Tau_l  = 6*24*60*60   # s - Maturation/Lignification time

# Geometry
L0 = 7.2E-2     # m - initial length
lg = 7.2E-2       # m - growing zone
R  = 5E-4       # m - radius
A  = np.pi*R**2 # m^2 - cross sectional area

if L0 < lg:
    print('Attention! L0 must be >= lg')

# BioMechanical properties of plant tissue
Lig  = 200          # rod stiffening ratio
EY   = 10E+6        # N/m^2 - Young Modulus
I    = np.pi*R**4/4 # m^4 - Second moment of inertia
B0   = EY*I         # Bending stiffness
Bmax = Lig*EY*I
nu   = 0.5          # Poisson's ratio        
mu   = 2*EY*(1+nu)  # shear modulus
J    = 2*I          # parameter depending on the cross-sectional shape
Jmax = Lig*J
rho  = 1000         # Kg/m^3 - density
alpha= 0.02          # internal oscillator sensitivity
beta = 0.8          # gravitropic sensitivity
eta  = 20           # proprioceptic sensitivity

# Loads
Q1   = 0            # N/m - distributed load along e1 per unit length
Q2   = -rho*g*A     # N/m - distributed load along e2 per unit length
Q3   = 0            # N/m - distributed load along e3 per unit length
al1  = 0            # N   - apical load along e1
al2  = 0            # N   - apical load along e2   
al3  = 0            # N   - apical load along e3   

###########################################  FEM IMPLEMENTATION  ##############################################    
# Create mesh
nx = 2**9
mesh     = IntervalMesh(2*nx,0,L0)
space_ref= mesh.coordinates()

# Time-step
Nr  = 2**6                   # number of time-steps for gravitropic delay
dt  = (Tau_r/Tau_s)/Nr       # time-step
Nrp = int((Tau_rp/Tau_s)/dt) # number of time-steps for proprioceptic delay

# Function space
P1      = FiniteElement('P',interval, 1)
element = MixedElement([P1, P1, P1, P1, P1, P1, P1, P1, P1, P1, P1, P1, P1, P1])
V       = FunctionSpace(mesh, element)

#P       = FunctionSpace(mesh, 'DG', 0)
# Some functions
#def Max(a, b): return (a+b+abs(a-b))/Constant(2)
#def Min(a, b): return (a+b-abs(a-b))/Constant(2) 
#def Sign(a): return a/abs(a) 

# Some constants
Zero = Constant(0)
EE   = Constant(np.e)

# Boundary conditions
def boundary_D_l(x, on_boundary):
    return on_boundary and near(x[0],0,1E-14)
bc_chi = DirichletBC(V.sub(0), Zero, boundary_D_l)
bc_phi = DirichletBC(V.sub(1), DOLFIN_PI/2, boundary_D_l)
bc_psi = DirichletBC(V.sub(2), DOLFIN_PI/2, boundary_D_l)
bc_st  = DirichletBC(V.sub(12),Zero, boundary_D_l)
bcs = [bc_chi,bc_phi,bc_psi,bc_st]

# Initial conditions for statolith direction
alphaH0 = DOLFIN_PI/2
thetaH0 = DOLFIN_PI/2

# Define expressions used in variational forms
h      = Expression('h', h=dt, degree=1)
alpha  = Constant(alpha)
beta   = Constant(beta)
eta    = Constant(eta)
q1     = Constant(Q1)
q2     = Constant(Q2)
q3     = Constant(Q3)
p1     = Constant(al1)
p2     = Constant(al2)
p3     = Constant(al3)
B0     = Constant(B0)
Bmax   = Constant(Bmax)
J0     = Constant(mu*J)
Jmax   = Constant(mu*Jmax)
r      = Constant(R)
Tg     = Constant(Tau_g)
Tm     = Constant(Tau_m)
Tmp    = Constant(Tau_mp)
Ts     = Constant(Tau_s)
Ta     = Constant(Tau_a)
Tl     = Constant(Tau_l)
v1     = Expression('cos(omega*t)', omega=2*np.pi*Tau_s/Tau_e, t=0, degree=1) # internal oscillator component (along d1)
v2     = Expression('sin(omega*t)', omega=2*np.pi*Tau_s/Tau_e, t=0, degree=1) # internal oscillator component (along d2)
l0     = Constant(L0)
Lg     = Constant(lg)
length = Expression('L0+lg*k*t*ts', L0=L0, lg=lg, ts=Tau_s, k=1/Tau_g, t=0., degree=1)

# Define variational problem in the reference domain [0,L0]
test_chi, test_phi, test_psi, test_u1S, test_u2S, test_u3S, test_w1, test_w2, test_wp1, test_wp2, test_alphaH, test_thetaH, test_st, test_stretch = TestFunctions(V)
# Retarded unknowns
U_r  = Function(V)
chi_r, phi_r, psi_r, u1S_r, u2S_r, u3S_r, w1_r, w2_r, wp1_r, wp2_r, alphaH_r, thetaH_r, st_r, stretch_r = split(U_r) # retarded functions for gravitropic delay
U_rp  = Function(V)
chi_rp, phi_rp, psi_rp, u1S_rp, u2S_rp, u3S_rp, w1_rp, w2_rp, wp1_rp, wp2_rp, alphaH_rp, thetaH_rp, st_rp, stretch_rp = split(U_rp) # retarded functions for gravitropic delay
# Previous step unknowns
U_n = Function(V)
chi_n, phi_n, psi_n, u1S_n, u2S_n, u3S_n, w1_n, w2_n, wp1_n, wp2_n, alphaH_n, thetaH_n, st_n, stretch_n = split(U_n)
# Unknowns
U = Function(V)
chi, phi, psi, u1S, u2S, u3S, w1, w2, wp1, wp2, alphaH, thetaH, st, stretch = split(U)
# Functional defined by using backward Euler in time # conditional(condition, true_value, false_value) 
B    = conditional(ge(st-length+Lg,DOLFIN_EPS), B0, Bmax-(Bmax-B0)*EE**((Lg+st-length)*Tg/(Lg*Tl))) # linear growth in time with slope Tg/Lg
muJ  = conditional(ge(st-length+Lg,DOLFIN_EPS), J0, Jmax-(Jmax-J0)*EE**((Lg+st-length)*Tg/(Lg*Tl)))
#REGR = conditional(ge(st-length+Lg,DOLFIN_EPS), 1/Tg, Zero) # Example 1: Constant growth
#REGR = conditional(ge(st-length+Lg,DOLFIN_EPS), conditional(ge(st-length+Lg1,DOLFIN_EPS),1/Tg1,1/Tg), Zero) # Example 2: Piecewise constant growth
REGR = conditional(ge(st-length+Lg,DOLFIN_EPS), (st-length+Lg)*2/(Tg*Lg), Zero) # Example 3: Linear growth
#REGR = conditional(ge(st-length+Lg,DOLFIN_EPS), (1+cos(2*DOLFIN_PI*(st-length+Lg/2)/Lg))/Tg, Zero) # Example 4: Sinusoidal growth
#REGR = conditional(ge(st-length+Lg,DOLFIN_EPS), (st-length+Lg)*(1+cos(2*DOLFIN_PI*(st-length+Lg/2)/Lg))*2/(Lg*Tg), Zero) # Example 5
n1 = p1+q1*(length-st) # contact force - e1 component
n2 = p2+q2*(length-st) # contact force - e2 component
n3 = p3+q3*(length-st) # contact force - e3 component 
u1 = (psi.dx(0)*sin(chi)-phi.dx(0)*cos(chi)*sin(psi))/stretch # flexural strain 1
U1 = u1-u1S
u2 = (psi.dx(0)*cos(chi)+phi.dx(0)*sin(chi)*sin(psi))/stretch # flexural strain 2
U2 = u2-u2S
u3 = (chi.dx(0)+phi.dx(0)*cos(psi))/stretch # torsional strain
U3 = u3-u3S
m1 = B*(cos(psi)*cos(phi)*(U1*cos(chi)-U2*sin(chi))-sin(phi)*(U1*sin(chi)+U2*cos(chi))) \
     + muJ*U3*sin(psi)*cos(phi) # resultant contact couple - e1 component
m2 = B*(cos(psi)*sin(phi)*(U1*cos(chi)-U2*sin(chi))+cos(phi)*(U1*sin(chi)+U2*cos(chi))) \
     + muJ*U3*sin(psi)*sin(phi) # resultant contact couple - e2 component
m3 = -B*sin(psi)*(U1*cos(chi)-U2*sin(chi)) \
     + muJ*U3*cos(psi)          # resultant contact couple - e3 component
u1_rp = (psi_rp.dx(0)*sin(chi_rp)-phi_rp.dx(0)*cos(chi_rp)*sin(psi_rp))/stretch_rp
u2_rp = (psi_rp.dx(0)*cos(chi_rp)+phi_rp.dx(0)*sin(chi_rp)*sin(psi_rp))/stretch_rp
H1_r = cos(thetaH_r)
H2_r = sin(thetaH_r)*cos(alphaH_r)
RHS_alpha = cos(alphaH)*sin(psi)*sin(phi)+(cos(psi)*sin(chi)*sin(phi)-cos(chi)*cos(phi))*sin(alphaH)
RHS_theta = cos(thetaH)*( cos(alphaH)*(cos(chi)*cos(phi)-cos(psi)*sin(chi)*sin(phi))+sin(alphaH)*sin(psi)*sin(phi) )-sin(thetaH)*( cos(phi)*sin(chi)+cos(chi)*cos(psi)*sin(phi) )    
F =   (st.dx(0)-stretch)*test_st*dx \
    + ((stretch-stretch_n)-h*Ts*stretch*REGR)*test_stretch*dx \
    - m1*test_chi.dx(0)*dx + stretch*(n3*sin(psi)*sin(phi)-n2*cos(psi))*test_chi*dx \
    - m2*test_psi.dx(0)*dx + stretch*(n1*cos(psi)-n3*sin(psi)*cos(phi))*test_psi*dx \
    - m3*test_phi.dx(0)*dx + stretch*(n2*sin(psi)*cos(phi)-n1*sin(psi)*sin(phi))*test_phi*dx \
    + ((u1S-u1S_n)*r-h*(Ts*REGR)*(alpha*v1+beta*w1+eta*r*wp1))*test_u1S*dx \
    + ((u2S-u2S_n)*r-h*(Ts*REGR)*(alpha*v2+beta*w2+eta*r*wp2))*test_u2S*dx \
    + u3S*test_u3S*dx \
    + ((w1-w1_n)*Tm/Ts+h*(w1+H2_r))*test_w1*dx \
    + ((w2-w2_n)*Tm/Ts+h*(w2-H1_r))*test_w2*dx \
    + ((wp1-wp1_n)*Tmp/Ts+h*(wp1+u1_rp))*test_wp1*dx \
    + ((wp2-wp2_n)*Tmp/Ts+h*(wp2+u2_rp))*test_wp2*dx \
    + ((alphaH-alphaH_n)*sin(thetaH)-RHS_alpha*h*Ts/Ta)*test_alphaH*dx \
    + ((thetaH-thetaH_n)-RHS_theta*h*Ts/Ta)*test_thetaH*dx 
du = TrialFunction(V)
J = derivative(F,U,du)

# LINEAR PROBLEM FOR PICARD
chi_t, phi_t, psi_t, u1S_t, u2S_t, u3S_t, w1_t, w2_t, wp1_t, wp2_t, alphaH_t, thetaH_t, st_t, stretch_t  = TrialFunctions(V)
U_k = Function(V)
chi_k, phi_k, psi_k, u1S_k, u2S_k, u3S_k, w1_k, w2_k, wp1_k, wp2_k, alphaH_k, thetaH_k, st_k, stretch_k  = split(U_k)
u1_k = psi_t.dx(0)*sin(chi_k)-phi_t.dx(0)*cos(chi_k)*sin(psi_k)/stretch_k
U1_k = u1_k-u1S_t
u2_k = psi_t.dx(0)*cos(chi_k)+phi_t.dx(0)*sin(chi_k)*sin(psi_k)/stretch_k
U2_k = u2_k-u2S_t
u3_k = chi_t.dx(0)+phi_t.dx(0)*cos(psi_k)/stretch_k
U3_k = u3_k-u3S_t
B_k    = conditional(ge(st_k-length+Lg,DOLFIN_EPS), B0, Bmax-(Bmax-B0)*EE**((Lg+st_k-length)*Tg/(Lg*Tl))) # linear growth in time with slope Tg/Lg
muJ_k  = conditional(ge(st_k-length+Lg,DOLFIN_EPS), J0, Jmax-(Jmax-J0)*EE**((Lg+st_k-length)*Tg/(Lg*Tl)))
#REGR_k = conditional(ge(st_k-length+Lg,DOLFIN_EPS), 1/Tg, Zero) # Example 1: Constant growth
#REGR_k = conditional(ge(st_k-length+Lg,DOLFIN_EPS), conditional(ge(st_k-length+Lg1,DOLFIN_EPS),1/Tg1,1/Tg), Zero) # Example 2: Piecewise constant growth
REGR_k = conditional(ge(st_k-length+Lg,DOLFIN_EPS), (st_k-length+Lg)*2/(Tg*Lg), Zero) # Example 3: Linear growth
#REGR_k = conditional(ge(st_k-length+Lg,DOLFIN_EPS), (1+cos(2*DOLFIN_PI*(st_k-length+Lg/2)/Lg))/Tg, Zero) # Example 4
#REGR_k = conditional(ge(st_k-length+Lg,DOLFIN_EPS), (st_k-length+Lg)*(1+cos(2*DOLFIN_PI*(st_k-length+Lg/2)/Lg))*2/(Lg*Tg), Zero) # Example 5
n1_k = p1+q1*(length-st_k)
n2_k = p2+q2*(length-st_k)
n3_k = p3+q3*(length-st_k)
m1_k = B_k*(cos(psi_k)*cos(phi_k)*(U1_k*cos(chi_k)-U2_k*sin(chi_k)) -sin(phi_k)*(U1_k*sin(chi_k)+U2_k*cos(chi_k))) \
       + muJ_k*U3_k*sin(psi_k)*cos(phi_k)
m2_k = B_k*(cos(psi_k)*sin(phi_k)*(U1_k*cos(chi_k)-U2_k*sin(chi_k))+cos(phi_k)*(U1_k*sin(chi_k)+U2_k*cos(chi_k))) \
       + muJ_k*U3_k*sin(psi_k)*sin(phi_k)
m3_k = -B_k*sin(psi_k)*(U1_k*cos(chi_k)-U2_k*sin(chi_k)) \
       + muJ_k*U3_k*cos(psi_k)
RHS_alpha_k = cos(alphaH_k)*sin(psi_k)*sin(phi_k)+(cos(psi_k)*sin(chi_k)*sin(phi_k)-cos(chi_k)*cos(phi_k))*sin(alphaH_k)
RHS_theta_k = cos(thetaH_k)*( cos(alphaH_k)*(cos(chi_k)*cos(phi_k)-cos(psi_k)*sin(chi_k)*sin(phi_k))+sin(alphaH_k)*sin(psi_k)*sin(phi_k) ) -sin(thetaH_k)*( cos(phi_k)*sin(chi_k)+cos(chi_k)*cos(psi_k)*sin(phi_k) ) 
a =   (st_t.dx(0)-stretch_t)*test_st*dx \
    + stretch_t*(1-h*Ts*REGR_k)*test_stretch*dx \
    - m1_k*test_chi.dx(0)*dx \
    - m2_k*test_psi.dx(0)*dx \
    - m3_k*test_phi.dx(0)*dx \
    + (u1S_t*r-h*(Ts*REGR_k)*(beta*w1_t+eta*r*wp1_t))*test_u1S*dx \
    + (u2S_t*r-h*(Ts*REGR_k)*(beta*w2_t+eta*r*wp2_t))*test_u2S*dx \
    + u3S_t*test_u3S*dx \
    + (Tm/Ts+h)*w1_t*test_w1*dx \
    + (Tm/Ts+h)*w2_t*test_w2*dx \
    + (Tmp/Ts+h)*wp1_t*test_wp1*dx \
    + (Tmp/Ts+h)*wp2_t*test_wp2*dx \
    + alphaH_t*sin(thetaH_k)*test_alphaH*dx \
    + thetaH_t*test_thetaH*dx
L =   stretch_n*test_stretch*dx \
    - stretch_k*(n3_k*sin(psi_k)*sin(phi_k)-n2_k*cos(psi_k))*test_chi*dx \
    - stretch_k*(n1_k*cos(psi_k)-n3_k*sin(psi_k)*cos(phi_k))*test_psi*dx \
    - stretch_k*(n2_k*sin(psi_k)*cos(phi_k)-n1_k*sin(psi_k)*sin(phi_k))*test_phi*dx \
    + (r*u1S_n+h*REGR_k*Ts*alpha*v1)*test_u1S*dx \
    + (r*u2S_n+h*REGR_k*Ts*alpha*v2)*test_u2S*dx \
    + (w1_n*Tm/Ts-h*H2_r)*test_w1*dx \
    + (w2_n*Tm/Ts+h*H1_r)*test_w2*dx \
    + (wp1_n*Tmp/Ts-h*u1_rp)*test_wp1*dx \
    + (wp2_n*Tmp/Ts-h*u2_rp)*test_wp2*dx \
    + (alphaH_n*sin(thetaH_k)+RHS_alpha_k*h*Ts/Ta)*test_alphaH*dx \
    + (thetaH_n+RHS_theta_k*h*Ts/Ta)*test_thetaH*dx

################################################## INITIALIZE STORAGE ##################################################
t  = 0
Lt = L0
ArrowIndex = np.array([int(i) for i in np.linspace(0,2*nx,5)]) # material points where to plot directors

# Initialize rod coordinates
x = np.zeros(2*nx+1)
y = np.zeros(2*nx+1)
z = np.zeros(2*nx+1)

# Initialize tip Coordinates
Xtip, Ytip, Ztip = np.empty(0), np.empty(0), np.empty(0)
    
################################################## INITIAL ELASTIC EQUILIBRIUM ###########################################
# SOLVE INITIAL GUESS: STRAIGHT CONFIGURATION 
F0 =    (st.dx(0)-Constant(1))*test_st*dx \
      + (stretch-Constant(1))*test_stretch*dx \
      + chi.dx(0)*test_chi*dx \
      + psi.dx(0)*test_psi*dx \
      + phi.dx(0)*test_phi*dx \
      + u1S*test_u1S*dx \
      + u2S*test_u2S*dx \
      + u3S*test_u3S*dx \
      + w1*test_w1*dx \
      + w2*test_w2*dx \
      + wp1*test_wp1*dx \
      + wp2*test_wp2*dx \
      + (alphaH-alphaH0)*test_alphaH*dx \
      + (thetaH-thetaH0)*test_thetaH*dx
J0 = derivative(F0,U,du)
problem0 = NonlinearVariationalProblem(F0, U, bcs, J0)
solver0 = NonlinearVariationalSolver(problem0)
solver0.solve()    

# SOLVE INITIAL ELASTIC EQUILIBRIUM
F_n =   (st.dx(0)-Constant(1))*test_st*dx \
      + (stretch-Constant(1))*test_stretch*dx \
      - m1*test_chi.dx(0)*dx + stretch*(n3*sin(psi)*sin(phi)-n2*cos(psi))*test_chi*dx \
      - m2*test_psi.dx(0)*dx + stretch*(n1*cos(psi)-n3*sin(psi)*cos(phi))*test_psi*dx \
      - m3*test_phi.dx(0)*dx + stretch*(n2*sin(psi)*cos(phi)-n1*sin(psi)*sin(phi))*test_phi*dx \
      + u1S*test_u1S*dx \
      + u2S*test_u2S*dx \
      + u3S*test_u3S*dx \
      + w1*test_w1*dx \
      + w2*test_w2*dx \
      + wp1*test_wp1*dx \
      + wp2*test_wp2*dx \
      + (alphaH-alphaH0)*test_alphaH*dx \
      + (thetaH-thetaH0)*test_thetaH*dx
J_n = derivative(F_n,U,du)
problem_n = NonlinearVariationalProblem(F_n, U, bcs, J_n)
solver_n  = NonlinearVariationalSolver(problem_n)
solver_n.solve()  
    
# Initialization array of solutions at previous steps
SolutionArray = []
for n in range(max(Nr,Nrp)):
    SolutionArray.append(U.copy(deepcopy=True)) 

chi, phi, psi, u1S, u2S, u3S, w1, w2, wp1, wp2, alphaH, thetaH, st, stretch = U.split()

# Curve reconstruction from the angles
frame_num = 0 # Initialize frame counter
reconstruct()

# Plot and Save
proc = mp.Process(target=save_plot)
proc.daemon = True
proc.start()
proc.join()

proc2 = mp.Process(target=save_plot2)
proc2.daemon = True
proc2.start()
proc2.join()

#################################################### SOLVE ########################################################  
while t<10000:
    # Update frame counter
    frame_num += 1 
    # Update current time and all functions depending on time
    t += dt
    length.t = t
    v1.t     = t # internal oscillator component along d1
    v2.t     = t # internal oscillator component along d2
    # Update variables
    U_n.assign(SolutionArray[-1])    # update the previous step
    U_r.assign(SolutionArray[-Nr])   # update the retarded solution for gravitropism
    U_rp.assign(SolutionArray[-Nrp]) # update the retarded solution for proprioception
    
    start = time.time()
    try:
        # Solve the problem through nonlinear FEniCS solver
        Newton()
    except:
        # Use Picard's method 
        print('Newton did not converge: trying Picard first...')
        U.assign(SolutionArray[-1])
        U_k.assign(SolutionArray[-1])
        Picard()
        # Solve the problem through nonlinear FEniCS solver
        Newton(50,1E-10) 
    end = time.time()  

    # Update
    chi, phi, psi, u1S, u2S, u3S, w1, w2, wp1, wp2, alphaH, thetaH, st, stretch = U.split() # update variables
    SolutionArray = SolutionArray[1:]             # remove the first element
    SolutionArray.append(U.copy(deepcopy=True))   # add the last solution

    Lt = st(L0) # current length
    print('Frame: %d, Shoot Time: %g min, Shoot Length: %.10f cm, Computation Time: %g sec' % (frame_num, t*Tau_s/60, Lt*100, end-start))      
    
    ##### PLOT AND SAVE ####
    # Curve reconstruction
    reconstruct()

    # Plot and Save
    proc = mp.Process(target=save_plot)
    proc.daemon = True
    proc.start()
    proc.join()

    #proc2 = mp.Process(target=save_plot2)
    #proc2.daemon = True
    #proc2.start()
    #proc2.join()
    
    if np.mod(frame_num,max(Nr,Nrp))==0:
        save_data()
        print('Saving data...')
