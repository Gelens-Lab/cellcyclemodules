#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A modular approach for modeling the cell cycle based on functional response curves
"""
import numpy as np
import matplotlib.pyplot as plt
import Delay_Bistability_Analysis_Final as DBA
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from scipy.integrate import solve_ivp

# Set plot parameters
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 8       # Annotation fonts
rcParams['axes.labelsize'] = 9
rcParams['axes.titlesize'] = 9
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
plt.ion()

# COLOR SCHEME
# Define some colors to use in plots
COL_v = ['#471164ff','#472f7dff','#3e4c8aff','#32648eff','#297a8eff','#21918cff','#21a685ff','#3bbb75ff','#69cd5bff','#a5db36ff','#e2e418ff']
COL_apc = COL_v[9]
COL_cdk = COL_v[5]


# =============================================================================
# CODE TO GENERATE FIGURE 3 IN THE MAIN TEXT
# Block diagrams were added afterwards in inkscape
# =============================================================================
# Define some fixed parameter values
eps = 100   # here, eps is defned as the inverse from the text
K = 20
bdeg = 0.1
n = 15
r = 0.5

fig = plt.figure(figsize=(8, 2.7*2.5),constrained_layout=True)
G = gridspec.GridSpec(3, 3, figure = fig)

fig_ax1 = fig.add_subplot(G[0, 0])
fig_ax1.axis('off')
fig_ax1.annotate('A',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax2 = fig.add_subplot(G[0, 1])
fig_ax2.set_xlabel('[Cdk1] (nM)')
fig_ax2.set_ylabel(r'$[APC]^*$')
fig_ax2.annotate('B',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax3 = fig.add_subplot(G[0, 2])
fig_ax3_right = fig_ax3.twinx()
fig_ax3.set_xlabel('Time (min))')
fig_ax3.annotate('C',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax4 = fig.add_subplot(G[1, 0])
fig_ax4.axis('off')
fig_ax4.annotate('D',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax5 = fig.add_subplot(G[1, 1])
fig_ax5.set_xlabel('[Cdk1] (nM)')
fig_ax5.set_ylabel(r'$[APC]^*$')
fig_ax5.annotate('E',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax6 = fig.add_subplot(G[1, 2])
fig_ax6_right = fig_ax6.twinx()
fig_ax6.set_xlabel('Time (min)')
fig_ax6.annotate('F',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax7 = fig.add_subplot(G[2, 0])
fig_ax7.set_xlabel('Relative synthesis c')
fig_ax7.set_ylabel('Bistable width w (nM)')
fig_ax7.annotate('G',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax8 = fig.add_subplot(G[2, 1])
fig_ax8.set_xlabel('[Cdk1] (nM)')
fig_ax8.set_ylabel(r'$[APC]^*$')
fig_ax8.annotate('H',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax9 = fig.add_subplot(G[2, 2])
fig_ax9_right = fig_ax9.twinx()
fig_ax9.set_xlabel('Time (min)')
fig_ax9.annotate('I',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')


# PLOTS RELATED TO SYSTEM Ia (ULTRASENSITIVE SYSTEM)
# =============================================================================
### Plot the nullclines in the phase plane
# Cdk1 nullcline
z = 4
for c in [0.3,0.5,0.7]:
    Cdk_v = np.linspace(c,2.2,100)
    APC_v = c/Cdk_v
    # Convert non-dimensionalized values to original dimensions for plotting
    fig_ax2.plot(Cdk_v*K,APC_v,c=COL_v[z],label='c = '+str(c))    
    z = z + 3

    ### Determine steady state values of the overall system
    APC_v = np.linspace(1e-12,1-1e-12,10000000)
    Cdk1 = c/APC_v
    # Nullclines intersect where F = 0
    F = eps*(Cdk1**n/(Cdk1**n + 1) - APC_v)
    APC_roots = APC_v[1:][F[1:]*F[0:-1] < 0]    # Determine APC levels for which F = 0 (i.e. where F changes sign; i.e. if product of subsequent elements < 0)
    Cdk1_roots = c/APC_roots                    # Calculate Cdk1 levels where F = 0
    for X,Y in list(zip(Cdk1_roots,APC_roots)): # Determine stability of the steady states via time simulation
        y0 = [X+1e-6,Y+1e-6]                    # Use small perturbation from steady state as initial condition
        t_start = 0
        t_end = 50
        def f(t,y): return DBA.bist_2d_cubic(y,t,c,eps,r,n,0)   # alpha = 0 for ultrasensitive system  
        sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=0.01,rtol=1e-6,atol=1e-9)    # Solve the system
        if (abs(sol.y[0,:][-1]-X) < 1e-6) and (abs(sol.y[1,:][-1]-Y) < 1e-6):   # if time evolution approaches steady state: stable
            fc='k'  # Denote stable steady state with filled dot
        else:       # else: unstable
            fc='w'  # Denote unstable steady state with open dot     
        # Convert non-dimensionalized values to original dimensions for plotting
        fig_ax2.scatter(X*K,Y,s=10,zorder=5,edgecolor='k',facecolor=fc)

fig_ax2.set_ylim([0,1.05])
fig_ax2.set_xlim([0,2.2*K])

xtick_v = sorted([0,K,2*K])
xtick_label_v = [0,r'$K_{cdk,apc}$',r'$2K_{cdk,apc}$']
fig_ax2.set_xticks(xtick_v)
fig_ax2.set_xticklabels(xtick_label_v)

fig_ax2.legend(bbox_to_anchor=(0.55, 0.55, 1, 0),loc='lower left',frameon=False) 
fig_ax2.annotate('n = '+str(n),(5,0.1))

# Apc nullcline
Cdk_v = np.linspace(0,50,500)
APC_v = Cdk_v**n/(K**n + Cdk_v**n)
fig_ax2.plot(Cdk_v,APC_v,linestyle='-',c='k',alpha=0.5)

### Plot time trajectories in time domain for different initial conditions
c = 0.5
for y0,ls in [[[0,0],'-'],[[50/K,1],'--']]:
    t_start = 0
    t_end = 10
    
    def f(t,y): return DBA.bist_2d_cubic(y,t,c,eps,r,n,0)    # alpha = 0 for ultrasensitive system 
    sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9)
    
    t_v = sol.t
    Cdk1_sol = sol.y[0,:]
    Apc_sol = sol.y[1,:]
    
    # Convert non-dimensionalized values to original dimensions for plotting
    l_apc, = fig_ax3_right.plot(t_v/bdeg,Apc_sol,linestyle=ls,color=COL_apc,alpha=1)
    l_cdk1, = fig_ax3.plot(t_v/bdeg,Cdk1_sol*K,linestyle=ls,color=COL_cdk,alpha=1)
    
fig_ax3.set_ylabel('[Cdk1] (nM)',color=l_cdk1.get_color())
fig_ax3_right.set_ylabel(r'$[APC]^*$',color=l_apc.get_color())
fig_ax3.set_xlim([0,80])
fig_ax3.set_ylim([0,50])
fig_ax3_right.set_ylim([0,1.05])
fig_ax3.annotate('c = '+str(c)+'\nn = '+str(n)+'\nr = '+str(r)+'\n'+r'$K_{cdk,apc} = $'+str(K)+' nM',(40,3))


# PLOTS RELATED TO SYSTEM Ib (BISTABLE SYSTEM)
# =============================================================================
### Plot the number and location of steady states for system i as a function of c and bistable width
# Load csv data from screen
CSV = DBA.find_csv_bistability(n,r,'Cdk1','Cubic')
Array,Period_Grid,Amplitude_Grid,SS_Grid,SS_Pos_Grid,A_v,C_v,W_v = DBA.csv_to_array_bistability(CSV[1],CSV[0],'Cubic',r,n)
# Convert the position of the steady state in SS_Pos_Grid to numerical value; bistable system has value 0 in original data 
SS_Pos_Grid[SS_Pos_Grid=='Bottom'] = 1
SS_Pos_Grid[SS_Pos_Grid=='Middle'] = 2
SS_Pos_Grid[SS_Pos_Grid=='Top'] = 3

# For plotting the data, multiply the bistable width by Kcdkapc to convert from the non-dimensionalized system to original dimensions
X,Y = np.meshgrid(np.array(C_v),np.array(W_v)*K)
cs = fig_ax7.contourf(X,Y,SS_Pos_Grid,levels=[-0.5,0.5,1.5,2.5,3.5])

fig_ax7.annotate('Bi',(0.28,23),c='white',fontweight='bold')
fig_ax7.annotate('Osc',(0.35,8),c='white',fontweight='bold')
fig_ax7.annotate('Mono \nTop',(0.75,16),c='white',fontweight='bold')
fig_ax7.annotate('Mono \nLow',(0.08,16),c='white',fontweight='bold')

fig_ax7.annotate(r'$K_{cdk,apc}$ = '+str(int(K))+' nM\nn = '+str(int(n))+'\nr = '+str(r),(0.6,9),c='k')
fig_ax7.set_xlim([0,1.2])
fig_ax7.set_ylim([0,25])

for i in cs.collections:
    i.set_edgecolor('face')


### Plot the nullclines in the phase plane
# Select parameter values based on number and location of steady states (i.e. from fig_ax7)
xmin = (1 + r + np.sqrt((1 + r)**2 - 3*r))/3 
a_max = -1/(xmin*(xmin - r)*(xmin - 1))
a = 0.63*a_max
c = 0.335
w = DBA.width_cubic(r,n,a)[-1]*K

fig_ax7.scatter(c,w,c='k',s=5)
fig_ax7.annotate('E-F',(c+0.05,w),c='k',fontweight='bold')

# Cdk1 nullcline
APC_v = np.linspace(1e-12,1,1000)
CDK_v = c/APC_v
fig_ax5.plot(CDK_v*K,APC_v,color=COL_cdk,linestyle='-',label='[Cdk1]')
# APC nullcline
APC_v = np.linspace(0,1-1e-12,1000)
Xi_v = 1 + a*APC_v*(APC_v - 1)*(APC_v - r) 
CDK_v = Xi_v*(APC_v/(1 - APC_v))**(1/n)
fig_ax5.plot(CDK_v*K,APC_v,color=COL_apc,linestyle='-',label=r'$[APC]^*$')
fig_ax5.set_xlim([0,50])
fig_ax5.set_ylim([0,1.05])
fig_ax5.legend(bbox_to_anchor=(0.5, 0.5, 1, 0),loc='lower left',title='Nullclines',frameon=False) 

### Determine steady state values of the overall system
APC_v = np.linspace(1e-12,1-1e-12,10000000)
Xi = 1 + a*APC_v*(APC_v - 1)*(APC_v - r)     
Cdk1 = c/APC_v 
F = eps*(Cdk1**n/(Cdk1**n + Xi**n) - APC_v) # Nullclines intersect where F = 0
APC_roots = APC_v[1:][F[1:]*F[0:-1] < 0]    # Determine APC levels for which F = 0 (i.e. where F changes sign; i.e. if product of subsequent elements < 0)
Cdk1_roots = c/APC_roots                    # Calculate Cdk1 levels where F = 0
for X,Y in list(zip(Cdk1_roots,APC_roots)): # Determine stability of the steady states via time simulation
    y0 = [X+1e-6,Y+1e-6]                    # Use small perturbation from steady state as initial condition
    t_start = 0
    t_end = 50
    def f(t,y): return DBA.bist_2d_cubic(y,t,c,eps,r,n,a)   
    sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=0.01,rtol=1e-6,atol=1e-9)    # Solve the system
    if (abs(sol.y[0,:][-1]-X) < 1e-6) and (abs(sol.y[1,:][-1]-Y) < 1e-6):   # if time evolution approaches steady state: stable
        fc='k'  # Denote stable steady state with filled dot
    else:       # else: unstable
        fc='w'  # Denote unstable steady state with open dot     
    # Convert non-dimensionalized values to original dimensions for plotting
    fig_ax5.scatter(X*K,Y,s=10,zorder=5,edgecolor='k',facecolor=fc)


### Plot time trajectories in time domain for different initial conditions
for y0,ls in [[[0,0],'-'],[[50/K,1],'--']]:
    t_start = 0
    t_end = 20
    def f(t,y): return DBA.bist_2d_cubic(y,t,c,eps,r,n,a)    
    sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9)
    
    t_v = sol.t
    Cdk1_sol = sol.y[0,:]
    Apc_sol = sol.y[1,:]

    # Convert non-dimensionalized values to original dimensions for plotting    
    l_cdk1, = fig_ax6.plot(t_v/bdeg,Cdk1_sol*K,linestyle=ls,color=COL_cdk)
    l_apc, = fig_ax6_right.plot(t_v/bdeg,Apc_sol,linestyle=ls,color= COL_apc)

fig_ax6.set_ylabel('[Cdk1] (nM)',color=l_cdk1.get_color())
fig_ax6_right.set_ylabel(r'$[APC]^*$',color=l_apc.get_color())
fig_ax6.set_xlim([0,80])
fig_ax6.set_ylim([0,50])
fig_ax6_right.set_ylim([0,1.05])
    
    
    

# PLOTS RELATED TO SYSTEM Ib (OSCILLATORY SYSTEM)
# =============================================================================
### Plot the nullclines in the phase plane
# Select parameter values based on number and location of steady states (fig_ax7)
xmin = (1 + r + np.sqrt((1 + r)**2 - 3*r))/3 
a_max = -1/(xmin*(xmin - r)*(xmin - 1))
a = 0.2*a_max
c = 0.55
w = DBA.width_cubic(r,n,a)[-1]*K

fig_ax7.scatter(c,w,c='k',s=5)
fig_ax7.annotate('H-I',(c-0.15,w-1),c='k',fontweight='bold')

# Cdk1 nullcline
APC_v = np.linspace(1e-12,1,1000)
CDK_v = c/APC_v
fig_ax8.plot(CDK_v*K,APC_v,color=COL_cdk,linestyle='-',label='[Cdk1]')
# Apc nullcline
APC_v = np.linspace(0,1-1e-12,1000)
Xi_v = 1 + a*APC_v*(APC_v - 1)*(APC_v - r) 
CDK_v = Xi_v*(APC_v/(1 - APC_v))**(1/n)
fig_ax8.plot(CDK_v*K,APC_v,color=COL_apc,linestyle='-',label=r'$[APC]^*$')
fig_ax8.set_xlim([0,max(fig_ax5.get_xlim())])
fig_ax8.set_ylim([0,max(fig_ax5.get_ylim())])
fig_ax8.legend(bbox_to_anchor=(0.5, 0.5, 1, 0),loc='lower left',title='Nullclines',frameon=False) 

### Determine steady state values of the overall system
APC_v = np.linspace(1e-12,1-1e-12,10000000)
Xi = 1 + a*APC_v*(APC_v - 1)*(APC_v - r)     
Cdk1 = c/APC_v 
F = eps*(Cdk1**n/(Cdk1**n + Xi**n) - APC_v) # Nullclines intersect where F = 0
APC_roots = APC_v[1:][F[1:]*F[0:-1] < 0]    # Determine APC levels for which F = 0 (i.e. where F changes sign; i.e. if product of subsequent elements < 0)
Cdk1_roots = c/APC_roots                    # Calculate Cdk1 levels where F = 0
for X,Y in list(zip(Cdk1_roots,APC_roots)): # Determine stability of the steady states via time simulation
    y0 = [X+1e-6,Y+1e-6]                    # Use small perturbation from steady state as initial condition
    t_start = 0
    t_end = 50
    def f(t,y): return DBA.bist_2d_cubic(y,t,c,eps,r,n,a)   
    sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=0.01,rtol=1e-6,atol=1e-9)    # Solve the system
    if (abs(sol.y[0,:][-1]-X) < 1e-6) and (abs(sol.y[1,:][-1]-Y) < 1e-6):   # if time evolution approaches steady state: stable
        fc='k'  # Denote stable steady state with filled dot
    else:       # else: unstable
        fc='w'  # Denote unstable steady state with open dot     
    # Convert non-dimensionalized values to original dimensions for plotting
    fig_ax8.scatter(X*K,Y,s=10,zorder=5,edgecolor='k',facecolor=fc)


### Plot time trajectories in time domain for different initial conditions
y0 = [0,0]
t_start = 0
t_end = 25

def f(t,y): return DBA.bist_2d_cubic(y,t,c,eps,r,n,a)    
sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9)

t_v = sol.t
Cdk1_sol = sol.y[0,:]
Apc_sol = sol.y[1,:]

# Remove transient solution
t_new = DBA.remove_transient(Apc_sol,t_v)[1]
Cdk1_sol = sol.y[0,:][t_v >= t_new[0]]
Apc_sol = sol.y[1,:][t_v >= t_new[0]]
t_new = t_new - t_new[0]

# Convert non-dimensionalized values to original dimensions for plotting  
l_cdk1, = fig_ax9.plot(t_new/bdeg,Cdk1_sol*K,color=COL_cdk)
l_apc, = fig_ax9_right.plot(t_new/bdeg,Apc_sol,color= COL_apc)

fig_ax9.set_ylabel('[Cdk1] (nM)',color=l_cdk1.get_color())
fig_ax9_right.set_ylabel(r'$[APC]^*$',color=l_apc.get_color())
fig_ax9.set_xlim([0,80])
fig_ax9.set_ylim([0,50])
fig_ax9_right.set_ylim([0,1.05])

# Plot limit cycle in phase plane
fig_ax8.plot(Cdk1_sol*K,Apc_sol,c='k',alpha=0.5)


#%%
# =============================================================================
# SUPPLEMENTAL FIGURE RELATED TO FIGURE 3
# =============================================================================
# Fixed parameters
K = 20
n = 15
r = 0.5
bdeg = 0.1

fig = plt.figure(figsize=(8, 2.7),constrained_layout=True)
G = gridspec.GridSpec(1, 3, figure = fig)

fig_ax4 = fig.add_subplot(G[0, 0])
fig_ax4.set_xlabel('Relative synthesis c')
fig_ax4.set_ylabel('Bistable width w (nM)')
fig_ax4.annotate('A',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax2 = fig.add_subplot(G[0, 1])
fig_ax2.set_xlabel('Relative synthesis c')
fig_ax2.set_ylabel('Bistable width w (nM)')
fig_ax2.annotate('B',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax3 = fig.add_subplot(G[0, 2])
fig_ax3.set_xlabel('Relative synthesis c')
fig_ax3.set_ylabel('Bistable width w (nM)')
fig_ax3.annotate('C',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

### Load data for Cdk1 oscillations
CSV = DBA.find_csv_bistability(n,r,'Cdk1','Cubic')
Array,Period_Grid,Amplitude_Grid,SS_Grid,SS_Pos_Grid,A_v,C_v,W_v = DBA.csv_to_array_bistability(CSV[1],CSV[0],'Cubic',r,n)

# PERIOD OF THE OSCILLATIONS AS A FUNCTION OF C AND BISTABLE WIDTH FOR CUBIC XI
# =============================================================================
X,Y = np.meshgrid(np.array(C_v),np.array(W_v)*K)
cl = fig_ax4.contour(X,Y,Period_Grid/bdeg,levels=[1e-3],colors='k',linewidths=1)
Period_Grid[Period_Grid == 0] = np.nan
cs = fig_ax4.contourf(X,Y,Period_Grid/bdeg,levels=np.linspace(10,80,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax4,shrink=0.9,aspect=10,ticks=[10,45,80],location='bottom')
cb.ax.set_xlabel('[Cdk1] Period (min)')
cb.ax.xaxis.set_label_position('top')
fig_ax4.annotate(r'$K_{cdk,apc}$ = '+str(int(K))+' nM\nn = '+str(int(n))+'\nr = '+str(r),(0.6,12),c='k')
fig_ax4.set_xlim([0,1.2])
fig_ax4.set_ylim([0,25])

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)


# AMPLITUDE OF THE OSCILLATIONS AS A FUNCTION OF C AND BISTABLE WIDTH FOR CUBIC XI
# =============================================================================
cl = fig_ax2.contour(X,Y,Amplitude_Grid*K,levels=[1e-3],colors='k',linewidths=1)
Amplitude_Grid[Amplitude_Grid == 0] = np.nan
cs = fig_ax2.contourf(X,Y,Amplitude_Grid*K,levels=np.linspace(0,20,50),extend='max')
cb = fig.colorbar(cs,ax=fig_ax2,shrink=0.9,aspect=10,ticks=[0,10,20],location='bottom')
cb.ax.set_xlabel('[Cdk1] Amplitude (nM)')
cb.ax.xaxis.set_label_position('top')
fig_ax2.annotate(r'$K_{cdk,apc}$ = '+str(int(K))+' nM\nn = '+str(int(n))+'\nr = '+str(r),(0.6,12),c='k')
fig_ax2.set_xlim([0,max(fig_ax4.get_xlim())])
fig_ax2.set_ylim([0,max(fig_ax4.get_ylim())])

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)


### Load data for APC oscillations
CSV = DBA.find_csv_bistability(n,r,'Apc','Cubic')
Array,Period_Grid,Amplitude_Grid,SS_Grid,SS_Pos_Grid,A_v,C_v,W_v = DBA.csv_to_array_bistability(CSV[1],CSV[0],'Cubic',r,n)

# AMPLITUDE OF THE OSCILLATIONS AS A FUNCTION OF C AND BISTABLE WIDTH FOR CUBIC XI
# =============================================================================
X,Y = np.meshgrid(np.array(C_v),np.array(W_v)*K)
cl = fig_ax3.contour(X,Y,Amplitude_Grid,levels=[1e-3],colors='k',linewidths=1)
Amplitude_Grid[Amplitude_Grid == 0] = np.nan
cs = fig_ax3.contourf(X,Y,Amplitude_Grid,levels=np.linspace(0.5,1,50),extend='min')
cb = fig.colorbar(cs,ax=fig_ax3,shrink=0.9,aspect=10,ticks=[0.5,0.75,1],location='bottom')
cb.ax.set_xlabel(r'$[APC]^*$ Amplitude')
cb.ax.xaxis.set_label_position('top')
fig_ax3.annotate(r'$K_{cdk,apc}$ = '+str(int(K))+' nM\nn = '+str(int(n))+'\nr = '+str(r),(0.6,12),c='k')
fig_ax3.set_xlim([0,max(fig_ax4.get_xlim())])
fig_ax3.set_ylim([0,max(fig_ax4.get_ylim())])

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

# plt.savefig('Figure_Cubic_Bistability_Suppl.pdf', dpi=300) 


