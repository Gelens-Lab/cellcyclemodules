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
from matplotlib.lines import Line2D

# Set plot parameters
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 8       # Annotation fonts; specify before others
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
# CODE TO GENERATE FIGURE 5 IN THE MAIN TEXT
# Block diagrams were added afterwards in inkscape
# =============================================================================
# Define some fixed parameter values
eps = 100
K = 20
bdeg = 0.1
n = 15
r = 0.5

fig = plt.figure(figsize=(8, 2.7*2),constrained_layout=True)
G = gridspec.GridSpec(2, 3, figure = fig)

fig_ax1 = fig.add_subplot(G[0, 0])
fig_ax1.axis('off')
fig_ax1.annotate('A',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax2 = fig.add_subplot(G[0, 1])
fig_ax2.set_xlabel('Bistable width w (nM)')
fig_ax2.set_ylabel(r'Delay $\tau$ (min)')
fig_ax2.annotate('B',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax3 = fig.add_subplot(G[0, 2])
fig_ax3.set_xlabel('[Cdk1] (nM)')
fig_ax3.set_ylabel(r'$[APC]^*$')
fig_ax3.annotate('C',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax4 = fig.add_subplot(G[1, 0])
fig_ax4_right = fig_ax4.twinx()
fig_ax4.set_xlabel('Time t (min)')
fig_ax4.annotate('D',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')
fig_ax4.set_title(r'$\tau_1 = \tau_2$',weight='bold')

fig_ax5 = fig.add_subplot(G[1, 1])
fig_ax5_right = fig_ax5.twinx()
fig_ax5.set_xlabel('Time t (min)')
fig_ax5.annotate('E',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')
fig_ax5.set_title(r'$\tau_1 = \tau_2$',weight='bold')

fig_ax6 = fig.add_subplot(G[1, 2])
fig_ax6_right = fig_ax6.twinx()
fig_ax6.set_xlabel('Time t (min)')
fig_ax6.annotate('F',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')
fig_ax6.set_title(r'$\tau_1 \neq \tau_2$',weight='bold')


# PLOT BOUNDARIES OF THE OSCILLATORY REGION AS A FUNCTION OF BISTABLE WIDTH AND DELAY FOR DIFFERENT C
# =============================================================================
L_v = []    # Label vector
H_v = []    # Handle vector
z = 3
for c in [0.2,0.5,0.8]: 
    # Load data from csv file
    CSV = DBA.find_csv_delay_bistability(n,r,c,'Cdk1')
    Array,Period_Grid,Amplitude_Grid,A_v,Tau_v,W_v = DBA.csv_to_array_delay_bistability(CSV[1],CSV[0],r,n)
    X,Y = np.meshgrid(np.array(W_v)*K,np.array(Tau_v)/bdeg)

    cl = fig_ax2.contour(X,Y,Amplitude_Grid*K,levels=[1e-3],colors=COL_v[z])
    H_v.append(Line2D([],[],c=COL_v[z]))
    L_v.append('c = '+str(c))
    z = z + 3

fig_ax2.annotate('n = '+str(int(n))+'\nr = '+str(r)+'\n'+r'$K_{cdk,apc}$ = '+str(int(K))+' nM',(16,1.5),c='k')
fig_ax2.annotate('Oscillations',(0.05,0.52),c='k',xycoords='axes fraction',rotation=90)
fig_ax2.set_xlim([0,35])
fig_ax2.set_ylim([0,50])
fig_ax2.legend(H_v,L_v,bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=1,frameon=False)


# PLOT PHASE PLANE AND TIME TRACES OF THE SYSTEM
# =============================================================================
c = 0.5
a = 3
w = DBA.width_cubic(r,n,a)[-1]

### Select parameter values based on Fig5B
# POINT 1
tau1 = 2*bdeg
    
fig_ax2.scatter(w*K,tau1/bdeg,c='k',s=5)
fig_ax2.annotate('D',(w*K+0.5,tau1/bdeg+0.5),c='k',fontweight='bold')

y0 = [0,0]
t_start = 0
t_end = 200
dt = 0.001

from jitcdde import jitcdde, y, t                  
f = DBA.delay_bist_2d_cubic(y,t,tau1,c,eps,r,n,a)
DDE = jitcdde(f,verbose=False)
DDE.constant_past(y0)
DDE.step_on_discontinuities()

sol_1 = []
t_v = []
for time in np.arange(DDE.t, DDE.t+t_end, dt):
    t_v.append(time)
    sol_1.append(DDE.integrate(time))
sol_1 = np.array(sol_1)    
t_v = np.array(t_v) 
Cdk1_sol_1 = sol_1[:,0]
Apc_sol_1 = sol_1[:,1]

# Remove transient solution
t_new = DBA.remove_transient(Apc_sol_1,t_v,n_min=4)[1]
Cdk1_sol_1 = sol_1[:,0][t_v >= t_new[0]]
Apc_sol_1 = sol_1[:,1][t_v >= t_new[0]]
t_new = t_new - t_new[0]

l_cdk1, = fig_ax4.plot(t_new/bdeg,Cdk1_sol_1*K,color=COL_cdk)
l_apc, = fig_ax4_right.plot(t_new/bdeg,Apc_sol_1,color= COL_apc)
fig_ax4.set_ylabel('[Cdk1] (nM)',color=l_cdk1.get_color())
fig_ax4.set_ylim([0,50])
fig_ax4_right.set_ylim([0,1.05])
fig_ax4.set_xlim([0,150])


# POINT 2
tau2 = 20*bdeg

fig_ax2.scatter(w*K,tau2/bdeg,c='k',s=5)
fig_ax2.annotate('E',(w*K+0.5,tau2/bdeg+0.5),c='k',fontweight='bold')
# Indicate Cdk1 levels at folding points of nullcline
fig_ax5.hlines(DBA.width_cubic(r,n,a)[2]*K,0,150,linestyle='--',color=COL_cdk,alpha=0.5)
fig_ax5.hlines(DBA.width_cubic(r,n,a)[3]*K,0,150,linestyle='--',color=COL_cdk,alpha=0.5)

y0 = [0,0]
t_start = 0
t_end = 200
dt = 0.001

from jitcdde import jitcdde, y, t                  
f = DBA.delay_bist_2d_cubic(y,t,tau2,c,eps,r,n,a)
DDE = jitcdde(f,verbose=False)
DDE.constant_past(y0)
DDE.step_on_discontinuities()

sol_2 = []
t_v = []
for time in np.arange(DDE.t, DDE.t+t_end, dt):
    t_v.append(time)
    sol_2.append(DDE.integrate(time))
sol_2 = np.array(sol_2)    
t_v = np.array(t_v) 
Cdk1_sol_2 = sol_2[:,0]
Apc_sol_2 = sol_2[:,1]

# Remove transient solution
t_new = DBA.remove_transient(Apc_sol_2,t_v,n_min=2)[1]
Cdk1_sol_2 = sol_2[:,0][t_v >= t_new[0]]
Apc_sol_2 = sol_2[:,1][t_v >= t_new[0]]
t_new = t_new - t_new[0]

l_cdk1, = fig_ax5.plot(t_new/bdeg,Cdk1_sol_2*K,color=COL_cdk)
l_apc, = fig_ax5_right.plot(t_new/bdeg,Apc_sol_2,color= COL_apc)
fig_ax5.set_ylim([0,50])
fig_ax5_right.set_ylim([0,1.05])
fig_ax5.set_xlim([0,150])

# Add limit cycles and nullclines to the phase plane
fig_ax3.plot(Cdk1_sol_1*K,Apc_sol_1,c='k',alpha=0.3)
fig_ax3.plot(Cdk1_sol_2*K,Apc_sol_2,c='k',alpha=0.3)

# Cdk1 nullcline
APC_v = np.linspace(1e-12,1,1000)
CDK_v = c/APC_v
fig_ax3.plot(CDK_v*K,APC_v,color=COL_cdk,linestyle='-',label='[Cdk1] nullcline')

# Apc nullcline
APC_v = np.linspace(0,1-1e-12,1000)
Xi_v = 1 + a*APC_v*(APC_v - 1)*(APC_v - r) 
CDK_v = Xi_v*(APC_v/(1 - APC_v))**(1/n)
fig_ax3.plot(CDK_v*K,APC_v,color=COL_apc,linestyle='-',label=r'$[APC]^*$ nullcline')
fig_ax3.set_xlim([0,50])
fig_ax3.set_ylim([0,1.05])
fig_ax3.legend(bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=1,frameon=False) 
fig_ax3.annotate('D',(24,0.6),fontweight='bold')
fig_ax3.annotate('E',(41,0.6),fontweight='bold')



# PLOT TIME TRACES FOR STATE DEPENDENT DELAYS
# =============================================================================
tau1 = 3
tau2 = 0.75
# Indicate Cdk1 levels at folding points of nullcline
fig_ax6.hlines(DBA.width_cubic(r,n,a)[2]*K,0,150,linestyle='--',color=COL_cdk,alpha=0.5)
fig_ax6.hlines(DBA.width_cubic(r,n,a)[3]*K,0,150,linestyle='--',color=COL_cdk,alpha=0.5)

y0 = [0,0]
t_start = 0
t_end = 200
dt = 0.001

from jitcdde import jitcdde, y, t                  
f = DBA.delay_bist_2d_cubic_state_dependent(y,t,tau1,tau2,c,eps,r,n,a)
DDE = jitcdde(f,verbose=False,max_delay=1e2)
DDE.constant_past(y0)

sol = []
t_v = []
for time in np.arange(DDE.t, DDE.t+t_end, dt):
    t_v.append(time)
    sol.append(DDE.integrate_blindly(time))
sol = np.array(sol)    
t_v = np.array(t_v)  
Cdk1_sol = sol[:,0]
Apc_sol = sol[:,1]

t_new = DBA.remove_transient(Apc_sol,t_v,n_min=2)[1]
Cdk1_sol = sol[:,0][t_v >= t_new[0]]
Apc_sol = sol[:,1][t_v >= t_new[0]]
t_new = t_new - t_new[0]

l_cdk1, = fig_ax6.plot(t_new/bdeg,Cdk1_sol*K,color=COL_cdk)
l_apc, = fig_ax6_right.plot(t_new/bdeg,Apc_sol,color= COL_apc)

fig_ax6_right.set_ylabel(r'$[APC]^*$',color=l_apc.get_color())
fig_ax6.set_ylim([0,50])
fig_ax6_right.set_ylim([0,1.05])
fig_ax6.set_xlim([0,150])


# plt.savefig('Figure_5_Delay_Bistability.pdf', dpi=300) 


#%%
# =============================================================================
# SUPPLEMENTAL FIGURES RELATED TO FIGURE 5
# =============================================================================
# Fixed parameters
eps = 100
K = 20
bdeg = 0.1
n = 15
r = 0.5

fig = plt.figure(figsize=(8, 2.7),constrained_layout=True)
G = gridspec.GridSpec(1, 3, figure = fig)

fig_ax2 = fig.add_subplot(G[0, 0])
fig_ax2.set_xlabel('Bistable width w (nM)')
fig_ax2.set_ylabel(r'Delay $\tau$ (min)')
fig_ax2.annotate('A',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax4 = fig.add_subplot(G[0, 1])
fig_ax4.set_xlabel('Bistable width w (nM)')
fig_ax4.set_ylabel(r'Delay $\tau$ (min)')
fig_ax4.annotate('B',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax7 = fig.add_subplot(G[0, 2])
fig_ax7.set_xlabel('Bistable width w (nM)')
fig_ax7.set_ylabel(r'Delay $\tau$ (min)')
fig_ax7.annotate('C',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')


# PERIOD OF THE CDK1 OSCILLATIONS AS A FUNCTION OF W AND TAU FOR CUBIC XI
# =============================================================================
c = 0.5

# Load csv file with data for Cdk1 oscillations
CSV = DBA.find_csv_delay_bistability(n,r,c,'Cdk1')
Array,Period_Grid,Amplitude_Grid,A_v,Tau_v,W_v = DBA.csv_to_array_delay_bistability(CSV[1],CSV[0],r,n)
X,Y = np.meshgrid(np.array(W_v)*K,np.array(Tau_v)/bdeg)

cl = fig_ax2.contour(X,Y,Period_Grid/bdeg,levels=[1e-3],colors='k',linewidths=1)
Period_Grid[Period_Grid == 0] = np.nan
cs = fig_ax2.contourf(X,Y,Period_Grid/bdeg,levels=np.linspace(5,150,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax2,shrink=0.9,aspect=10,ticks=[5,50,100,150],location='bottom')
cb.ax.set_xlabel('[Cdk1] Period (min)')
cb.ax.xaxis.set_label_position('top')
fig_ax2.annotate(r'$K_{cdk}$ = '+str(int(K))+' nM\nn = '+str(int(n))+'\nr = '+str(r)+'\nc = '+str(c),(17,3),c='k')
fig_ax2.set_xlim([0,30])
fig_ax2.set_ylim([0,50])

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

# AMPLITUDE OF THE CDK1 OSCILLATIONS AS A FUNCTION OF W AND TAU FOR CUBIC XI
# =============================================================================
cl = fig_ax4.contour(X,Y,Amplitude_Grid*K,levels=[1e-3],colors='k',linewidths=1)
Amplitude_Grid[Amplitude_Grid == 0] = np.nan
cs = fig_ax4.contourf(X,Y,Amplitude_Grid*K,levels=np.linspace(0.5,60,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax4,shrink=0.9,aspect=10,ticks=[0.5,30,60],location='bottom')
cb.ax.set_xlabel('[Cdk1] Amplitude (nM)')
cb.ax.xaxis.set_label_position('top')
fig_ax4.annotate(r'$K_{cdk}$ = '+str(int(K))+' nM\nn = '+str(int(n))+'\nr = '+str(r)+'\nc = '+str(c),(17,3),c='k')
fig_ax4.set_xlim([0,30])
fig_ax4.set_ylim([0,50])

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)


# AMPLITUDE OF THE APC OSCILLATIONS AS A FUNCTION OF W AND TAU FOR CUBIC XI
# =============================================================================
# Load csv file with data for APC oscillations
CSV = DBA.find_csv_delay_bistability(n,r,c,'Apc')
Array,period_Grid,Amplitude_Grid,A_v,Tau_v,W_v = DBA.csv_to_array_delay_bistability(CSV[1],CSV[0],r,n)
X,Y = np.meshgrid(np.array(W_v)*K,np.array(Tau_v)/bdeg)

cl = fig_ax7.contour(X,Y,Amplitude_Grid,levels=[1e-3],colors='k',linewidths=1)
Amplitude_Grid[Amplitude_Grid == 0] = np.nan
cs = fig_ax7.contourf(X,Y,Amplitude_Grid,levels=np.linspace(0.5,1,50),extend='min')
cb = fig.colorbar(cs,ax=fig_ax7,shrink=0.9,aspect=10,ticks=[0.5,0.75,1],location='bottom')
cb.ax.set_xlabel(r'$[APC]^*$ Amplitude')
cb.ax.xaxis.set_label_position('top')
fig_ax7.annotate(r'$K_{cdk}$ = '+str(int(K))+' nM\nn = '+str(int(n))+'\nr = '+str(r)+'\nc = '+str(c),(17,3),c='k')
fig_ax7.set_xlim([0,30])
fig_ax7.set_ylim([0,50])

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

# plt.savefig('Figure_Delay_Bistability_Suppl.pdf', dpi=300) 
