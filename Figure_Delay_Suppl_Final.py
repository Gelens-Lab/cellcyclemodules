#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import Delay_Bistability_Analysis_Final as DBA
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.lines import Line2D

rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 8       # Annotation fonts
rcParams['axes.labelsize'] = 9
rcParams['axes.titlesize'] = 9
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
plt.ion()

# COLOR SCHEME
COL_v = ['#471164ff','#472f7dff','#3e4c8aff','#32648eff','#297a8eff','#21918cff','#21a685ff','#3bbb75ff','#69cd5bff','#a5db36ff','#e2e418ff']
COL_apc = COL_v[9]
COL_cdk = COL_v[5]

# Fixed parameters
eps = 100
K = 20
bdeg = 0.1

fig = plt.figure(figsize=(8, 2.7*3),constrained_layout=True)
G = gridspec.GridSpec(3, 3, figure = fig)

fig_ax1 = fig.add_subplot(G[0, 0])
fig_ax1.axis('off')
fig_ax1.annotate('A',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax2 = fig.add_subplot(G[0, 2])
fig_ax2.set_xlabel('Relative synthesis c')
fig_ax2.set_ylabel(r'Delay $\tau$ (min)')
fig_ax2.annotate('C',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax3 = fig.add_subplot(G[0, 1])
fig_ax3.set_xlabel('Relative synthesis c')
fig_ax3.set_ylabel(r'Delay $\tau$ (min)')
fig_ax3.annotate('B',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax4 = fig.add_subplot(G[1, 0])
fig_ax4.set_xlabel('Relative synthesis c')
fig_ax4.set_ylabel(r'Delay $\tau$ (min)')
fig_ax4.annotate('D',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax5 = fig.add_subplot(G[1, 1])
fig_ax5_right = fig_ax5.twinx()
fig_ax5.set_xlabel('Time t (min)')
fig_ax5.annotate('E',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax6 = fig.add_subplot(G[1, 2])
fig_ax6.set_xlabel('[Cdk1] (nM)')
fig_ax6.set_ylabel(r'$[APC]^*$')
fig_ax6.annotate('F',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax7 = fig.add_subplot(G[2, 0])
fig_ax7.set_xlabel('Relative synthesis c')
fig_ax7.set_ylabel(r'Delay $\tau$ (min)')
fig_ax7.annotate('G',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax8 = fig.add_subplot(G[2, 1])
fig_ax8_right = fig_ax8.twinx()
fig_ax8.set_xlabel('Time t (min)')
fig_ax8.annotate('H',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax9 = fig.add_subplot(G[2, 2])
fig_ax9.set_xlabel('[Cdk1] (nM)')
fig_ax9.set_ylabel(r'$[APC]^*$')
fig_ax9.annotate('I',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

n = 15

# PERIOD OF THE OSCILLATIONS AS A FUNCTION OF A AND C FOR CUBIC XI
# =============================================================================
CSV = DBA.find_csv_delay(n,'Cdk1')
Array,Period_Grid,Amplitude_Grid,Tau_v,C_v = DBA.csv_to_array_delay(CSV[1],CSV[0])
X,Y = np.meshgrid(np.array(C_v),np.array(Tau_v)/bdeg)
cl = fig_ax2.contour(X,Y,Period_Grid/bdeg,levels=[1e-3],colors='k')
Period_Grid[Period_Grid == 0] = np.nan
cs = fig_ax2.contourf(X,Y,Period_Grid/bdeg,levels=np.linspace(20,140,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax2,shrink=0.9,ticks=[20,80,140],location='bottom')
cb.ax.set_xlabel('[Cdk1] Period (min)')
cb.ax.xaxis.set_label_position('top')
fig_ax2.annotate('n = '+str(int(n)),(0.75,0.1),c='white',xycoords='axes fraction')
fig_ax2.set_xlim([0,1.2])
fig_ax2.set_ylim([0,30])

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)


# AMPLITUDE OF THE OSCILLATIONS AS A FUNCTION OF A AND C FOR CUBIC XI
# =============================================================================
CSV = DBA.find_csv_delay(n,'Cdk1')
Array,Period_Grid,Amplitude_Grid,Tau_v,C_v = DBA.csv_to_array_delay(CSV[1],CSV[0])
X,Y = np.meshgrid(np.array(C_v),np.array(Tau_v)/bdeg)
cl = fig_ax4.contour(X,Y,Period_Grid/bdeg,levels=[1e-3],colors='k')
Amplitude_Grid[Amplitude_Grid == 0] = np.nan
cs = fig_ax4.contourf(X,Y,Amplitude_Grid*K,levels=np.linspace(5,40,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax4,shrink=0.9,ticks=[5,22,40],location='bottom')
cb.ax.set_xlabel('[Cdk1] Amplitude (nM)')
cb.ax.xaxis.set_label_position('top')
fig_ax4.annotate('n = '+str(int(n)),(0.75,0.1),c='k',xycoords='axes fraction')
fig_ax4.set_xlim([0,max(fig_ax2.get_xlim())])
fig_ax4.set_ylim([0,max(fig_ax2.get_ylim())])

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

CSV = DBA.find_csv_delay(n,'Apc')
Array,Period_Grid,Amplitude_Grid,Tau_v,C_v = DBA.csv_to_array_delay(CSV[1],CSV[0])
X,Y = np.meshgrid(np.array(C_v),np.array(Tau_v)/bdeg)
cl = fig_ax7.contour(X,Y,Period_Grid/bdeg,levels=[1e-3],colors='k')
Amplitude_Grid[Amplitude_Grid == 0] = np.nan
cs = fig_ax7.contourf(X,Y,Amplitude_Grid,levels=np.linspace(0.5,1,50),extend='min')
cb = fig.colorbar(cs,ax=fig_ax7,shrink=0.9,ticks=[0.5,0.75,1],location='bottom')
cb.ax.set_xlabel(r'$[APC]^*$ Amplitude')
cb.ax.xaxis.set_label_position('top')
fig_ax7.annotate('n = '+str(int(n)),(0.75,0.1),c='k',xycoords='axes fraction')
fig_ax7.set_xlim([0,max(fig_ax2.get_xlim())])
fig_ax7.set_ylim([0,max(fig_ax2.get_ylim())])

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)


# BOUNDARIES OF THE OSCILLATORY REGION AS A FUNCTION OF A AND C FOR DIFFERENT N
# =============================================================================
# r = 0.5
L_v = []    # Label vector
H_v = []    # Handle vector
z = 3
for n in [5,15]:    
    CSV = DBA.find_csv_delay(n,'Cdk1')
    Array,Period_Grid,Amplitude_Grid,Tau_v,C_v = DBA.csv_to_array_delay(CSV[1],CSV[0])
    X,Y = np.meshgrid(np.array(C_v),np.array(Tau_v)/bdeg)

    cl = fig_ax3.contour(X,Y,Amplitude_Grid*K,levels=[1e-3],colors=COL_v[z],linewidths=2)
       
    H_v.append(Line2D([],[],c=COL_v[z]))
    L_v.append('n = '+str(int(n)))
    z = z + 3

fig_ax3.annotate('Oscillations',(0.2,0.8),c='k',xycoords='axes fraction')
fig_ax3.set_xlim([0,max(fig_ax2.get_xlim())])
fig_ax3.set_ylim([0,max(fig_ax2.get_ylim())])
fig_ax3.legend(H_v,L_v,bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=2,frameon=False)


# TIME TRACES OF THE SYSTEM
# =============================================================================
n = 15

### POINT 1
tau1 = 1 
c1 = 0.15 

fig_ax2.scatter(c1,tau1/bdeg,c='white',s=5)
fig_ax2.annotate('E-F',(c1+0.05,tau1/bdeg+0.5),c='white',fontweight='bold')
fig_ax4.scatter(c1,tau1/bdeg,c='white',s=5)
fig_ax4.annotate('E-F',(c1+0.05,tau1/bdeg+0.5),c='white',fontweight='bold')
fig_ax7.scatter(c1,tau1/bdeg,c='white',s=5)
fig_ax7.annotate('E-F',(c1+0.05,tau1/bdeg+0.5),c='white',fontweight='bold')

y0 = [0,0]
t_start = 0
t_end = 100
dt = 0.01

from jitcdde import jitcdde, y, t                  
f = DBA.delay_bist_2d_cubic(y,t,tau1,c1,eps,0,n,0)

DDE = jitcdde(f,verbose=False)
DDE.set_integration_parameters(max_step=0.01)
DDE.constant_past(y0)
DDE.step_on_discontinuities()

sol_1 = []
t_v = []
for time in np.arange(DDE.t, DDE.t+t_end, dt):
    t_v.append(time)
    sol_1.append(DDE.integrate(time))
sol_1 = np.array(sol_1)    
t_v = np.array(t_v)-tau1  

Cdk1_sol_1 = sol_1[:,0]
Apc_sol_1 = sol_1[:,1]

l_cdk1, = fig_ax5.plot(t_v/bdeg,Cdk1_sol_1*K,color=COL_cdk)
l_apc, = fig_ax5_right.plot(t_v/bdeg,Apc_sol_1,color=COL_apc)
fig_ax5.set_ylabel('[Cdk1] (nM)',color=l_cdk1.get_color())
fig_ax5_right.set_ylabel(r'$[APC]^*$',color=l_apc.get_color())
fig_ax5.set_xlim([0,500])
fig_ax5.set_ylim([0,75])
fig_ax5_right.set_ylim([0,1.05])


### POINT 2
tau2 = 2.5
c2 = 0.6

fig_ax2.scatter(c2,tau2/bdeg,c='white',s=5)
fig_ax2.annotate('H-I',(c2+0.05,tau2/bdeg+0.5),c='white',fontweight='bold')
fig_ax4.scatter(c2,tau2/bdeg,c='white',s=5)
fig_ax4.annotate('H-I',(c2+0.05,tau2/bdeg+0.5),c='white',fontweight='bold')
fig_ax7.scatter(c2,tau2/bdeg,c='white',s=5)
fig_ax7.annotate('H-I',(c2+0.05,tau2/bdeg+0.5),c='white',fontweight='bold')

y0 = [0,0]
t_start = 0
t_end = 100
dt = 0.01

from jitcdde import jitcdde, y, t                  
f = DBA.delay_bist_2d_cubic(y,t,tau2,c2,eps,0,n,0)

DDE = jitcdde(f,verbose=False)
DDE.set_integration_parameters(max_step=0.01)
DDE.constant_past(y0)
DDE.step_on_discontinuities()

sol_2 = []
t_v = []
for time in np.arange(DDE.t, DDE.t+t_end, dt):
    t_v.append(time)
    sol_2.append(DDE.integrate(time))
sol_2 = np.array(sol_2)    
t_v = np.array(t_v)-tau2  

Cdk1_sol_2 = sol_2[:,0]
Apc_sol_2 = sol_2[:,1]

l_cdk1, = fig_ax8.plot(t_v/bdeg,Cdk1_sol_2*K,color=COL_cdk)
l_apc, = fig_ax8_right.plot(t_v/bdeg,Apc_sol_2,color= COL_apc)
fig_ax8.set_ylabel('[Cdk1] (nM)',color=l_cdk1.get_color())
fig_ax8_right.set_ylabel(r'$[APC]^*$',color=l_apc.get_color())
fig_ax8.set_xlim([0,max(fig_ax5.get_xlim())])
fig_ax8.set_ylim([0,max(fig_ax5.get_ylim())])
fig_ax8_right.set_ylim([0,max(fig_ax5_right.get_ylim())])


# PHASE PLANES OF THE SYSTEM
# =============================================================================
### POINT 1
fig_ax6.plot(Cdk1_sol_1*K,Apc_sol_1,c='k',alpha=0.3)

# Cdk1 nullcline
APC_v = np.linspace(1e-12,1,1000)
CDK_v = c1/APC_v
fig_ax6.plot(CDK_v*K,APC_v,color=COL_cdk,linestyle='-',label='[Cdk1] nullcline')

# Apc nullcline
APC_v = np.linspace(0,1-1e-12,1000)
Xi_v = 1
CDK_v = Xi_v*(APC_v/(1 - APC_v))**(1/n)
fig_ax6.plot(CDK_v*K,APC_v,color=COL_apc,linestyle='-',label=r'$[APC]^*$ nullcline')
fig_ax6.set_xlim([0,max(fig_ax5.get_ylim())])
fig_ax6.set_ylim([0,max(fig_ax5_right.get_ylim())])
fig_ax6.legend(bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=1,frameon=False)


### POINT 2
fig_ax9.plot(Cdk1_sol_2*K,Apc_sol_2,c='k',alpha=0.3)

# Cdk1 nullcline
APC_v = np.linspace(1e-12,1,1000)
CDK_v = c2/APC_v
fig_ax9.plot(CDK_v*K,APC_v,color=COL_cdk,linestyle='-')

# Apc nullcline
APC_v = np.linspace(0,1-1e-12,1000)
Xi_v = 1 
CDK_v = Xi_v*(APC_v/(1 - APC_v))**(1/n)
fig_ax9.plot(CDK_v*K,APC_v,color=COL_apc,linestyle='-')
fig_ax9.set_xlim([0,max(fig_ax5.get_ylim())])
fig_ax9.set_ylim([0,max(fig_ax5_right.get_ylim())])

# plt.savefig('Figure_Delay_Suppl.pdf', dpi=300) 
