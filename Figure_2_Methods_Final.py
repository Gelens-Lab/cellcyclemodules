#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A modular approach for modeling the cell cycle based on functional response curves
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import Delay_Bistability_Analysis_Final as DBA

# Set plot parameters
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 8       # Annotation fonts
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
plt.ion()

# COLOR SCHEME
# Define some colors to use in plots
COL_v = ['#471164ff','#472f7dff','#3e4c8aff','#32648eff','#297a8eff','#21918cff','#21a685ff','#3bbb75ff','#69cd5bff','#a5db36ff','#e2e418ff']

# =============================================================================
# CODE TO GENERATE FIGURE 2 IN THE MAIN TEXT
# Block diagrams were added afterwards in inkscape
# =============================================================================
fig = plt.figure(figsize=(8, 2.5*2.7),constrained_layout=True)
G = gridspec.GridSpec(3, 3, figure = fig)

fig_ax1 = fig.add_subplot(G[0, 0])
fig_ax1.axis('off')
fig_ax1.annotate('A',(-0.2,1.05),size=12, weight='bold',xycoords='axes fraction')
fig_ax1.set_title('Ultrasensitivity',fontweight='bold')

fig_ax2 = fig.add_subplot(G[0, 1])
fig_ax2.axis('off')
fig_ax2.annotate('B',(-0.2,1.05),size=12, weight='bold',xycoords='axes fraction')
fig_ax2.set_title('Bistability',fontweight='bold')

fig_ax3 = fig.add_subplot(G[0, 2])
fig_ax3.axis('off')
fig_ax3.annotate('C',(-0.2,1.05),size=12, weight='bold',xycoords='axes fraction')
fig_ax3.set_title('Delay',fontweight='bold')

fig_ax4 = fig.add_subplot(G[1, 0])
fig_ax4.set_xlabel('Input')
fig_ax4.set_ylabel('Output')

fig_ax5 = fig.add_subplot(G[1, 1])
fig_ax5.set_xlabel('Input')
fig_ax5.set_ylabel('Output')

fig_ax6 = fig.add_subplot(G[1, 2])
fig_ax6.set_xlabel('Time')

fig_ax7 = fig.add_subplot(G[2, 0])
fig_ax7.axis('off')
fig_ax7.annotate('D',(-0.2,1.05),size=12, weight='bold',xycoords='axes fraction')


# PLOT HILL FUNCTION
# =============================================================================
K = 0.5
In = np.linspace(0,2*K,100)
z = 4   # Index to select colors
for n in [1,5,15]:
    Out = In**n/(K**n + In**n)
    fig_ax4.plot(In,Out,c=COL_v[z],label='n = '+str(n))

    z = z + 3

fig_ax4.set_yticks([0,0.5,1])
fig_ax4.set_xticks([0,K,2*K])
fig_ax4.set_xticklabels([0,r'$K$',r'$2K$'])
fig_ax4.set_xlim([0,max(In)])

fig_ax4.vlines(K,0,0.5,color='k',linestyle=':')
fig_ax4.hlines(0.5,0,K,color='k',linestyle=':')

fig_ax4.legend(loc='upper left',frameon=False)



# PLOT BISTABLE RESPONSE
# =============================================================================
n = 15
r = 0.5

In = np.linspace(0,2*K,500)
fig_ax5.set_xlim([0,max(In)])
fig_ax5.annotate('n = '+str(int(n)),(0.75,0.75),xycoords='axes fraction') 

# Calculate maximum value for alpha parameter that results in a positive left fold
xmin = (1 + r + np.sqrt((1 + r)**2 - 3*r))/3 
a_max = -1/(xmin*(xmin - r)*(xmin - 1))
a_v = np.linspace(0,a_max,3)

L = []  # Legend labels
H = []  # Legend handles
z = 4   # Color index
for a in a_v:
    Out = np.linspace(0,1-1e-12,100000)
    Xi_v = 1 + a*Out*(Out - r)*(Out - 1)
    w = DBA.width_cubic(r,n,a)[-1]
    
    In = K*Xi_v*(Out/(1 - Out))**(1/n)   
    
    h, = fig_ax5.plot(In,Out,color=COL_v[z])
    
    H.append(h)
    L.append(r'$\alpha$ = '+str(round(a,1)))
    
    z = z + 3
 
xtick_v = sorted([0,K,2*K])
xtick_label_v = [0,r'$K$',r'$2K$']
fig_ax5.set_xticks(xtick_v)
fig_ax5.set_xticklabels(xtick_label_v)
fig_ax5.set_yticks([0,0.5,1])

# Add legend
fig_ax5.legend(H,L,bbox_to_anchor=(-0.01, 0.2, 1, 0),loc='lower left',frameon=False)


# PLOT DELAYED RESPONSE
# =============================================================================
tau = 30
t = np.linspace(0,80,1000)
t_act = 20
A = [0 if i < t_act else 1 for i in t]
B = [0 if i < t_act+tau else 1 for i in t]
A, = fig_ax6.plot(t,A,c=COL_v[4])
fig_ax6.annotate('In',(t_act-10,0.75),size=10,color=A.get_color())
B, = fig_ax6.plot(t,B,c=COL_v[7])
fig_ax6.annotate('Out',(t_act+tau+3,0.75),size=10,color=B.get_color())
fig_ax6.set_yticks([0,0.5,1])
fig_ax6.set_xticks([])
fig_ax6.hlines(0.5,t_act+3,t_act+tau-3,colors='k',alpha=0.5)
fig_ax6.annotate(r'$\tau$',(33,0.55),size=10)

# plt.savefig('Figure_2.pdf', dpi=300) 
 

#%%
# =============================================================================
# CODE TO GENERATE THE FIGURE IN METHODS SECTION
# =============================================================================
fig = plt.figure(figsize=(7, 1.4*2.7),constrained_layout=True)
G = gridspec.GridSpec(2, 3, figure = fig)

fig_ax1 = fig.add_subplot(G[0, 0])
fig_ax1.set_xlabel('X (nM)')
fig_ax1.set_ylabel(r'$Y/Y_{tot}$')
fig_ax1.set_title('Ultrasensitive response',fontweight='bold')
fig_ax1.annotate('A',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax2 = fig.add_subplot(G[0, 1])
fig_ax2.set_ylabel('X (nM)')
fig_ax2.set_xlabel(r'$Y/Y_{tot}$')
fig_ax2.set_title('Inverted ultrasensitive',fontweight='bold')
fig_ax2.annotate('B',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax3 = fig.add_subplot(G[0, 2])
fig_ax3.set_xlabel(r'$Y/Y_{tot}$')
fig_ax3.set_ylabel(r'$\xi$')
fig_ax3.set_title('Scaling function',fontweight='bold')
fig_ax3.annotate('C',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax4 = fig.add_subplot(G[1, 0])
fig_ax4.set_xlabel('X (nM)')
fig_ax4.set_ylabel(r'$Y/Y_{tot}$')
fig_ax4.set_title('S-shaped response',fontweight='bold')
fig_ax4.annotate('D',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax5 = fig.add_subplot(G[1, 1])
fig_ax5.set_ylabel('X (nM)')
fig_ax5.set_xlabel(r'$Y/Y_{tot}$')
fig_ax5.set_title('Inverted S-shaped response',fontweight='bold')
fig_ax5.annotate('E',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax6 = fig.add_subplot(G[1, 2])
fig_ax6.axis('off')
fig_ax6.annotate(r'$X = Cdk1, Y = APC, Y_{tot} = APC_{tot}$'+'\n\nor\n\n'+r'$X = CycB, Y = Cdk1, Y_{tot} = CycB$',(-0.25,0.2),size=9,xycoords='axes fraction')


# HILL RESPONSE
# =============================================================================
K = 20
n = 15
X = np.linspace(0,2*K,100)
Y = X**n/(K**n + X**n)
# Plot Hill function
fig_ax1.plot(X,Y,c=COL_v[5],label='n = '+str(n))
# Plot inverted Hill function
fig_ax2.plot(Y,X,c=COL_v[5],label='n = '+str(n))


# SCALING FUNCTION XI
# =============================================================================
a = 9
r = 0.5
APC_v = np.linspace(0,1-5e-5,1000)
Xi_v = 1 + a*APC_v*(APC_v - r)*(APC_v - 1)
# Plot scaling funtion
fig_ax3.plot(APC_v,Xi_v,color=COL_v[9])


# BISTABLE RESPONSE 
# =============================================================================
CDK_v = K*Xi_v*(APC_v/(1 - APC_v))**(1/n)   

# Plot bistable response
fig_ax4.plot(CDK_v,APC_v,color=COL_v[5])
fig_ax4.set_xlim(fig_ax1.get_xlim())
# Plot inverted bistable response
fig_ax5.plot(APC_v,CDK_v,color=COL_v[5])
fig_ax5.set_ylim(fig_ax1.get_xlim())

xtick_v = sorted([0,1,r])
xtick_label_v = [str(i) if i != r else 'r' for i in xtick_v]
fig_ax3.set_xticks(xtick_v)
fig_ax3.set_xticklabels(xtick_label_v)
fig_ax3.vlines(r,0,1,linestyle=':',color='k')
fig_ax3.hlines(1,0,1,linestyle=':',color='k')
