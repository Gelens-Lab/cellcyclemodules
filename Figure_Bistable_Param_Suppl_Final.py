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

# =============================================================================
# CODE TO GENERATE SUPPLEMENTARY FIGURE 1
# Effect of different parameter values on the bistable response and oscillations
# =============================================================================
# Define some fixed parameter values
eps = 100
bdeg = 0.1

fig = plt.figure(figsize=(8, 2.7*3),constrained_layout=True)
G = gridspec.GridSpec(3, 3, figure = fig)

fig_ax4 = fig.add_subplot(G[0, 0])
fig_ax4.set_xlabel(r'$[APC]^*$')
fig_ax4.set_ylabel(r'$\xi$')
fig_ax4.annotate('A',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax5 = fig.add_subplot(G[0, 1])
fig_ax5.set_xlabel('[Cdk1] (nM)')
fig_ax5.set_ylabel(r'$[APC]^*$')
fig_ax5.annotate('B',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax6 = fig.add_subplot(G[0, 2])
fig_ax6.set_xlabel('Relative synthesis c')
fig_ax6.set_ylabel('Bistable width w (nM)')
fig_ax6.annotate('C',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax1 = fig.add_subplot(G[1, 0])
fig_ax1.set_ylabel(r'$[APC]^*$')
fig_ax1.set_xlabel('[Cdk1] (nM)')
fig_ax1.annotate('D',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax2 = fig.add_subplot(G[1, 1])
fig_ax2.set_xlabel('[Cdk1] (nM)')
fig_ax2.set_ylabel(r'$[APC]^*$')
fig_ax2.annotate('E',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax3 = fig.add_subplot(G[1, 2])
fig_ax3.set_xlabel('Relative synthesis c')
fig_ax3.set_ylabel('Bistable width w (nM)')
fig_ax3.annotate('F',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax7 = fig.add_subplot(G[2, 0])
fig_ax7.set_xlabel('[Cdk1] (nM)')
fig_ax7.set_ylabel(r'$[APC]^*$')
fig_ax7.annotate('G',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax8 = fig.add_subplot(G[2, 1])
fig_ax8.set_xlabel('[Cdk1] (nM)')
fig_ax8.set_ylabel(r'$[APC]^*$')
fig_ax8.annotate('H',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax9 = fig.add_subplot(G[2, 2])
fig_ax9.set_xlabel('Relative synthesis c')
fig_ax9.set_ylabel('Bistable width w (nM)')
fig_ax9.annotate('I',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')


# =============================================================================
# EFFECT OF N ON CUBIC BISTABILITY
# =============================================================================
K = 20
r = 0.5
c = 0.2
a = 7

# Original Hill response curve
CDK_v = np.linspace(0,3,1000)
z = 1
for n in [1,5,15,300]:
    APC_v = CDK_v**n/(1 + CDK_v**n)
    fig_ax1.plot(K*CDK_v,APC_v,c=COL_v[z],linestyle='-')
    fig_ax1.set_xlim([0,50])

    z = z + 3

# Bistable response in the phase plane 
L_v = []    # Label vector
H_v = []    # Handle vector
# Cdk1 nullcline
APC_v = np.linspace(1e-12,1,1000)
CDK_v = c/APC_v
h, = fig_ax2.plot(CDK_v*K,APC_v,color='k',linestyle=':',alpha=0.5)

z = 1
for n in [1,5,15,300]:
    # Apc nullcline
    APC_v = np.linspace(0,1-1e-12,1000)
    Xi_v = 1 + a*APC_v*(APC_v - 1)*(APC_v - r) 
    CDK_v = Xi_v*(APC_v/(1 - APC_v))**(1/n)
    CDK_v[-1] = 100
    h, = fig_ax2.plot(CDK_v*K,APC_v,color=COL_v[z],linestyle='-')

    z = z + 3

fig_ax2.annotate('c = '+str(c)+'\n'+'r = '+str(r)+'\n'+'a = '+str(a)+'\n'+r'$K_{cdk}$ = '+str(K)+' nM',(28,0.25),c='k')
fig_ax2.set_xlim([0,max(fig_ax1.get_xlim())])
fig_ax2.set_ylim([0,max(fig_ax1.get_ylim())])


# Oscillatory region in parameter space
L_v = []    # Label vector
H_v = []    # Handle vector
z = 1
for n in [1,5,15,300]: 
    # Load data from parameter scan
    CSV = DBA.find_csv_bistability(n,r,'Cdk1','Cubic')  
    Array,Period_Grid,Amplitude_Grid,SS_Grid,SS_Pos_Grid,A_v,C_v,W_v = DBA.csv_to_array_bistability(CSV[1],CSV[0],'Cubic',r,n)
    X,Y = np.meshgrid(np.array(C_v),np.array(W_v)*K)

    cl = fig_ax3.contour(X,Y,Amplitude_Grid*K,levels=[1e-6],colors=COL_v[z])
    H_v.append(Line2D([],[],c=COL_v[z]))
    L_v.append('n = '+str(int(n)))
    z = z + 3

fig_ax3.annotate('r = '+str(r)+'\n'+r'$K_{cdk}$ = '+str(K)+' nM',(0.55,0.5),c='k',xycoords='axes fraction')
fig_ax3.set_xlim([0,1.2])
fig_ax3.set_ylim([0,30])

fig_ax2.legend(H_v,L_v,bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=2,frameon=False)   



# =============================================================================
# EFFECT OF R ON CUBIC BISTABILITY
# =============================================================================
n = 15

# Cubic scaling function
z = 4
for r in [0.25,0.5,0.75]:
    x_max = (1+r-np.sqrt((1+r)**2-3*r))/3   # Apc right fold
    x_min = (1+r+np.sqrt((1+r)**2-3*r))/3   # Apc left fold
    
    APC_v = np.linspace(0,1-1e-12,100000)
    Xi_v = 1 + a*APC_v*(APC_v - r)*(APC_v - 1)
    h, = fig_ax4.plot(APC_v,Xi_v,color=COL_v[z],linestyle='-')
            
    z = z + 3

fig_ax4.set_yticks([0,0.5,1,1.5,2])
fig_ax4.set_ylim([0,2])
fig_ax4.hlines(1,0,1,linestyle=':',color='k',alpha=0.5)


# Bistable response in the phase plane 
L_v = []    # Label vector
H_v = []    # Handle vector
# Cdk1 nullcline
APC_v = np.linspace(1e-12,1,1000)
CDK_v = c/APC_v
h, = fig_ax5.plot(CDK_v*K,APC_v,color='k',linestyle=':',alpha=0.5)
H_v.append(h)
L_v.append('Cdk1 nullcline')

z = 4
for r in [0.25,0.5,0.75]:
    # Apc nullcline
    APC_v = np.linspace(0,1-1e-12,1000)
    Xi_v = 1 + a*APC_v*(APC_v - 1)*(APC_v - r) 
    CDK_v = Xi_v*(APC_v/(1 - APC_v))**(1/n)
    h, = fig_ax5.plot(CDK_v*K,APC_v,color=COL_v[z],linestyle='-')
    
    z = z + 3

fig_ax5.annotate('c = '+str(c)+'\n'+'n = '+str(n)+'\n'+'a = '+str(a)+'\n'+r'$K_{cdk}$ = '+str(K)+' nM',(28,0.6),c='k')
    
fig_ax5.set_xlim([0,50])
fig_ax5.set_ylim([0,1.05])


# Oscillatory region in parameter space
L_v = []    # Label vector
H_v = []    # Handle vector
z = 4
for r in [0.25,0.5,0.75]:    
    # Load data from parameter scan
    CSV = DBA.find_csv_bistability(n,r,'Cdk1','Cubic')  
    Array,Period_Grid,Amplitude_Grid,SS_Grid,SS_Pos_Grid,A_v,C_v,W_v = DBA.csv_to_array_bistability(CSV[1],CSV[0],'Cubic',r,n)
    X,Y = np.meshgrid(np.array(C_v),np.array(W_v)*K)

    cl = fig_ax6.contour(X,Y,Amplitude_Grid*K,levels=[1e-6],colors=COL_v[z])
    H_v.append(Line2D([],[],c=COL_v[z]))
    L_v.append('r = '+str(r))
    z = z + 3

fig_ax6.annotate('n = '+str(n)+'\n'+r'$K_{cdk}$ = '+str(K)+' nM',(0.08,22),c='k')
fig_ax6.set_xlim([0,1.2])
fig_ax6.set_ylim([0,30])

fig_ax5.legend(H_v,L_v,bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=2,frameon=False)   



# =============================================================================
# EFFECT OF K ON CUBIC BISTABILITY
# =============================================================================
r = 0.5

# Original Hill response curve
CDK_v = np.linspace(0,4,10000)
z = 4
for K in [10,20,30]:
    APC_v = CDK_v**n/(1 + CDK_v**n)
    fig_ax7.plot(K*CDK_v,APC_v,c=COL_v[z],linestyle='-')

    z = z + 3

fig_ax7.set_ylim([0,1.05])
fig_ax7.set_xlim([0,50])


# Bistable response in the phase plane 
L_v = []    # Label vector
H_v = []    # Handle vector
# Cdk1 nullcline
z = 4
for K in [10,20,30]:
    APC_v = np.linspace(1e-12,1,1000)
    CDK_v = c/APC_v
    h, = fig_ax8.plot(CDK_v*K,APC_v,color=COL_v[z],linestyle=':')

    z = z + 3

z = 4
# Apc nullcline
APC_v = np.linspace(0,1-1e-12,1000)
Xi_v = 1 + a*APC_v*(APC_v - 1)*(APC_v - r) 
CDK_v = Xi_v*(APC_v/(1 - APC_v))**(1/n)
for K in [10,20,30]:
    h, = fig_ax8.plot(CDK_v*K,APC_v,color=COL_v[z],linestyle='-')
    
    z = z + 3

fig_ax8.annotate('c = '+str(c)+'\n'+'n = '+str(n)+'\n'+'a = '+str(a)+'\n'+'r = '+str(r),(35,0.5),c='k')
    
fig_ax8.set_xlim([0,50])
fig_ax8.set_ylim([0,1.05])


# Oscillatory region in parameter space
L_v = []    # Label vector
H_v = []    # Handle vector
z = 4
CSV = DBA.find_csv_bistability(n,r,'Cdk1','Cubic')  
Array,Period_Grid,Amplitude_Grid,SS_Grid,SS_Pos_Grid,A_v,C_v,W_v = DBA.csv_to_array_bistability(CSV[1],CSV[0],'Cubic',r,n)
for K in [10,20,30]:
    X,Y = np.meshgrid(np.array(C_v),np.array(W_v)*K)
    
    cl = fig_ax9.contour(X,Y,Amplitude_Grid*K,levels=[1e-6],colors=COL_v[z])
    H_v.append(Line2D([],[],c=COL_v[z]))
    L_v.append(r'$K_{cdk}$ = '+str(K)+' nM')
    z = z + 3

fig_ax9.annotate('n = '+str(n)+'\n'+'r = '+str(r),(0.7,0.1),c='k',xycoords='axes fraction')
fig_ax9.set_xlim([0,1.2])
fig_ax9.set_ylim([0,30])

fig_ax8.legend(H_v,L_v,bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=2,frameon=False)   

# plt.savefig('Figure_Bistable_Param_Suppl.pdf', dpi=300) 

