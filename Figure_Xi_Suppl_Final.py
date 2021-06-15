#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A modular approach for modeling the cell cycle based on functional response curves
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

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

# =============================================================================
# COMPARISON OF DIFFERENT SCALING FUNCTIONS XI
# =============================================================================
fig = plt.figure(figsize=(8, 2*2.7),constrained_layout=True)
G = gridspec.GridSpec(2, 4, figure = fig)

fig_ax1 = fig.add_subplot(G[0, 0])
fig_ax1.set_ylabel(r'$\xi$')
fig_ax1.set_xlabel(r'$[APC]^*$')
fig_ax1.set_title('Cubic',fontweight='bold')
fig_ax1.annotate('A',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax2 = fig.add_subplot(G[0, 1])
fig_ax2.set_ylabel(r'$\xi$')
fig_ax2.set_xlabel(r'$[APC]^*$')
fig_ax2.set_title('Quadratic',fontweight='bold')
fig_ax2.annotate('B',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax3 = fig.add_subplot(G[0, 2])
fig_ax3.set_xlabel(r'$[APC]^*$')
fig_ax3.set_ylabel(r'$\xi$')
fig_ax3.set_title('Linear',fontweight='bold')
fig_ax3.annotate('C',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax4 = fig.add_subplot(G[1, 0])
fig_ax4.set_xlabel('[Cdk1] (nM)')
fig_ax4.set_ylabel(r'$[APC]^*$')
fig_ax4.annotate('E',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax5 = fig.add_subplot(G[1, 1])
fig_ax5.set_xlabel('[Cdk1] (nM)')
fig_ax5.set_ylabel(r'$[APC]^*$')
fig_ax5.annotate('F',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax6 = fig.add_subplot(G[1, 2])
fig_ax6.set_xlabel('[Cdk1] (nM)')
fig_ax6.set_ylabel(r'$[APC]^*$')
fig_ax6.annotate('G',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax7 = fig.add_subplot(G[0, 3])
fig_ax7.set_ylabel(r'$\xi$')
fig_ax7.set_xlabel(r'$[APC]^*$')
fig_ax7.set_title('Piecewise Linear',fontweight='bold')
fig_ax7.annotate('D',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax8 = fig.add_subplot(G[1, 3])
fig_ax8.set_xlabel('[Cdk1] (nM)')
fig_ax8.set_ylabel(r'$[APC]^*$')
fig_ax8.annotate('H',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')


### Scaling functions
n = 15
K = 20


# Cubic
a = 9
r = 0.5
APC_v = np.linspace(0,1-5e-5,1000)
Xi_v = 1 + a*APC_v*(APC_v - r)*(APC_v - 1)
fig_ax1.plot(APC_v,Xi_v,color=COL_v[5])
fig_ax1.annotate('a = '+str(a),(0.6,1.3))
CDK_v = K*Xi_v*(APC_v/(1 - APC_v))**(1/n)
fig_ax4.plot(CDK_v,APC_v,color=COL_v[5])

# Quadratic
a = 5
r = 0.5
APC_v = np.linspace(0,1-5e-6,1000)
Xi_v = 1 + a*(APC_v - r)*(APC_v - 1)
fig_ax2.plot(APC_v,Xi_v,color=COL_v[5])
fig_ax2.annotate('a = '+str(a),(0.6,3))
CDK_v = K*Xi_v*(APC_v/(1 - APC_v))**(1/n)
CDK_v[-1] = 60
fig_ax5.plot(CDK_v,APC_v,color=COL_v[5])


# Linear
a = 0.3
r = 0.5
APC_v = np.linspace(0,1-5e-6,1000)
Xi_v = (1 - a)/(r - 1)*(APC_v - 1) + a 
fig_ax3.plot(APC_v,Xi_v,color=COL_v[5])
fig_ax3.annotate('a = '+str(a),(0.6,1.5))
CDK_v = K*Xi_v*(APC_v/(1 - APC_v))**(1/n)
CDK_v[-1] = 40
fig_ax6.plot(CDK_v,APC_v,color=COL_v[5])


fig_ax1.hlines(1,0,1,linestyle='--',color='k')
fig_ax2.hlines(1,0,1,linestyle='--',color='k')
fig_ax3.hlines(1,0,1,linestyle='--',color='k')
fig_ax7.hlines(1,0,1,linestyle='--',color='k')


# Piecewise linear approximation of a cubic scaling function Xi
r = 0.5
a = 7
x_max = (1+r-np.sqrt((1+r)**2-3*r))/3               # APC coordinate at maximum of cubic Xi
xi_max = (1 + a*x_max*(x_max - 1)*(x_max - r))      # Maximum Xi value
x_min = (1+r+np.sqrt((1+r)**2-3*r))/3               # APC coordinate at minimium of cubic Xi
xi_min = (1 + a*x_min*(x_min - 1)*(x_min - r))      # Minimum Xi value

# Define piecewise linear function through extrema calculated above
X1 = np.linspace(0,x_max,1000)
Xi_1 = (xi_max-1)/(x_max)*X1+1
X2 = np.linspace(x_max,x_min,1000)
Xi_2 = (xi_min-xi_max)/(x_min-x_max)*(X2 - x_max) + xi_max
X3 = np.linspace(x_min,1-1e-12,1000)
Xi_3 = (1-xi_min)/(1-x_min)*(X3 - x_min) + xi_min
Xi_v = np.append(np.append(Xi_1,Xi_2),Xi_3)
APC_v = np.append(np.append(X1,X2),X3)
CDK_v = Xi_v*(APC_v/(1 - APC_v))**(1/n)
# Set last value of Cdk1 nullcline manually approach horizontal assymptote at APC = 1
CDK_v[-1] = 3

fig_ax7.plot(APC_v,Xi_v,color=COL_v[5])
fig_ax7.annotate('a = '+str(a),(0.6,1.25))
fig_ax8.plot(CDK_v*K,APC_v,color=COL_v[5])


# Plot original cubic
Xi_v = 1 + a*APC_v*(APC_v - 1)*(APC_v - r)    
CDK_v = Xi_v*(APC_v/(1 - APC_v))**(1/n)
fig_ax7.plot(APC_v,Xi_v,color='k',linestyle=':',alpha=0.5)
fig_ax8.plot(CDK_v*K,APC_v,color='k',linestyle=':',alpha=0.5)
fig_ax8.set_xlim([0,40])

# plt.savefig('Figure_Alternative_Xi.pdf', dpi=300) 
