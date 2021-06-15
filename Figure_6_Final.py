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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

# =============================================================================
# CODE TO GENERATE FIGURE 6 IN THE MAIN TEXT
# Block diagrams were added afterwards in inkscape
# =============================================================================
# Define some fixed parameter values
eps_apc = 100
eps_cdk = 100
bdeg = 0.1
r = 0.5
n = 15

fig = plt.figure(figsize=(8, 2.7*2.6),constrained_layout=True)
G = gridspec.GridSpec(3, 1, figure = fig)
G1 = G[0].subgridspec(1, 4,width_ratios=[0.6,1,1.1,1])
G3 = G[1].subgridspec(1, 3)
G2 = G[2].subgridspec(1, 4)

fig_ax1 = fig.add_subplot(G1[0, 0])
fig_ax1.axis('off')
fig_ax1.annotate('A',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax2 = fig.add_subplot(G1[0, 1])
fig_ax2.set_xlabel(r'Relative synthesis')
fig_ax2.set_ylabel(r'$W_{cdk,apc}$ (nM)')

fig_ax3 = fig.add_subplot(G1[0, 2])
fig_ax3.axis('off')
fig_ax3.annotate('B',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax4 = fig.add_subplot(G1[0, 3])
fig_ax4.set_xlabel(r'Relative synthesis')
fig_ax4.set_ylabel(r'$W_{cdk,apc}$ (nM)')

fig_ax5 = fig.add_subplot(G2[0, 0])
fig_ax5.set_xlabel(r'[Cdk1] (nM)')
fig_ax5.set_ylabel(r'$[APC]^*$')
fig_ax5.annotate('F',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax6 = fig.add_subplot(G2[0, 1])
fig_ax6.set_xlabel(r'[CycB] (nM)')
fig_ax6.set_ylabel(r'[Cdk1] (nM)')
fig_ax6.annotate('G',(-0.3,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax7 = fig.add_subplot(G2[0, 2])
fig_ax7.set_xlabel(r'[CycB] (nM)')
fig_ax7.set_ylabel(r'[Cdk1] (nM)')
fig_ax7.annotate('H',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax8 = fig.add_subplot(G2[0, 3])
fig_ax8.set_xlabel(r'[Cdk1] (nM)')
fig_ax8.set_ylabel(r'$[APC]^*$')
fig_ax8.annotate('I',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax9 = fig.add_subplot(G3[0, 0])
fig_ax9.set_xlabel(r'$K_{cyc,cdk}$ (nM)')
fig_ax9.set_ylabel(r'$K_{cdk,apc}$ (nM)')
fig_ax9.annotate('C',(-0.3,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax10 = fig.add_subplot(G3[0, 1])
fig_ax10.set_xlabel(r'$K_{cyc,cdk}$ (nM)')
fig_ax10.set_ylabel(r'$K_{cdk,apc}$ (nM)')
fig_ax10.annotate('D',(-0.3,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax11 = fig.add_subplot(G3[0, 2])
fig_ax11.set_xlabel(r'$K_{cyc,cdk}$ (nM)')
fig_ax11.set_ylabel(r'$K_{cdk,apc}$ (nM)')
fig_ax11.annotate('E',(-0.3,1),size=12, weight='bold',xycoords='axes fraction')


# PLOT OSCILLATORY REGIONS IN PARAMETER SPACE
# Width of the APC bistable curve vs. the relative synthesis c
# =============================================================================
Kcyc = 40   # Kcyccdk
Kcdk = 20   # Kcdkapc
d = Kcyc/Kcdk

### Read and plot data for system i; a screen for changing values of parameter alpha_apc and parameter c had been performed
# Load the csv file
CSV = DBA.find_csv_bistability(n,r,'Cdk1','Cubic')
# Retrieve required data from csv file
# The values of parameter alpha are converted to the width of the bistable function 
Array,Period_Grid,Amplitude_Grid,SS_Grid,SS_Pos_Grid,A_v,C_v,W_v = DBA.csv_to_array_bistability(CSV[1],CSV[0],'Cubic',r,n)
# For plotting the data, multiply the bistable width by Kcdkapc to convert from the non-dimensionalized system to original dimensions
# Likewise, divide the period by bdeg
X,Y = np.meshgrid(np.array(C_v),np.array(W_v)*Kcdk)     
cl = fig_ax2.contour(X,Y,Period_Grid/bdeg,levels=[1e-3],colors=COL_v[9],linewidths=2,zorder=3)
   
### Read and plot data for system iii; a screen for changing values of parameter alpha_apc and parameter c had been performed
# Alpha_cdk was fixed at two different values; 0 to retrieve an ultrasensitive response,
# and 9.48 to retrieve a bistable curve from CycB to Cdk1 with a width around 30 nM:
# DBA.width_cubic(0.5,15,9.48)[-1]*Kcyc ~ 30
for a_cdk,ax in [[0,fig_ax2],[9.48,fig_ax4]]:
    # Load the csv file
    CSV = DBA.find_csv_series_bistability_aapc_c(a_cdk,'Cdk1')
    # Retrieve required data from csv file
    # The values of parameter alpha are converted to the width of the bistable function 
    Array,Period_Grid,Amplitude_Grid,A_apc_v,C_v,W_cdk_v = DBA.csv_to_array_series_bistability_aapc_c(CSV[1],CSV[0],r,n)
    # For plotting the data, multiply the bistable width and Cdk1 amplitude by Kcdkapc to convert from the non-dimensionalized system to original dimensions
    # Divide parameter c by d to get the relative synthesis of system iii
    X,Y = np.meshgrid(np.array(C_v),np.array(W_cdk_v)*Kcdk)
    cl = ax.contour(X/d,Y,Amplitude_Grid*Kcdk,levels=[1e-3],colors='k',linewidths=1)
    Amplitude_Grid[Amplitude_Grid == 0] = np.nan
    cs = ax.contourf(X/d,Y,Amplitude_Grid*Kcdk,levels=np.linspace(5,55,50),extend='both')
    
    ax.set_ylim(0,35)
    ax.set_xlim(0,0.75)
    
    # Annotate the plot with parameter values
    w = DBA.width_cubic(r,n,a_cdk)[-1]*Kcyc     # Calculate width of the bistable curve from CycB to Cdk1: ~30 nM
    ax.annotate(r'$W_{cyc,cdk} = $'+str(int(w))+' nM\n'+r'$K_{cyc,cdk} = $'+str(int(Kcyc))+' nM\n'+r'$K_{cdk,apc} = $'+str(int(Kcdk))+' nM\n'+'n = '+str(int(n))+'\nr = '+str(r),(0.02,21),c='k')

    for i in cs.collections:
        i.set_edgecolor('face')

cb = fig.colorbar(cs,ax=fig_ax4,aspect=10,shrink=0.9,ticks=[5,30,55],pad=0)
cb.set_label('[Cdk1] Amplitude (nM)',rotation=270,labelpad=15)
cb.solids.set_rasterized(True)


# PLOT OSCILLATORY REGIONS IN PARAMETER SPACE
# Effect of altering the threshold values K
# =============================================================================
# A screen was performed to determine the oscillatory behaviour of system iii
# for different values of the threshold values K
# By altering K, the width of the bistable curve is also affected. Therefore, 
# the desired value of a (i.e. 9.48) was first multiplied by K (see supplemental),
# leading to the large values for a_apc and a_cdk here, i.e. 9.48*20 and 9.48*40 
for a_apc,a_cdk,ax,an,col in [[189.6,0,fig_ax9,'(iii-a)','w'],[0,379.2,fig_ax10,'(iii-b)','k'],[189.6,379.2,fig_ax11,'(iii-c)','w']]:
    # Load and plot the data from the csv file for Cdk1 oscillations
    CSV = DBA.find_csv_series_bistability_K('Cdk1',a_apc,a_cdk)
    Array,Period_Grid,Amplitude_Grid,K_cyc_v,K_cdk_v,W_cyc_v,W_cdk_v = DBA.csv_to_array_series_bistability_K(CSV[1],CSV[0])
    X,Y = np.meshgrid(np.array(K_cyc_v),np.array(K_cdk_v))
    cl = ax.contour(X,Y,Amplitude_Grid*Kcdk,levels=[4],colors='k')
    Amplitude_Grid[Amplitude_Grid == 0] = np.nan
    # Convert the non-dimensionalized Cdk1 amplitude to original dimensions by multiplying with Kcdkapc
    # Each row of the amplitude vector needs to be muliplied by different Kcdk, so convert K_cdk_v to correct dimensions
    cs = ax.contourf(X,Y,Amplitude_Grid*np.array([[j] for j in K_cdk_v]),levels=np.linspace(5,55,50),extend='both')

    w_cyc = int(round(DBA.width_cubic(r,n,a_cdk/Kcyc)[-1]*Kcyc,0))
    w_cdk = int(round(DBA.width_cubic(r,n,a_apc/Kcdk)[-1]*Kcdk,0))

    ax.set_title('System '+an,fontweight='bold')
    ax.annotate(r'$W_{cyc,cdk} \approx$'+str(w_cyc)+' nM\n'+r'$W_{cdk,apc} \approx$'+str(w_cdk)+' nM',(18,40),color=col)

    for i in cs.collections:
        i.set_edgecolor('face')

cb = fig.colorbar(cs,ax=fig_ax11,aspect=10,shrink=0.9,ticks=[5,30,55],pad=0)
cb.set_label('[Cdk1] Amplitude (nM)',rotation=270,labelpad=25)
cb.solids.set_rasterized(True)

fig_ax9.set_xlim([15,50])
fig_ax10.set_xlim([15,50])
fig_ax11.set_xlim([15,50])


# PLOT SMALL AND LARGE AMPLITUDE OSCILLATIONS IN THE PHASE PLANE
# =============================================================================
### SYSTEM III-A
# Fix some parameter values for interesting regions in Fig6B
c = 0.5
Kcyc = 40
Kcdk = 28
d = Kcyc/Kcdk
a_cdk = 0
a_apc = 189.6/Kcdk #9.48

fig_ax9.scatter(Kcyc,Kcdk,s=5,c='w',zorder=3)
fig_ax9.annotate('F',(41,30),c='w',fontweight='bold')

### Cdk1-APC phase plane; convert to original dimensions before plotting   
# Apc nullcline
APC_v = np.linspace(0,1-1e-12,1000) 
Xi_apc_v = 1 + a_apc*(APC_v)*((APC_v) - 1)*((APC_v) - r)   
CDK_v = Xi_apc_v*(APC_v/(1 - APC_v))**(1/n)
fig_ax5.plot(CDK_v*Kcdk,APC_v,c='k',linestyle='-',alpha=0.25)
fig_ax5.set_xlim([0,60])
fig_ax5.set_ylim([0,1.05]) 

### Add limit cycles to the phase planes
t_start = 0
t_end = 15
t_eval = np.linspace(t_start,t_end,10000)
y0 = [1e-12,1e-12,1e-12]
def f(t,y): return DBA.bist_2d_cubic_series(y,t,c,d,eps_apc,eps_cdk,r,n,a_apc,a_cdk)    
sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=0.1,rtol=1e-6,atol=1e-9)
Cdk1_sol = sol.y[1,:]
Apc_sol = sol.y[2,:]
# Remove the transient solution
t_new = DBA.remove_transient(Apc_sol,sol.t,25)[1]
Cdk1_sol = sol.y[1,:][sol.t >= t_new[0]]
Apc_sol = sol.y[2,:][sol.t >= t_new[0]]
fig_ax5.plot(Cdk1_sol*Kcdk,Apc_sol,c=COL_v[1],alpha=1)


### SYSTEM III-B
# Fix some parameter values for interesting regions in Fig6B
c = 0.5
Kcyc = 40
Kcdk = 18
d = Kcyc/Kcdk
a_cdk = 379.2/Kcyc #9.48
a_apc = 0

fig_ax10.scatter(Kcyc,Kcdk,s=5,c='k',clip_on=False)
fig_ax10.annotate('G',(42,18),c='k',fontweight='bold')

### Cyc-Cdk1 phase plane    
# Cdk nullcline 
CDKCYC_v = np.linspace(0,1-1e-12,1000)   # CDKCYC_v = Cdk1/(d*CycB)
Xi_cdk_v = 1 + a_cdk*CDKCYC_v*(CDKCYC_v - 1)*(CDKCYC_v - r)   
Cyc_v = Xi_cdk_v*(CDKCYC_v/(1 - CDKCYC_v))**(1/n)
# Convert non-dimensionalized values to original dimensions
# Multiply Cyc_v with Kcyccdk
# Multiply CDKCYC_v with Kcyccdk*Cyc_v (see supplemental information)
fig_ax6.plot(Cyc_v*Kcyc,CDKCYC_v*Cyc_v*Kcyc,c='k',linestyle='-',alpha=0.25)
fig_ax6.set_xlim([0,80])
fig_ax6.set_ylim([0,60]) 

### Add limit cycles to the phase planes
t_start = 0
t_end = 15
t_eval = np.linspace(t_start,t_end,10000)
y0 = [1e-12,1e-12,1e-12]
def f(t,y): return DBA.bist_2d_cubic_series(y,t,c,d,eps_apc,eps_cdk,r,n,a_apc,a_cdk)    
sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=0.1,rtol=1e-6,atol=1e-9)
Cyc_sol = sol.y[0,:]
Cdk1_sol = sol.y[1,:]
Apc_sol = sol.y[2,:]
# Remove the transient solution
t_new = DBA.remove_transient(Apc_sol,sol.t,25)[1]
Cyc_sol = sol.y[0,:][sol.t >= t_new[0]]
Cdk1_sol = sol.y[1,:][sol.t >= t_new[0]]
t_new = t_new - t_new[0]
fig_ax6.plot(Cyc_sol*Kcyc,Cdk1_sol*Kcdk,c=COL_v[10],alpha=1)


### SYSTEM III-C
# Fix some parameter values for interesting regions in Fig6B
c = 0.5
Kcyc = 40
Kcdk = 18
for Kcdk,col,textcol in [[18,COL_v[10],'k'],[28,COL_v[1],'w']]:
    d = Kcyc/Kcdk
    a_cdk = 379.2/Kcyc #9.48
    a_apc = 189.6/Kcdk #9.48
    
    fig_ax11.scatter(Kcyc,Kcdk,s=5,c=textcol,clip_on=False)
    fig_ax11.annotate('H,I',(38,Kcdk+2),c=textcol,fontweight='bold')
    
    ### Cyc-Cdk1 phase plane    
    # Cdk nullcline 
    CDKCYC_v = np.linspace(0,1-1e-12,1000)   # CDKCYC_v = Cdk1/(d*CycB)
    Xi_cdk_v = 1 + a_cdk*CDKCYC_v*(CDKCYC_v - 1)*(CDKCYC_v - r)   
    Cyc_v = Xi_cdk_v*(CDKCYC_v/(1 - CDKCYC_v))**(1/n)
    # Convert non-dimensionalized values to original dimensions
    # Multiply Cyc_v with Kcyccdk
    # Multiply CDKCYC_v with Kcyccdk*Cyc_v (see supplemental information)
    fig_ax7.plot(Cyc_v*Kcyc,CDKCYC_v*Cyc_v*Kcyc,c='k',linestyle='-',alpha=0.125)       

    ### Cdk1-APC phase plane; convert to original dimensions before plotting   
    # Apc nullcline
    APC_v = np.linspace(0,1-1e-12,1000) 
    Xi_apc_v = 1 + a_apc*(APC_v)*((APC_v) - 1)*((APC_v) - r)   
    CDK_v = Xi_apc_v*(APC_v/(1 - APC_v))**(1/n)
    fig_ax8.plot(CDK_v*Kcdk,APC_v,c=col,linestyle='-',alpha=0.75)
   
    ### Add limit cycles to the phase planes
    t_start = 0
    t_end = 15
    t_eval = np.linspace(t_start,t_end,10000)
    y0 = [1e-12,1e-12,1e-12]
    def f(t,y): return DBA.bist_2d_cubic_series(y,t,c,d,eps_apc,eps_cdk,r,n,a_apc,a_cdk)    
    sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=0.1,rtol=1e-6,atol=1e-9)
    Cyc_sol = sol.y[0,:]
    Cdk1_sol = sol.y[1,:]
    Apc_sol = sol.y[2,:]
    # Remove the transient solution
    t_new = DBA.remove_transient(Apc_sol,sol.t,22)[1]
    Cyc_sol = sol.y[0,:][sol.t >= t_new[0]]
    Cdk1_sol = sol.y[1,:][sol.t >= t_new[0]]
    Apc_sol = sol.y[2,:][sol.t >= t_new[0]]
     
    fig_ax7.plot(Cyc_sol*Kcyc,Cdk1_sol*Kcdk,c=col,alpha=1)
    fig_ax8.plot(Cdk1_sol*Kcdk,Apc_sol,c=col,alpha=1,zorder=3)

fig_ax7.set_xlim([0,80])
fig_ax7.set_ylim([0,60]) 
fig_ax8.set_xlim([0,60])
fig_ax8.set_ylim([0,1.05]) 

# plt.savefig('Figure_6_Two_Switches.pdf', dpi=300) 
 