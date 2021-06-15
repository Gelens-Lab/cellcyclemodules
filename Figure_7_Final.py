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
# CODE TO GENERATE FIGURE 7 IN THE MAIN TEXT
# Block diagrams were added afterwards in inkscape
# =============================================================================
# Define some fixed parameter values (see Table 2 main text)
r = 0.5
n = 15
a_cdk = 5
a_apc = 5
a_e2f = 5
deld = 0.05
delb = 0.05
eps_apc = 100   # Inverse from defined in text
eps_cdk = 100
eps_e2f = 100

Kd = 120
Kb = 40
Kcdk = 20
  
dsyn = 0.15
ddeg = 0.009
bsyn = 0.03
bdeg = 0.003

fig = plt.figure(figsize=(8, 2.7*1.8),constrained_layout=True)
G = gridspec.GridSpec(2, 1, figure = fig)
G1 = G[0].subgridspec(1, 4, width_ratios=[1,0.7,0.7,0.7])
G2 = G[1].subgridspec(1, 3, width_ratios=[0.95,0.75,0.75])

fig_ax1 = fig.add_subplot(G1[0, 0])
fig_ax1.axis('off')
fig_ax1.annotate('A',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax2 = fig.add_subplot(G1[0, 1])
fig_ax2.set_xlabel('Time t (h)')
fig_ax2.set_ylabel('Concentration (a.u.)')
fig_ax2.annotate('B',(-0.55,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax3 = fig.add_subplot(G1[0, 2])
fig_ax3.set_xlabel(r'[CycD] (a.u.)')
fig_ax3.set_ylabel(r'$[E2F]^*$ (a.u.)')
fig_ax3.annotate('C',(-0.55,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax4 = fig.add_subplot(G1[0, 3])
fig_ax4.set_xlabel(r'[CycB] (a.u.)')
fig_ax4.set_ylabel(r'[Cdk1] (a.u.)')
fig_ax4.annotate('D',(-0.55,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax5 = fig.add_subplot(G2[0, 0])
fig_ax5.axis('off')

fig_ax6 = fig.add_subplot(G2[0, 2])
fig_ax6.set_xlabel(r'$b_{syn}$ (nM/min)')
fig_ax6.set_ylabel(r'$b_{deg}$ (1/min)')
fig_ax6.annotate('F',(-0.45,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax7 = fig.add_subplot(G2[0, 1])
fig_ax7.set_xlabel(r'$d_{syn}$ (nM/min)')
fig_ax7.set_ylabel(r'$d_{deg}$ (1/min)')
fig_ax7.annotate('E',(-0.45,1),size=12, weight='bold',xycoords='axes fraction')

# TIME EVOLUTION 
# =============================================================================
t_start = 0
t_end = 10000
dt_max = 10
y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]

t_rp = t_end
t_g1 = t_end
delta_t_g1 = 0
t_g2 = t_end
delta_t_g2 = 0

t_in = t_end
def f(t,y): return DBA.bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb) 
     
sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=dt_max,rtol=1e-6,atol=1e-9)

# Remove the transient solution
t_new = DBA.remove_transient(sol.y[0,:],sol.t)[1]
CycD_sol = sol.y[0,:][sol.t >= t_new[0]]
E2F_sol = sol.y[1,:][sol.t >= t_new[0]]
CycB_sol = sol.y[2,:][sol.t >= t_new[0]]
Cdk_sol = sol.y[3,:][sol.t >= t_new[0]]
Apc_sol = sol.y[2,:][sol.t >= t_new[0]]
t_new = (t_new - t_new[0])/60
    
# Normalize solutions for their maximal value
fig_ax2.plot(t_new,CycD_sol/max(CycD_sol),c=COL_v[8],label='[CycD]',alpha=1)
fig_ax2.plot(t_new,E2F_sol,c=COL_v[0],label=r'$[E2F]^*$',alpha=1)

base = 1.2
fig_ax2.plot(t_new,base+CycB_sol/max(CycB_sol),c=COL_v[10],label='[CycB]',alpha=1)
fig_ax2.plot(t_new,base+Cdk_sol/max(Cdk_sol),c=COL_v[5],label='[Cdk1]',alpha=1)

fig_ax2.set_yticks([0,0.5,1,base+0,base+0.5,base+1])
fig_ax2.set_yticklabels([0,0.5,1,0,0.5,1])

fig_ax2.legend(bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=2,frameon=False)

fig_ax6.scatter(bsyn,bdeg,s=3,c='w',zorder=3)
fig_ax7.scatter(dsyn,ddeg,s=3,c='w',zorder=3)


# PHASE PLANES
# =============================================================================
# Plot E2F nullcline
E2F_v = np.linspace(1e-12,1-1e-12,500)
Xi_e2f_v = 1 + a_e2f*E2F_v*(E2F_v - 1)*(E2F_v - r)
CycD_v = Kd*Xi_e2f_v*(E2F_v/(1 - E2F_v))**(1/n)
fig_ax3.plot(CycD_v/max(CycD_sol),E2F_v,COL_v[0],linestyle='-')
fig_ax3.plot(CycD_sol/max(CycD_sol),E2F_sol,c='k',linestyle='-',alpha=0.5)
fig_ax3.set_xlim([0.25,1])
fig_ax3.set_ylim([-0.05,1.05])
fig_ax3.scatter(DBA.width_cubic(r,n,a_e2f)[2]*Kd/max(CycD_sol),DBA.width_cubic(r,n,a_e2f)[0],s=10,c='k')
fig_ax3.legend([Line2D([],[],c='w',marker='o',markerfacecolor='k')],['$[E2F]^*$ activation \nthreshold'],bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=1,frameon=False)


# Plot Cdk1 nullcline
CDKCYC_v = np.linspace(1e-12,1-1e-12,500)
Xi_cdk_v = 1 + a_cdk*CDKCYC_v*(CDKCYC_v - 1)*(CDKCYC_v - r)     
CycB_v = Kb*Xi_cdk_v*(CDKCYC_v/(1 - CDKCYC_v))**(1/n)
fig_ax4.plot(CycB_v/max(CycB_sol),(CDKCYC_v*CycB_v)/max(Cdk_sol),COL_v[5],linestyle='-')
fig_ax4.plot(CycB_sol/max(CycB_sol),Cdk_sol/max(Cdk_sol),c='k',alpha=0.5)
fig_ax4.set_xlim([0.5,1.1])
fig_ax4.set_ylim([-0.05,1.05])
fig_ax4.scatter(DBA.width_cubic(r,n,a_cdk)[2]*Kb/max(CycB_sol),DBA.width_cubic(r,n,a_cdk)[0],s=10,c='k')
fig_ax4.legend([Line2D([],[],c='w',marker='o',markerfacecolor='k')],['$[Cdk1]$ activation \nthreshold'],bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=1,frameon=False)


# PERIOD OF THE OSCILLATIONS AS A FUNCTION OF dsyn AND ddeg
# =============================================================================
CSV = DBA.find_csv_chain('dsyn','ddeg')

Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Dsyn_v,Ddeg_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'dsyn','ddeg')
X,Y = np.meshgrid(np.array(Dsyn_v),np.array(Ddeg_v))
Period_Grid = Period_Grid/60
cl = fig_ax7.contour(X,Y,Period_Grid,levels=[1e-3],colors='k',linewidths=1)
Period_Grid[Period_Grid == 0] = np.nan
cs = fig_ax7.contourf(X,Y,Period_Grid,levels=np.linspace(10,80,50),extend='both')

for i in cs.collections:
    i.set_edgecolor('face')

# Grey overlay for regions with irregular oscillations, i.e. where the average APC or E2F
# amplitude is smaller than 0.95
Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Dsyn_v,Ddeg_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'dsyn','ddeg')
X,Y = np.meshgrid(np.array(Dsyn_v),np.array(Ddeg_v))
Period_Grid[((E2F_Grid < 0.95) | (APC_Grid < 0.95)) & (Period_Grid != 0)] = -1
Period_Grid[Period_Grid != -1] = np.nan
cs = fig_ax7.contourf(X,Y,Period_Grid,levels=1,extend='both',colors='#cccccc')


# PERIOD OF THE OSCILLATIONS AS A FUNCTION OF bsyn AND bdeg
# =============================================================================
CSV = DBA.find_csv_chain('bsyn','bdeg')

Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Bsyn_v,Bdeg_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'bsyn','bdeg')
X,Y = np.meshgrid(np.array(Bsyn_v),np.array(Bdeg_v))
Period_Grid = Period_Grid/60
cl = fig_ax6.contour(X,Y,Period_Grid,levels=[1e-3],colors='k',linewidths=1)
Period_Grid[Period_Grid == 0] = np.nan
cs = fig_ax6.contourf(X,Y,Period_Grid,levels=np.linspace(10,80,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax6,shrink=0.9,aspect=20,ticks=[10,45,80],location='right',pad=0)
cb.set_label('Period (h)',rotation=270,labelpad=15)

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

# Grey overlay for regions with irregular oscillations, i.e. where the average APC or E2F
# amplitude is smaller than 0.95
Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Bsyn_v,Bdeg_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'bsyn','bdeg')
X,Y = np.meshgrid(np.array(Bsyn_v),np.array(Bdeg_v))
Period_Grid[((E2F_Grid < 0.95) | (APC_Grid < 0.95)) & (Period_Grid != 0)] = -1
Period_Grid[Period_Grid != -1] = np.nan
cs = fig_ax6.contourf(X,Y,Period_Grid,levels=1,extend='both',colors='#cccccc')

# plt.savefig('Figure_7_Chain.pdf', dpi=300) 


#%%
# =============================================================================
# SUPPLEMENTAL FIGURES RELATED TO FIGURE 7
# =============================================================================

# =============================================================================
# TIME TRACES FOR IRREGULAR OSCILLATIONS
# =============================================================================

# Fixed parameters
r = 0.5
n = 15
a_cdk = 5
a_apc = 5
a_e2f = 5
deld = 0.05
delb = 0.05
eps_apc = 100
eps_cdk = 100
eps_e2f = 100

Kd = 120
Kb = 40
Kcdk = 20

fig = plt.figure(figsize=(6, 2.7*3),constrained_layout=True)
G = gridspec.GridSpec(3, 2, figure = fig)

fig_ax1 = fig.add_subplot(G[0, 0])
fig_ax1.set_xlabel(r'$d_{syn}$ (nM/min)')
fig_ax1.set_ylabel(r'$d_{deg}$ (1/min)')
fig_ax1.annotate('A',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax2 = fig.add_subplot(G[0, 1])
fig_ax2.set_xlabel(r'$b_{syn}$ (nM/min)')
fig_ax2.set_ylabel(r'$b_{deg}$ (1/min)')
fig_ax2.annotate('B',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax4 = fig.add_subplot(G[1, 0])
fig_ax4.set_xlabel('Time (h)')
fig_ax4.set_ylabel('Concentration (a.u.)')
fig_ax4.set_title('E2F',fontweight='bold')
fig_ax4.annotate('C',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax5 = fig.add_subplot(G[1, 1])
fig_ax5.set_xlabel('Time (h)')
fig_ax5.set_ylabel('Concentration (a.u.)')
fig_ax5.set_title('E2F',fontweight='bold')
fig_ax5.annotate('D',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax7 = fig.add_subplot(G[2, 0])
fig_ax7.set_xlabel('Time (h)')
fig_ax7.set_ylabel('Concentration (a.u.)')
fig_ax7.set_title('E2F',fontweight='bold')
fig_ax7.annotate('E',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax8 = fig.add_subplot(G[2, 1])
fig_ax8.set_xlabel('Time (h)')
fig_ax8.set_ylabel('Concentration (a.u.)')
fig_ax8.set_title('E2F',fontweight='bold')
fig_ax8.annotate('F',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')


# PERIOD OF THE OSCILLATIONS AS A FUNCTION OF dsyn AND ddeg
# =============================================================================
CSV = DBA.find_csv_chain('dsyn','ddeg')

Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Dsyn_v,Ddeg_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'dsyn','ddeg')
X,Y = np.meshgrid(np.array(Dsyn_v),np.array(Ddeg_v))
Period_Grid = Period_Grid/60
cl = fig_ax1.contour(X,Y,Period_Grid,levels=[1e-3],colors='k',linewidths=1)
Period_Grid[Period_Grid == 0] = np.nan
cs = fig_ax1.contourf(X,Y,Period_Grid,levels=np.linspace(10,80,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax1,shrink=0.9,aspect=20,ticks=[10,45,80],location='bottom')
cb.ax.set_xlabel('Period (h)')

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

# Grey overlay for regions with irregular oscillations, i.e. where the average APC or E2F
# amplitude is smaller than 0.95
Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Dsyn_v,Ddeg_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'dsyn','ddeg')
X,Y = np.meshgrid(np.array(Dsyn_v),np.array(Ddeg_v))
Period_Grid[((E2F_Grid < 0.95) | (APC_Grid < 0.95)) & (Period_Grid != 0)] = -1
Period_Grid[Period_Grid != -1] = np.nan
cs = fig_ax1.contourf(X,Y,Period_Grid,levels=1,extend='both',colors='#cccccc')



# PERIOD OF THE OSCILLATIONS AS A FUNCTION OF bsyn AND bdeg
# =============================================================================
CSV = DBA.find_csv_chain('bsyn','bdeg')

Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Bsyn_v,Bdeg_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'bsyn','bdeg')
X,Y = np.meshgrid(np.array(Bsyn_v),np.array(Bdeg_v))
Period_Grid = Period_Grid/60
cl = fig_ax2.contour(X,Y,Period_Grid,levels=[1e-3],colors='k',linewidths=1)
Period_Grid[Period_Grid == 0] = np.nan
cs = fig_ax2.contourf(X,Y,Period_Grid,levels=np.linspace(10,80,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax2,shrink=0.9,aspect=20,ticks=[10,45,80],location='bottom')
cb.ax.set_xlabel('Period (h)')

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

# Grey overlay for regions with irregular oscillations, i.e. where the average APC or E2F
# amplitude is smaller than 0.95
Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Bsyn_v,Bdeg_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'bsyn','bdeg')
X,Y = np.meshgrid(np.array(Bsyn_v),np.array(Bdeg_v))
Period_Grid[((E2F_Grid < 0.95) | (APC_Grid < 0.95)) & (Period_Grid != 0)] = -1
Period_Grid[Period_Grid != -1] = np.nan
cs = fig_ax2.contourf(X,Y,Period_Grid,levels=1,extend='both',colors='#cccccc')


bsyn = 0.03
bdeg = 0.003
ddeg = 0.006
for dsyn,ax,an in [[0.18,fig_ax4,'C'],[0.25,fig_ax7,'E']]:
    fig_ax1.scatter(dsyn,ddeg,s=5,c='k')
    fig_ax1.annotate(an,(dsyn+0.005,ddeg+0.0005),fontweight='bold')
    
    t_start = 0
    t_end = 10000
    dt_max = 10
    y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]
    
    t_rp = t_end
    t_g1 = t_end
    delta_t_g1 = 0
    t_g2 = t_end
    delta_t_g2 = 0
    
    t_in = t_end
    def f(t,y): return DBA.bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb) 
         
    sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=dt_max,rtol=1e-6,atol=1e-9)
    
    t_v = sol.t
    E2F_sol = sol.y[1,:]
    Cdk_sol = sol.y[3,:]
    Apc_sol = sol.y[4,:]
              
    t_new = DBA.remove_transient(E2F_sol,t_v,n_min=1)[1]
    E2F_sol = sol.y[1,:][t_v >= t_new[0]]
    Cdk_sol = sol.y[3,:][t_v >= t_new[0]]
    Apc_sol = sol.y[4,:][t_v >= t_new[0]]
    t_new = (t_new - t_new[0])/60
        
    ax.plot(t_new,E2F_sol,c='k',label='E2F',alpha=1)
    ax.set_ylim([0,1.05])
    
dsyn = 0.15
ddeg = 0.009
for bsyn,bdeg,ax,an in [[0.015,0.004,fig_ax5,'D'],[0.025,0.0045,fig_ax8,'F']]:
    fig_ax2.scatter(bsyn,bdeg,s=5,c='k')
    fig_ax2.annotate(an,(bsyn+0.001,bdeg),fontweight='bold')
    
    t_start = 0
    t_end = 10000
    dt_max = 10
    y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]
    
    t_rp = t_end
    t_g1 = t_end
    delta_t_g1 = 0
    t_g2 = t_end
    delta_t_g2 = 0
    
    t_in = t_end
    def f(t,y): return DBA.bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb) 
         
    sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=dt_max,rtol=1e-6,atol=1e-9)
    
    t_v = sol.t
    E2F_sol = sol.y[1,:]
    Cdk_sol = sol.y[3,:]
    Apc_sol = sol.y[4,:]
              
    t_new = DBA.remove_transient(E2F_sol,t_v,n_min=1)[1]
    E2F_sol = sol.y[1,:][t_v >= t_new[0]]
    Cdk_sol = sol.y[3,:][t_v >= t_new[0]]
    Apc_sol = sol.y[4,:][t_v >= t_new[0]]
    t_new = (t_new - t_new[0])/60
        
    ax.plot(t_new,E2F_sol,c='k',label='E2F',alpha=1)
    ax.set_ylim([0,1.05])

# plt.savefig('Figure_Chain_Grey_Areas_Suppl.pdf', dpi=300) 
    
#%%
# =============================================================================
# DURATION OF SEPARATE CELL CYCLE PHASES
# =============================================================================
fig = plt.figure(figsize=(8, 2.7*2),constrained_layout=True)
G = gridspec.GridSpec(2, 3, figure = fig)

fig_ax4 = fig.add_subplot(G[0, 0])
fig_ax4.set_xlabel(r'$d_{syn}$ (nM/min)')
fig_ax4.set_ylabel(r'$d_{deg}$ (1/min)')
fig_ax4.annotate('A',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')
fig_ax4.set_title('G1 Phase',fontweight='bold')

fig_ax5 = fig.add_subplot(G[0, 1])
fig_ax5.set_xlabel(r'$d_{syn}$ (nM/min)')
fig_ax5.set_ylabel(r'$d_{deg}$ (1/min)')
fig_ax5.annotate('B',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')
fig_ax5.set_title('S/G2 Phase',fontweight='bold')

fig_ax6 = fig.add_subplot(G[0, 2])
fig_ax6.set_xlabel(r'$d_{syn}$ (nM/min)')
fig_ax6.set_ylabel(r'$d_{deg}$ (1/min)')
fig_ax6.annotate('C',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')
fig_ax6.set_title('M Phase',fontweight='bold')

fig_ax1 = fig.add_subplot(G[1, 0])
fig_ax1.set_xlabel(r'$b_{syn}$ (nM/min)')
fig_ax1.set_ylabel(r'$b_{deg}$ (1/min)')
fig_ax1.annotate('D',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax2 = fig.add_subplot(G[1, 1])
fig_ax2.set_xlabel(r'$b_{syn}$ (nM/min)')
fig_ax2.set_ylabel(r'$b_{deg}$ (1/min)')
fig_ax2.annotate('E',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax3 = fig.add_subplot(G[1, 2])
fig_ax3.set_xlabel(r'$b_{syn}$ (nM/min)')
fig_ax3.set_ylabel(r'$b_{deg}$ (1/min)')
fig_ax3.annotate('F',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')


# PHASE DURATIONS AS A FUNCTION OF dsyn AND ddeg
# =============================================================================
CSV = DBA.find_csv_chain('dsyn','ddeg')
Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Dsyn_v,Ddeg_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'dsyn','ddeg')
X,Y = np.meshgrid(np.array(Dsyn_v),np.array(Ddeg_v))
G1_Grid = G1_Grid/60
SG2_Grid = SG2_Grid/60
M_Grid = M_Grid/60
cl = fig_ax4.contour(X,Y,G1_Grid,levels=[1e-3],colors='k',linewidths=1)
cl = fig_ax5.contour(X,Y,SG2_Grid,levels=[1e-3],colors='k',linewidths=1)
cl = fig_ax6.contour(X,Y,M_Grid,levels=[1e-3],colors='k',linewidths=1)
G1_Grid[G1_Grid == 0] = np.nan
SG2_Grid[SG2_Grid == 0] = np.nan
M_Grid[M_Grid == 0] = np.nan
cs = fig_ax4.contourf(X,Y,G1_Grid,levels=np.linspace(0.1,30,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax4,shrink=0.9,ticks=[0.1,15,30],location='bottom')
cb.ax.set_xlabel('Duration (h)')
for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

cs = fig_ax5.contourf(X,Y,SG2_Grid,levels=np.linspace(0.1,30,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax5,shrink=0.9,ticks=[0.1,15,30],location='bottom')
cb.ax.set_xlabel('Duration (h)')
for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

cs = fig_ax6.contourf(X,Y,M_Grid,levels=np.linspace(0.1,30,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax6,shrink=0.9,ticks=[0.1,15,30],location='bottom')
cb.ax.set_xlabel('Duration (h)')
for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Dsyn_v,Ddeg_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'dsyn','ddeg')
X,Y = np.meshgrid(np.array(Dsyn_v),np.array(Ddeg_v))
G1_Grid[((E2F_Grid < 0.95) | (APC_Grid < 0.95)) & (Period_Grid != 0)] = -1
G1_Grid[G1_Grid != -1] = np.nan
cs = fig_ax4.contourf(X,Y,G1_Grid,levels=1,extend='both',colors='#cccccc')

SG2_Grid[((E2F_Grid < 0.95) | (APC_Grid < 0.95)) & (Period_Grid != 0)] = -1
SG2_Grid[SG2_Grid != -1] = np.nan
cs = fig_ax5.contourf(X,Y,SG2_Grid,levels=1,extend='both',colors='#cccccc')

M_Grid[((E2F_Grid < 0.95) | (APC_Grid < 0.95)) & (Period_Grid != 0)] = -1
M_Grid[M_Grid != -1] = np.nan
cs = fig_ax6.contourf(X,Y,M_Grid,levels=1,extend='both',colors='#cccccc')


# PHASE DURATIONS AS A FUNCTION OF bsyn AND bdeg
# =============================================================================
CSV = DBA.find_csv_chain('bsyn','bdeg')
Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Bsyn_v,Bdeg_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'bsyn','bdeg')
X,Y = np.meshgrid(np.array(Bsyn_v),np.array(Bdeg_v))
G1_Grid = G1_Grid/60
SG2_Grid = SG2_Grid/60
M_Grid = M_Grid/60
cl = fig_ax1.contour(X,Y,G1_Grid,levels=[1e-3],colors='k',linewidths=1)
cl = fig_ax2.contour(X,Y,SG2_Grid,levels=[1e-3],colors='k',linewidths=1)
cl = fig_ax3.contour(X,Y,M_Grid,levels=[1e-3],colors='k',linewidths=1)
G1_Grid[G1_Grid == 0] = np.nan
SG2_Grid[SG2_Grid == 0] = np.nan
M_Grid[M_Grid == 0] = np.nan
cs = fig_ax1.contourf(X,Y,G1_Grid,levels=np.linspace(0.1,30,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax1,shrink=0.9,ticks=[0.1,15,30],location='bottom')
cb.ax.set_xlabel('Duration (h)')
for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

cs = fig_ax2.contourf(X,Y,SG2_Grid,levels=np.linspace(0.1,30,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax2,shrink=0.9,ticks=[0.1,15,30],location='bottom')
cb.ax.set_xlabel('Duration (h)')
for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

cs = fig_ax3.contourf(X,Y,M_Grid,levels=np.linspace(0.1,30,50),extend='both')
cb = fig.colorbar(cs,ax=fig_ax3,shrink=0.9,ticks=[0.1,15,30],location='bottom')
cb.ax.set_xlabel('Duration (h)')
for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Bsyn_v,Bdeg_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'bsyn','bdeg')
X,Y = np.meshgrid(np.array(Bsyn_v),np.array(Bdeg_v))
G1_Grid[((E2F_Grid < 0.95) | (APC_Grid < 0.95)) & (Period_Grid != 0)] = -1
G1_Grid[G1_Grid != -1] = np.nan
cs = fig_ax1.contourf(X,Y,G1_Grid,levels=1,extend='both',colors='#cccccc')

SG2_Grid[((E2F_Grid < 0.95) | (APC_Grid < 0.95)) & (Period_Grid != 0)] = -1
SG2_Grid[SG2_Grid != -1] = np.nan
cs = fig_ax2.contourf(X,Y,SG2_Grid,levels=1,extend='both',colors='#cccccc')

M_Grid[((E2F_Grid < 0.95) | (APC_Grid < 0.95)) & (Period_Grid != 0)] = -1
M_Grid[M_Grid != -1] = np.nan
cs = fig_ax3.contourf(X,Y,M_Grid,levels=1,extend='both',colors='#cccccc')

# plt.savefig('Figure_Chain_Phase_Duration_Suppl.pdf', dpi=300) 



#%%
# =============================================================================
# CHAIN WITH FOUR SWITCHES
# =============================================================================
# Define some fixed parameter values (see Table 2 main text)
r = 0.5
n = 15
a_cdk = 5
a_apc = 5
a_e2f = 5
deld = 0.05
delb = 0.05
eps_apc = 100   # Inverse from defined in text
eps_cdk = 100
eps_e2f = 100

Kd = 120
Kb = 40
Kcdk = 20
  
dsyn = 0.15
ddeg = 0.009
bsyn = 0.03
bdeg = 0.003

fig = plt.figure(figsize=(8, 2.7*1.6),constrained_layout=True)
G = gridspec.GridSpec(2, 2, figure = fig,width_ratios=[0.75,1])

fig_ax1 = fig.add_subplot(G[0, 0])
fig_ax1.axis('off')
fig_ax1.annotate('A',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax2 = fig.add_subplot(G[0, 1])
fig_ax2.set_xlabel(r'Time (h)')
fig_ax2.set_ylabel('Concentration (a.u.)')
fig_ax2.annotate('B',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax3 = fig.add_subplot(G[1, 1])
fig_ax3.set_xlabel(r'Time (h)')
fig_ax3.set_ylabel('Concentration (a.u.)')


def bist_7d_chain(y,t,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb):
    CycD,E2F,CycA,FoxM1,CycB,Cdk1,Apc = y
    dy = [0,0,0,0,0,0,0] 

    # Define paramters for additional CycA swithc similar as the CycD switch
    a_fox = a_e2f
    eps_fox = eps_e2f
    asyn = dsyn
    adeg = ddeg 
    dela = deld
    Ka = Kd
    
    Xi_apc = 1 + a_apc*Apc*(Apc - 1)*(Apc - r)        
    Xi_cdk = 1 + a_cdk*(Cdk1/CycB)*((Cdk1/CycB) - 1)*((Cdk1/CycB) - r)        
    Xi_e2f = 1 + a_e2f*E2F*(E2F - 1)*(E2F - r)
    Xi_fox = 1 + a_fox*FoxM1*(FoxM1 - 1)*(FoxM1 - r)

    dy[0] = dsyn - ddeg*CycD*(Apc + deld)
    dy[1] = eps_e2f*(CycD**n/(CycD**n + (Kd*Xi_e2f)**n) - E2F)
    dy[2] = asyn*E2F - adeg*CycA*(Apc + dela)
    dy[3] = eps_fox*(CycA**n/(CycA**n + (Ka*Xi_fox)**n) - FoxM1)
    dy[4] = bsyn*FoxM1 - bdeg*CycB*(Apc + delb)
    dy[5] = eps_cdk*(CycB**(n+1)/(CycB**n + (Kb*Xi_cdk)**n) - Cdk1)
    dy[6] = eps_apc*(Cdk1**n/(Cdk1**n + (Kcdk*Xi_apc)**n) - Apc)
    
    return np.array(dy)


# TIME EVOLUTION 
# =============================================================================
t_start = 0
t_end = 12000
dt_max = 10
y0 = [1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12]

def f(t,y): return bist_7d_chain(y,t,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb) 
     
sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=dt_max,rtol=1e-6,atol=1e-9)

# Remove the transient solution
t_new = DBA.remove_transient(sol.y[0,:],sol.t,1)[1]
CycD_sol = sol.y[0,:][sol.t >= t_new[0]]
E2F_sol = sol.y[1,:][sol.t >= t_new[0]]
CycA_sol = sol.y[2,:][sol.t >= t_new[0]]
FoxM1_sol = sol.y[3,:][sol.t >= t_new[0]]
CycB_sol = sol.y[4,:][sol.t >= t_new[0]]
Cdk_sol = sol.y[5,:][sol.t >= t_new[0]]
Apc_sol = sol.y[6,:][sol.t >= t_new[0]]
t_new = (t_new - t_new[0])/60
    
# Normalize solutions for their maximal value
fig_ax2.plot(t_new,CycD_sol/max(CycD_sol),c=COL_v[0],label='[CycD]',alpha=1)
fig_ax2.plot(t_new,CycA_sol/max(CycA_sol),c=COL_v[4],label='[CycA]',alpha=1)
fig_ax2.plot(t_new,CycB_sol/max(CycB_sol),c=COL_v[8],label='[CycB]',alpha=1)
fig_ax2.plot(t_new,Cdk_sol/max(Cdk_sol),c='grey',label=r'$[Cdk]^*$',alpha=1)
fig_ax2.legend(bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=4,frameon=False)

fig_ax3.plot(t_new,E2F_sol,c=COL_v[2],label=r'$[E2F]^*$',alpha=1)
fig_ax3.plot(t_new,FoxM1_sol,c=COL_v[6],label=r'$[FoxM1]^*$',alpha=1)
fig_ax3.plot(t_new,Apc_sol,c=COL_v[10],label=r'$[Apc]^*$',alpha=1)
fig_ax3.legend(bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=3,frameon=False)

# plt.savefig('Figure_Chain_7D_Suppl.pdf', dpi=300) 



