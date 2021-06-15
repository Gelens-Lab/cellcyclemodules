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
import pandas as pd
import os
from scipy.integrate import solve_ivp

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


# =============================================================================
# CODE TO GENERATE FIGURE 8 IN THE MAIN TEXT
# Drawing was added afterwards in inkscape
# =============================================================================
# Define some fixed parameter values (see Table 2 main text)
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
  
dsyn = 0.15
ddeg = 0.009
bsyn = 0.03
bdeg = 0.003

fig = plt.figure(figsize=(8, 2.7*3),constrained_layout=True)
G = gridspec.GridSpec(3, 3, figure = fig)

fig_ax1 = fig.add_subplot(G[0, 0])
fig_ax1.set_xlabel(r'$d_{syn}$ (nM/min)')
fig_ax1.set_ylabel(r'Width $w_{cdk,apc}$')
fig_ax1.annotate('A',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')
fig_ax1.set_title('Compensation',fontweight='bold')

fig_ax2 = fig.add_subplot(G[0, 1])
fig_ax2.set_xlabel('Time t (h)')
fig_ax2.set_ylabel('Concentration (a.u.)')
fig_ax2.annotate('B',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')
fig_ax2.set_title('Restriction Point',fontweight='bold')

fig_ax3 = fig.add_subplot(G[0, 2])
fig_ax3.set_xlabel('Time t (h)')
fig_ax3.set_ylabel('Concentration (a.u.)')
fig_ax3.annotate('C',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')
fig_ax3.set_title('DNA Damage G1',fontweight='bold')

fig_ax4 = fig.add_subplot(G[1, 0])
fig_ax4.set_xlabel(r'Hours in G1 before damage')
fig_ax4.set_ylabel(r'Duration damage (h)')
fig_ax4.annotate('D',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax5 = fig.add_subplot(G[1, 1])
fig_ax5.set_xlabel('Time t (h)')
fig_ax5.set_ylabel('Concentration (a.u.)')
fig_ax5.annotate('E',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')
fig_ax5.set_title('DNA Damage G2',fontweight='bold')

fig_ax6 = fig.add_subplot(G[1, 2])
fig_ax6.set_xlabel(r'Hours in G2 before damage')
fig_ax6.set_ylabel(r'Duration damage (h)')
fig_ax6.annotate('F',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax7= fig.add_subplot(G[2, 0])
fig_ax7.axis('off')
fig_ax7.annotate('G',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax8 = fig.add_subplot(G[2, 1])
fig_ax8.set_xlabel(r'$\omega_{circadian}/\omega_{cdk}$')
fig_ax8.set_ylabel(r'Coupling strength $A_{cdk}$')
fig_ax8.annotate('H',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax9 = fig.add_subplot(G[2, 2])
fig_ax9.set_xlabel(r'Time (h)')
fig_ax9_right = fig_ax9.twinx()
fig_ax9.annotate('I',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

APC_col = COL_v[0]
CycD_col = COL_v[3]
CycB_col = COL_v[6]
E2F_col = COL_v[9]

# COMPENSATION
# =============================================================================
CSV = DBA.find_csv_chain('dsyn','acdk')
Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Dsyn_v,A_cdk_v,W_cyccdk_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'dsyn','acdk')

X,Y = np.meshgrid(np.array(Dsyn_v),np.array(W_cyccdk_v)*Kb)
Period_Grid = Period_Grid/60
cl = fig_ax1.contour(X,Y,Period_Grid,levels=[1e-3],colors='k',linewidths=1)
Period_Grid[Period_Grid == 0] = np.nan
cs = fig_ax1.contourf(X,Y,Period_Grid,levels=np.linspace(10,80,25),extend='both')
cb = fig.colorbar(cs,ax=fig_ax1,shrink=0.9,ticks=[10,45,80],location='bottom')
cb.ax.set_xlabel('Period (h)')

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

# Grey overlay for regions with irregular oscillations, i.e. where the average APC or E2F
# amplitude is smaller than 0.95
Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,Dsyn_v,A_cdk_v,W_cyccdk_v = DBA.csv_to_array_chain(CSV[1],CSV[0],'dsyn','acdk')
X,Y = np.meshgrid(np.array(Dsyn_v),np.array(W_cyccdk_v)*Kb)
Period_Grid[((E2F_Grid < 0.95) | (APC_Grid < 0.95)) & (Period_Grid != 0)] = -1
Period_Grid[Period_Grid != -1] = np.nan
cs = fig_ax1.contourf(X,Y,Period_Grid,levels=1,extend='both',colors='#cccccc')

fig_ax1.set_ylim([15,40])

fig_ax1.scatter(0.155,20,c='w',s=5)
fig_ax1.scatter(0.25,20,c='w',s=5)
fig_ax1.scatter(0.25,28,c='w',s=5)
fig_ax1.scatter(0.25,28,c='w',s=5)
fig_ax1.arrow(0.165,20,0.07,0,color='w',head_width=0.75,head_length=0.01,length_includes_head = True)
fig_ax1.arrow(0.25,21,0,6,color='w',head_width=0.0075,head_length=0.8,length_includes_head = True)


# TIME EVOLUTION FOR RESTRICTION POINT
# If system passed E2F bistable switch, CycD synthesis can be reduced and
# the cell will still complete the started round of division
# =============================================================================
# Calculations for unperturbed cells
t_start = 0
t_end = 10000
dt_max = 10
y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]

t_rp = t_end    # Time where CycD synthesis drops
t_g1 = t_end    # Time where CycD degradation increases
delta_t_g1 = 0  # Duration of DNA damage in G1
t_g2 = t_end    # Time where Cdk1 activation threshold increases
delta_t_g2 = 0  # Duration of DNA damage in G2
def f(t,y): return DBA.bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb) 

sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=dt_max,rtol=1e-6,atol=1e-9)

# Remove transient solution
t_new = DBA.remove_transient(sol.y[0,:],sol.t,2)[1]
CycD_sol = sol.y[0,:][sol.t >= t_new[0]]
E2F_sol = sol.y[1,:][sol.t >= t_new[0]]
CycB_sol = sol.y[2,:][sol.t >= t_new[0]]
Apc_sol = sol.y[4,:][sol.t >= t_new[0]] 
t_shift = (t_new - t_new[0])/60

fig_ax2.plot(t_shift,E2F_sol,c=E2F_col,alpha=1)
fig_ax2.plot(t_shift,Apc_sol,c=APC_col,alpha=1)
fig_ax2.plot(t_shift,CycB_sol/max(CycB_sol),c=CycB_col,alpha=1)

# Determine G2 phase
SG2 = t_new[(E2F_sol > 0.95) & (Apc_sol < 0.95)]
SG2 = np.split(SG2,np.where(SG2[1:]-SG2[0:-1] > 1.5*dt_max)[0]+1)

# Specify time where CycD synthesis drops based on G2 phase
t_rp = SG2[1][0]+(SG2[1][-1]-SG2[1][0])/5
t_g1 = t_end
delta_t_g1 = 0
t_g2 = t_end
delta_t_g2 = 0
def f(t,y): return DBA.bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb) 

sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=dt_max,rtol=1e-6,atol=1e-9)

# Remove transient solution
t_new_dam = DBA.remove_transient(sol.y[0,:],sol.t,2)[1]
E2F_sol_dam = sol.y[1,:][sol.t >= t_new_dam[0]]
CycB_sol_dam = sol.y[2,:][sol.t >= t_new_dam[0]]
Apc_sol_dam = sol.y[4,:][sol.t >= t_new_dam[0]]
t_shift_dam = (t_new_dam - t_new_dam[0])/60

baseline = 1.25
# Normalize concentrations for levels in unperturbed cells
fig_ax2.plot(t_shift_dam,baseline+E2F_sol_dam,c=E2F_col,label='$[E2F]^*$',alpha=1)
fig_ax2.plot(t_shift_dam,baseline+Apc_sol_dam,c=APC_col,label='$[APC]^*$',alpha=1)
fig_ax2.plot(t_shift_dam,baseline+CycB_sol_dam/max(CycB_sol),c=CycB_col,label='[CycB]',alpha=1)
fig_ax2.fill_between(np.linspace((t_rp - t_new[0])/60,t_shift[-1],100),baseline,baseline+1,color='k',alpha=0.1)
fig_ax2.legend(bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=2,frameon=False)

fig_ax2.set_yticks([0,0.5,1,baseline+0,baseline+0.5,baseline+1])
fig_ax2.set_yticklabels([0,0.5,1,0,0.5,1])


# TIME EVOLUTION FOR DNA DAMAGE IN G1
# =============================================================================
fig_ax3.plot(t_shift,CycD_sol/max(CycD_sol),c=CycD_col,alpha=1)
fig_ax3.plot(t_shift,Apc_sol,c=APC_col,alpha=1)

# Determine G1 phase
G1 = t_new[(E2F_sol < 0.95) & (Apc_sol < 0.95)]
G1 = np.split(G1,np.where(G1[1:]-G1[0:-1] > 1.5*dt_max)[0]+1)
t_g1 = G1[1][0]+(G1[1][-1]-G1[1][0])/2
delta_t_g1 = 15*60  # Duration of DNA damage in minutes
t_g2 = t_end
delta_t_g2 = 0
t_rp = t_end
def f(t,y): return DBA.bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb) 

sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=dt_max,rtol=1e-6,atol=1e-9)
t_new_dam = DBA.remove_transient(sol.y[0,:],sol.t,2)[1]
CycD_sol_dam = sol.y[0,:][sol.t >= t_new_dam[0]]
Apc_sol_dam = sol.y[4,:][sol.t >= t_new_dam[0]] 
t_shift_dam = (t_new_dam - t_new_dam[0])/60

baseline = 1.25
fig_ax3.plot(t_shift_dam,baseline+CycD_sol_dam/max(CycD_sol),c=CycD_col,label='[CycD]',alpha=1)
fig_ax3.plot(t_shift_dam,baseline+Apc_sol_dam,c=APC_col,label='$[APC]^*$',alpha=1)
fig_ax3.legend(bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=2,frameon=False)

fig_ax3.fill_between(np.linspace((t_g1 - t_new_dam[0])/60,(t_g1 - t_new_dam[0] + delta_t_g1)/60,100),baseline,max(baseline+CycD_sol_dam/max(CycD_sol)),color='k',alpha=0.1)

CycD_threshold = DBA.width_cubic(r,n,a_e2f)[2]*Kd/max(CycD_sol)
fig_ax3.hlines(baseline+CycD_threshold,0,((t_new_dam - t_new_dam[0])/60)[-1],colors=CycD_col,linestyle='--')

fig_ax3.set_yticks([0,0.5,1,baseline+0,baseline+0.5,baseline+1])
fig_ax3.set_yticklabels([0,0.5,1,0,0.5,1])


# DELAY IN M PHASE DUE TO DNA DAMAGE IN G1
# =============================================================================
CSV = DBA.find_csv_chain_dna_damage('G1')
Array,Delay_M_Grid,t_g1_v,delta_t_g1_v = DBA.csv_to_array_chain_dna_damage(CSV[1],CSV[0],'G1')
t_g1_v = t_g1_v/60
delta_t_g1_v = delta_t_g1_v/60

X,Y = np.meshgrid(np.array(t_g1_v)-t_g1_v[0],np.array(delta_t_g1_v))
# Duration of interphase = time between two M-phases
Delay_M_Grid = Delay_M_Grid/60
# Only plot elongation relative to no damage
Delay_M_Grid = Delay_M_Grid-Delay_M_Grid[0,0]

cs = fig_ax4.contourf(X,Y,Delay_M_Grid,levels=np.linspace(0,10,50),extend='both')

cb = fig.colorbar(cs,ax=fig_ax4,shrink=0.9,ticks=[0,5,10],location='bottom')
cb.ax.set_xlabel('Interphase elongation (h)')

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)



# TIME EVOLUTION FOR DNA DAMAGE IN G2
# =============================================================================
fig_ax5.plot(t_shift,CycB_sol/max(CycB_sol),c=CycB_col,alpha=1)
fig_ax5.plot(t_shift,Apc_sol,c=APC_col,alpha=1)

# Determine SG2 phase
t_rp = t_end
t_g1 = t_end
delta_t_g1 = 0
SG2 = t_new[(E2F_sol > 0.95) & (Apc_sol < 0.95)]
SG2 = np.split(SG2,np.where(SG2[1:]-SG2[0:-1] > 1.5*dt_max)[0]+1)
t_g2 = SG2[1][0]+(SG2[1][-1]-SG2[1][0])/2
delta_t_g2 = 15*60  # Duration of DNA damage in minutes
def f(t,y): return DBA.bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb) 

sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=dt_max,rtol=1e-6,atol=1e-9)
t_new_dam = DBA.remove_transient(sol.y[0,:],sol.t,2)[1]
CycB_sol_dam = sol.y[2,:][sol.t >= t_new_dam[0]]
Apc_sol_dam = sol.y[4,:][sol.t >= t_new_dam[0]] 
t_shift_dam = (t_new_dam - t_new_dam[0])/60

baseline = 1.25
fig_ax5.plot(t_shift_dam,baseline+CycB_sol_dam/max(CycB_sol),c=CycB_col,label='[CycB]',alpha=1)
fig_ax5.plot(t_shift_dam,baseline+Apc_sol_dam,c=APC_col,label='$[APC]^*$',alpha=1)
fig_ax5.legend(bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=2,frameon=False)

CycB_threshold = DBA.width_cubic(r,n,a_cdk)[2]*Kb/max(CycB_sol)
fig_ax5.hlines(baseline+CycB_threshold,0,(t_g2 - t_new[0])/60,colors=CycB_col,linestyle='--')
CycB_threshold = DBA.width_cubic(r,n,30)[2]*Kb/max(CycB_sol)    # If damage, parameter alpha is set to 30
fig_ax5.hlines(baseline+CycB_threshold,(t_g2 - t_new[0])/60,(t_g2 - t_new[0] + delta_t_g2)/60,colors=CycB_col,linestyle='--')
fig_ax5.fill_between(np.linspace((t_g2 - t_new[0])/60,(t_g2 - t_new[0] + delta_t_g2)/60),baseline,baseline+CycB_threshold,color='k',alpha=0.1)
CycB_threshold = DBA.width_cubic(r,n,a_cdk)[2]*Kb/max(CycB_sol)
fig_ax5.hlines(baseline+CycB_threshold,(t_g2 - t_new[0] + delta_t_g2)/60,t_shift[-1],colors=CycB_col,linestyle='--')

fig_ax5.set_yticks([0,0.5,1,baseline+0,baseline+0.5,baseline+1,baseline+1.5,baseline+2])
fig_ax5.set_yticklabels([0,0.5,1,0,0.5,1,1.5,2])


# DELAY IN M PHASE DUE TO DNA DAMAGE IN G2
# =============================================================================
CSV = DBA.find_csv_chain_dna_damage('G2')
Array,Delay_M_Grid,t_g2_v,delta_t_g2_v = DBA.csv_to_array_chain_dna_damage(CSV[1],CSV[0],'G2')
t_g2_v = t_g2_v/60
delta_t_g2_v = delta_t_g2_v/60

X,Y = np.meshgrid(np.array(t_g2_v)-t_g2_v[0],np.array(delta_t_g2_v))
# Duration of interphase = time between two M-phases
Delay_M_Grid = Delay_M_Grid/60
# Only plot elongation relative to no damage
Delay_M_Grid = Delay_M_Grid-Delay_M_Grid[0,0]

cs = fig_ax6.contourf(X,Y,Delay_M_Grid,levels=np.linspace(0,10,50),extend='both')

cb = fig.colorbar(cs,ax=fig_ax6,shrink=0.9,ticks=[0,5,10],location='bottom')
cb.ax.set_xlabel('Interphase elongation (h)')

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)


# =============================================================================
# ARNOLD TONGUES
# =============================================================================
# Determine unforced cell cycle period
t_start = 0
t_end = 30000
t_eval = np.linspace(t_start,t_end,t_end)
t_incr_max = 10
y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]
def f(t,y): return DBA.bist_5d_chain_sin(y,t,0,0,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)
sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=t_incr_max,rtol=1e-6,atol=1e-9)
period_0 = DBA.Oscillator_analysis(sol.y[3,:],sol.t)[-2]    # Natural period ~ 24h
if (type(period_0) != str) and (period_0 != 0):
    omega_0 = 2*np.pi/period_0

print(50*'-')
print('Natural Cdk1 period = '+str(period_0/60))
print(50*'-')

# Load screen data
CSV = DBA.find_csv_arnold()
Array,Locking_Grid,omega_v,A_a_v = DBA.csv_to_array_arnold(CSV[1],CSV[0])

# Plot Arnold Tongues
X,Y = np.meshgrid(np.array(omega_v),np.array(A_a_v))
Locking_Grid[Locking_Grid == 0] = np.nan
X,Y = np.meshgrid(np.array(omega_v)/omega_0,np.array(A_a_v))
cs = fig_ax8.contourf(X,Y,Locking_Grid)

for i in cs.collections:
    i.set_edgecolor('face')


### Plot time traces for different natural frequencies but constant forcing frequency
# Natural frequencies can be changed by altering bsyn and bdeg (here chosen from Fig 7F)
A_a = 8
omega = 0.00436  # Forcing period of circadian clock ~ 24h
period = 2*np.pi/omega

print(50*'-')
print('Forcing period = '+str(period/60))
print(50*'-')

y_ticks_v = []
for bsyn,bdeg,baseline,lbl in [[0.03,0.003,0,'i'],[0.02,0.002,1.25,'ii'],[0.04,0.004,2.5,'iii']]:
    t_start = 0
    t_end = 30000
    t_eval = np.linspace(t_start,t_end,t_end)
    dt_max = 10
    y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]

    def f(t,y): return DBA.bist_5d_chain_sin(y,t,0,0,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)
    sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=dt_max,rtol=1e-6,atol=1e-9)
    period_0 = DBA.Oscillator_analysis(sol.y[3,:],sol.t)[-2]
    if (type(period_0) != str) and (period_0 != 0):
        omega_0 = 2*np.pi/period_0
        print('Natural Cdk1 period = '+str(period_0/60))

    fig_ax8.scatter(period_0/period,A_a,s=5,c='w')
    if lbl == 'iii':
        fig_ax8.annotate(lbl,(period_0/period-0.1,6.5),c='w')
    else:
        fig_ax8.annotate(lbl,(period_0/period,6.5),c='w')
        
    def f(t,y): return DBA.bist_5d_chain_sin(y,t,omega,A_a,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)
    sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=dt_max,rtol=1e-6,atol=1e-9)
    t_v = sol.t
    Cdk_sol = sol.y[3,:][t_v > t_end//3]
    t_v = t_v[t_v > t_end//3]

    t_new = DBA.remove_transient(Cdk_sol,t_v,3)[1]
    Cdk_sol = Cdk_sol[t_v >= t_new[0]]/max(Cdk_sol[t_v >= t_new[0]]) 

    # Determine forced Cdk1 period
    dy_osc = Cdk_sol[1:]-Cdk_sol[0:-1]
    dy_osc[dy_osc == 0] = 1e-12     
    t_max = t_new[1:-1][(dy_osc[0:-1]*dy_osc[1:] < 0) & (dy_osc[0:-1] > 0)]
    if len(t_max) > 1:
        # Time differences between maxima
        dt_max = t_max[1:]-t_max[0:-1]
        p_err = 0.005
        period_osc = np.sum(DBA.pattern_recognition(dt_max,p_err))
        print('Forced Cdk1 period = '+str(period_osc/60))

    alpha_forcing = a_cdk + A_a + A_a*np.sin(omega*t_new)
    alpha_forcing = alpha_forcing/max(alpha_forcing)
    
    l, = fig_ax9.plot((t_new - t_new[0])/60,baseline+Cdk_sol,c=COL_v[5])
    l_right, = fig_ax9_right.plot((t_new - t_new[0])/60,baseline + alpha_forcing,c=COL_v[9])

    y_ticks_v.append(baseline)
    y_ticks_v.append(baseline+1)

fig_ax9.set_ylabel('[Cdk1] (a.u.)',c=l.get_color())
fig_ax9_right.set_ylabel(r'Circadian rhythm (a.u.)',c=l_right.get_color())
fig_ax9.set_yticks(y_ticks_v)
fig_ax9.set_yticklabels([0,1,0,1,0,1])
for i,an in [[0.5,'i'],[1.7,'ii'],[2.85,'iii']]:
    fig_ax9.annotate(an,(-18,i),annotation_clip=False)
fig_ax9.set_title(r'$\omega_{circadian}$ = $2\pi/$'+str(int(round(period/60,0)))+r' $h^{-1}$')

fig_ax9_right.set_yticks(y_ticks_v)
fig_ax9_right.set_yticks(y_ticks_v)
fig_ax9_right.set_yticklabels([0,1,0,1,0,1])
fig_ax9.set_xlim([0,150])
fig_ax9.set_ylim([-0.05,3.55])
fig_ax9_right.set_ylim([-0.05,3.55])

# plt.savefig('Figure_8_Control.pdf', dpi=300) 


#%%
# =============================================================================
# SUPPLEMENTAL FIGURE RELATED TO FIGURE 8
# =============================================================================
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
  
dsyn = 0.15
ddeg = 0.009
bsyn = 0.03
bdeg = 0.003

fig = plt.figure(figsize=(10, 2.7*3),constrained_layout=True)
G = gridspec.GridSpec(3, 3, figure = fig)

fig_ax1 = fig.add_subplot(G[0, 0])
fig_ax1.set_xlabel(r'$\omega_{circadian}/\omega_{cdk}$')
fig_ax1.set_ylabel(r'Coupling strength $A_{cdk}$')
fig_ax1.annotate('A',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax2 = fig.add_subplot(G[0, 1])
fig_ax2.set_xlabel('Time t (h)')
fig_ax2.set_ylabel('Concentration (a.u.)')
fig_ax2.annotate('B',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax3 = fig.add_subplot(G[0, 2])
fig_ax3.set_xlabel('Time t (h)')
fig_ax3.set_ylabel('Concentration (a.u.)')
fig_ax3.annotate('C',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax4 = fig.add_subplot(G[1, 0])
fig_ax4.set_xlabel('Time t (h)')
fig_ax4.set_ylabel('Concentration (a.u.)')
fig_ax4.annotate('D',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax5 = fig.add_subplot(G[1, 1])
fig_ax5.set_xlabel('Time t (h)')
fig_ax5.set_ylabel('Concentration (a.u.)')
fig_ax5.annotate('E',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax6 = fig.add_subplot(G[1, 2])
fig_ax6.set_xlabel('Time t (h)')
fig_ax6.set_ylabel('Concentration (a.u.)')
fig_ax6.annotate('F',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax7 = fig.add_subplot(G[2, 0])
fig_ax7.set_xlabel('Time t (h)')
fig_ax7.set_ylabel('Concentration (a.u.)')
fig_ax7.annotate('G',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax8 = fig.add_subplot(G[2, 1])
fig_ax8.set_xlabel('Time t (h)')
fig_ax8.set_ylabel('Concentration (a.u.)')
fig_ax8.annotate('H',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax9 = fig.add_subplot(G[2, 2])
fig_ax9.set_xlabel('Time t (h)')
fig_ax9.set_ylabel('Concentration (a.u.)')
fig_ax9.annotate('I',(-0.35,1),size=12, weight='bold',xycoords='axes fraction')

pmax = 5
pq_v = []
for p in range(1,pmax+1):
    for q in range(1,pmax+1):
        if p/q not in [j[1] for j in pq_v]:
            pq_v.append([str(p)+'/'+str(q),p/q])
            
t_start = 0
t_end = 15000
t_eval = np.linspace(t_start,t_end,t_end)
dt_max = 10
y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]
def f(t,y): return DBA.bist_5d_chain_sin(y,t,0,0,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)
sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=dt_max,rtol=1e-6,atol=1e-9)
t_v = sol.t
Cdk_sol = sol.y[3,:]
period_0 = DBA.Oscillator_analysis(Cdk_sol,t_v)[-2]    # ~24h
if (type(period_0) != str) and (period_0 != 0):
    omega_0 = 2*np.pi/period_0


# Load screen data
CSV = DBA.find_csv_arnold()
Array,Locking_Grid,omega_v,A_a_v = DBA.csv_to_array_arnold(CSV[1],CSV[0])
# Plot Arnold Tongues
X,Y = np.meshgrid(np.array(omega_v),np.array(A_a_v))
Locking_Grid[Locking_Grid == 0] = np.nan
X,Y = np.meshgrid(np.array(omega_v)/omega_0,np.array(A_a_v))
cs = fig_ax1.contourf(X,Y,Locking_Grid)
for i in cs.collections:
    i.set_edgecolor('face')
    
    
for ratio,A,an,ax in [[0.5,7,'B',fig_ax2],[0.5,2,'C',fig_ax3],[1.1,7,'D',fig_ax4],[1.1,2,'E',fig_ax5],[1.65,2,'F',fig_ax6],[1.65,7,'G',fig_ax7],[1.8,7,'H',fig_ax8],[2.5,7,'I',fig_ax9]]:

        fig_ax1.scatter(ratio,A,s=3,c='k')
        fig_ax1.annotate(an,(ratio+0.05,A+0.1),c='k')
        omega = ratio*omega_0
        period = 2*np.pi/omega
        
        eps_apc = 100
        eps_cdk = 100
        eps_e2f = 100
    
        t_start = 0
        t_end = 30000
        t_incr_max = 10
        t_eval = np.linspace(t_start,t_end,10*t_end)
        y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]
        
        def f(t,y): return DBA.bist_5d_chain_sin(y,t,omega,A,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)
        sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=t_incr_max,rtol=1e-6,atol=1e-9)
        t_v = sol.t
        Cdk_sol = sol.y[3,:]/max(sol.y[3,:])
        alpha_forcing = (a_cdk + A + A*np.sin(omega*t_v))/max(a_cdk + A + A*np.sin(omega*t_v))
        ax.plot(t_v/60,alpha_forcing,label='Circadian',color=COL_v[9])
        ax.plot(t_v/60,Cdk_sol,label='Cdk1',color=COL_v[5])   
        ax.set_xlim([100,t_v[-1]/60])
        
        y_osc = np.array(Cdk_sol[t_v > t_end//3])
        t_osc = np.array(t_v[t_v > t_end//3])
        # Calculate derivative of solution
        dy_osc = y_osc[1:]-y_osc[0:-1]
        dy_osc[dy_osc == 0] = 1e-12
        # Extrema are located where product of two neighbouring elements in dy is negative, i.e.
        # sign change
        t_max = t_osc[1:-1][(dy_osc[0:-1]*dy_osc[1:] < 0) & (dy_osc[0:-1] > 0)]
        if len(t_max) > 1:
            # Time differences between maxima
            dt_max = t_max[1:]-t_max[0:-1]

            p_err = 0.005
            period_osc = np.sum(DBA.pattern_recognition(dt_max,p_err))
            if (period_osc != 0):
                for i in pq_v:
                    if (abs(i[1] - period_osc/period) < p_err):
                        ax.set_title(i[0][-1]+':'+i[0][0],fontweight='bold')
                        break
 
fig_ax8.legend(bbox_to_anchor=(0, -0.3, 1, 0),loc='upper center',ncol=2,frameon=False)   
