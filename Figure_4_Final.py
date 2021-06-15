#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A modular approach for modeling the cell cycle based on functional response curves
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from scipy.integrate import solve_ivp
from datetime import datetime,timedelta
from scipy import optimize
import pandas as pd
import os
import Delay_Bistability_Analysis_Final as DBA

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

# Define code to perform numerical continuation
class Continuation(object):
    def __init__(self, func, dim):
        # func: class instance defining the equations of the system 
        # func must be initialized with parameter values
        # func must be callable, with inputs y = [u1,u2,...,u_dim] and bp, where 
        # bp is the current value of the bifurcation parameter
        self.func = func
        self.dim = dim

    def predict(self, h, x, v):
        # h [float]: increment for calculating the Jacobian
        # x [1D array]: vector containing the values of the variables u1,u2,...,u_dim, bp ON the response curve
        # v [1D array]: tangent vector (dim+1 components) at previous point ON the response curve
                        
        # Calculate Jacobian J at previous point x on response curve
        J = np.zeros((self.dim,self.dim+1))
        
        for i in range(self.dim):
            for j in range(self.dim+1):
                X = []
                for k in range(len(x)):
                    if k == j:
                        X.append(x[k]+h)
                    else:
                        X.append(x[k])
                J[i,j] = (self.func(X[0:-1],X[-1])[i] - self.func(x[0:-1],x[-1])[i])/h

        # Extend the Jacobian with the transpose of v        
        Jext = np.vstack((J,v))

        # Solve for tangent vector vn, and new predicted point NEARBY 
        # the steady state response curve       
        vn = np.linalg.solve(Jext,self.dim*[0]+[1])
        x_new = np.array(x) + self.ds*vn
        
        return x_new, vn
        
    def correct(self, x_0, v):
        # x_0 = [u1_0,u2_0,...,u_dim_0,bp_0]: coordinates of steady state point 
        # NEARBY the response curve 
        # v [1D array]: tangent vector at previous point NEARBY the response curve    
        x_0 = np.array(x_0)
        v = np.array(v)
        def tosolve(x):
            x = np.array(x)
            return self.func(x[0:-1],x[-1]) + [sum((x_0 - x)*v)]        
        # New point ON steady state response curve is returned
        x_cor = optimize.fsolve(tosolve,x_0,maxfev=1000)
        return x_cor

    def compute_responsecurve(self, ds, bp_start, bp_end, ds_min=1e-12, xstart=[], t_out=30):        
        # ds [float]: increment along tangent vector during continuation
        # bp_start [float]: start value of bifurcation parameter for continuation curve
        # bp_end [float]: end value of bifurcation parameter for continuation curve
        # xstart [1D list]: vector containing initial coordinates of u1,u2,...,u_dim at bp_start
        self.ds = ds
        
        if (len(xstart) == self.dim):
            xstart = xstart + [bp_start]

        # Initialize vectors for u1, u2, bp, v, and stability
        if bp_start < bp_end:
            vstart = self.dim*[0]+[1]  
        elif bp_start > bp_end:
            vstart = self.dim*[0]+[-1]
        vstart = vstart/np.linalg.norm(vstart)
        vv = [vstart]
        
        uv = []
        uv.append(xstart)
    
        # Create a list for storing the folds in the bifurcation curve
        folds = []
        
        # Define time to exit from while loop
        time = datetime.now()
        time_out = time + timedelta(seconds=t_out)
        while (min(bp_start,bp_end) <= uv[-1][-1] <= max(bp_start,bp_end)) and (datetime.now() < time_out):
            X,V = self.predict(1e-6,uv[-1],vv[-1])
            Xcor = self.correct(X, V)

            # Check if Xcor falls within a valid domain
            if (Xcor[-1] >= min(bp_start, bp_end)) and (Xcor[-1] <= max(bp_start, bp_end)) and ((-1e-6 <= np.array(self.func(Xcor[0:-1], Xcor[-1]))).all()) and ((1e-6 >= np.array(self.func(Xcor[0:-1], Xcor[-1]))).all()) and ((-1e-6 <= np.array(Xcor)).all()):
                uv.append(Xcor)
                vv.append(V)
                            
                # If the last component of v, i.e. bifurcation
                # parameter, changes sign, we have a fold
                if (vv[-1][-1]*vv[-2][-1] < 0):
                    folds.append(Xcor)
                
            # Decrease increment ds if no valid Xcor has been found                 
            elif np.abs(self.ds) >= ds_min:
                self.ds = self.ds/10                  
            
            else:
                break
               
        self.uv = np.array(uv)
        self.vv = np.array(vv)
        self.folds = np.array(folds)

        return(self)
    
# =============================================================================
# CODE TO GENERATE FIGURE 4 IN THE MAIN TEXT
# =============================================================================
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
fig_ax3.set_xlabel('[Cdk1] (nM)')
fig_ax3.set_ylabel(r'$[APC]^*$')
fig_ax3.annotate('C',(-0.27,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax4 = fig.add_subplot(G[1, 0])
fig_ax4.set_xlabel('Time t (min)')
fig_ax4.set_ylabel('[Cdk1] (nM)')
fig_ax4.annotate('D',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax5 = fig.add_subplot(G[1, 1])
fig_ax5.set_xlabel(r'$b_{syn}$ (nM/min)')
fig_ax5.set_ylabel(r'$b_{deg}$ (1/min)')
fig_ax5.set_title('Piecewise fit',weight='bold')
fig_ax5.annotate('E',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax6 = fig.add_subplot(G[1, 2])
fig_ax6.set_xlabel(r'$b_{syn}$ (nM/min)')
fig_ax6.set_ylabel(r'$b_{deg}$ (1/min)')
fig_ax6.set_title('Mass action model',weight='bold')
fig_ax6.annotate('F',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax7 = fig.add_subplot(G[2, 0])
fig_ax7.set_xlabel('[Cdk1] (nM)')
fig_ax7.set_ylabel(r'$[APC]^*$')
fig_ax7.annotate('G',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax8 = fig.add_subplot(G[2, 1])
fig_ax8.set_xlabel('Time t (min)')
fig_ax8.set_ylabel('[Cdk1] (nM)')
fig_ax8.annotate('H',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

fig_ax9 = fig.add_subplot(G[2, 2])
fig_ax9.set_xlabel('Time t (min)')
fig_ax9.set_ylabel('[Cdk1] (nM)')
fig_ax9.annotate('I',(-0.25,1),size=12, weight='bold',xycoords='axes fraction')

# COLOR SCHEME
# Define some colors to use in plots
COL_v = ['#471164ff','#472f7dff','#3e4c8aff','#32648eff','#297a8eff','#21918cff','#21a685ff','#3bbb75ff','#69cd5bff','#a5db36ff','#e2e418ff']

# =============================================================================
#     PARAMETER SET 1
# =============================================================================
# Choose parameter set for mass action model; Table 1 in the main text
params = {'CDKtot': 40,              
          'GWLtot': 40,            
          'PP2Atot': 40,           
          'ENSAtot': 200,           
          'n': 5,
          'kpe': 10**0.6,
          'kpg': 10**(-1.15),
          'kdg': 10**0.85,
          'kass1': 10**0.7,
          'kdis1': 10**1.45,
          'kcat1': 10**1.2,
          'kact': 10**(-0.2),
          'kinact': 10**0.2}

bsyn = 1
bdeg = 0.1

fig_ax5.scatter(bsyn,bdeg,s=3,c='k',zorder=3)
fig_ax6.scatter(bsyn,bdeg,s=3,c='k',zorder=3)


# MASS ACTION MODEL
#######################
# Time traces
t_start = 0
t_end = 1000 #500
y0 = [1e-6,1e-6,1e-6,1e-6,1e-6]    
def g(t,y): return DBA.PP2A_GWL_ENSA_Osc(y,t,bsyn,bdeg,params['kpg'],params['kdg'],params['kpe'],params['kass1'],params['kdis1'],params['kcat1'],params['kact'],params['kinact'],params['GWLtot'],params['ENSAtot'],params['PP2Atot']) 
sol = solve_ivp(g,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9)
Cdk1_sol = sol.y[0,:]
Apc_sol = sol.y[4,:]
t_v = sol.t
# Plot time traces in time domain; remove transient
t_new = DBA.remove_transient(Apc_sol,t_v,2)[1]
Cdk1_sol = sol.y[0,:][t_v >= t_new[0]]
Apc_sol = sol.y[4,:][t_v >= t_new[0]]
t_new = t_new - t_new[0]
fig_ax4.plot(t_new,Cdk1_sol,color='k',alpha=0.75,linestyle='-',label='Mass action',zorder=3)
# Plot time traces in phase plane
fig_ax3.plot(Cdk1_sol,Apc_sol,color='k',alpha=0.75,linestyle='-',label='Mass action',zorder=3) 
 

# Response curve
Model = DBA.PP2A_GWL_ENSA(**params)
g = Continuation(Model,3)
g.compute_responsecurve(0.05,0,40,xstart=[0,0,0],t_out=10)

PP2A = params['PP2Atot'] - g.uv[:,1] 
APC = g.uv[:,-1]/(g.uv[:,-1] + params['kinact']*PP2A/params['kact'])

fig_ax2.plot(g.uv[:,-1],APC,color='k',linestyle='--',alpha=0.5,label='Mass action',zorder=3) 
fig_ax2.set_xlim(0,params['CDKtot'])

fig_ax3.plot(g.uv[:,-1],APC,color='k',linestyle='--',alpha=0.5,label='Mass action',zorder=3) 
fig_ax3.set_xlim([16,26])
fig_ax3.set_ylim([0.15,0.8])


# PIECEWISE FIT
#######################
# Obtain parameters from mass action response curve and plot piecewise response
if len(g.folds) == 2:
    # Mass action model
    CDK_Folds = [i[-1] for i in g.folds]
    CDK_LF = np.min(CDK_Folds)
    CDK_RF = np.max(CDK_Folds)   
    # Define threshold K for piecewise fit in between both thresholds
    K = CDK_LF + (CDK_RF - CDK_LF)/2

    LFs = [i for i in g.folds if i[-1] == CDK_LF][0]
    RFs = [i for i in g.folds if i[-1] == CDK_RF][0]

    C1_LF = LFs[1]
    C1_RF = RFs[1]
    
    PP2A_LF = params['PP2Atot'] - C1_LF
    PP2A_RF = params['PP2Atot'] - C1_RF

    APC_LF = CDK_LF/(CDK_LF + params['kinact']*PP2A_LF/params['kact']) 
    APC_RF = CDK_RF/(CDK_RF + params['kinact']*PP2A_RF/params['kact']) 
 
    c = bsyn/(K*bdeg)
    fig_ax2.scatter(CDK_LF,APC_LF,s=5,c=COL_v[9])
    fig_ax2.scatter(CDK_RF,APC_RF,s=5,c=COL_v[9]) 
    
    # Calculate Xi values such that folds of piecewise linear coincide with folds of cubic
    # These values correspond to the values given in the Methods section
    lim = 0.95 # Set this parameter manually to adjust the upper branch of the response curve
    Xi_LF = (1/(APC_LF/(lim - APC_LF))**(1/params['n']))*CDK_LF/K
    Xi_RF = (1/(APC_RF/(lim - APC_RF))**(1/params['n']))*CDK_RF/K   
    # Define piecewise xi function
    X1 = np.linspace(0,APC_RF,1000)
    Xi_1 = (Xi_RF-1)/(APC_RF)*X1+1
    X2 = np.linspace(APC_RF,APC_LF,1000)
    Xi_2 = (Xi_LF-Xi_RF)/(APC_LF-APC_RF)*(X2 - APC_RF) + Xi_RF
    X3 = np.linspace(APC_LF,1-1e-12,1000)
    Xi_3 = (1-Xi_LF)/(1-APC_LF)*(X3 - APC_LF) + Xi_LF
    Xi_v = np.append(np.append(Xi_1,Xi_2),Xi_3)
    APC_v = np.append(np.append(X1,X2),X3)
    CDK_v = Xi_v*(APC_v/(lim - APC_v))**(1/params['n'])

    CDK_v[-1] = params['CDKtot']
    fig_ax2.plot(CDK_v*K,APC_v,color=COL_v[9],linestyle='--',alpha=0.75,label='Piecewise fit')
    fig_ax3.plot(CDK_v*K,APC_v,color=COL_v[9],linestyle='--',alpha=0.75,label='Piecewise fit')
    fig_ax2.legend(frameon=False)  

    #print(APC_LF,APC_RF,Xi_LF,Xi_RF)
    
# Time traces
eps = 500
t_start = 0
t_end = 50
y0 = [1e-6,1e-6]
def f(t,y): return DBA.bist_2d_piecewise_asymmetric(y,t,c,eps,params['n'],APC_RF,APC_LF,Xi_RF,Xi_LF,lim) 
sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9)
Cdk1_sol = sol.y[0,:]
Apc_sol = sol.y[1,:]
t_v = sol.t
# Plot time traces in time domain
t_new = DBA.remove_transient(Apc_sol,t_v,2)[1]
Cdk1_sol = sol.y[0,:][t_v >= t_new[0]]
Apc_sol = sol.y[1,:][t_v >= t_new[0]]
t_new = t_new - t_new[0]
fig_ax4.plot(t_new/bdeg,Cdk1_sol*K,color=COL_v[9],label='Piecewise fit, '+r'$\epsilon = $'+str(1/eps),alpha=1)
fig_ax4.set_ylim([10,26])
fig_ax4.set_xlim([0,200])
fig_ax4.legend(frameon=False)  
# Plot time traces in phase plane
fig_ax3.plot(Cdk1_sol*K,Apc_sol,color=COL_v[9],linestyle='--',zorder=3,label='Piecewise fit') 



# Load and plot the data from the csv file for Cdk1 oscillations
path = os.path.dirname(__file__)+'/Screens/Mass_Action/Final/' 
Array = pd.read_csv(path+'Screen_Mass_Action_Cdk1.csv',index_col=0)
X_v = Array['bsyn'].unique()
Y_v = Array['bdeg'].unique()

# Convert dataframe data to 2D array
Period_Module_Grid = np.array([np.array(Array.loc[Array['bdeg']==j,list(Array.columns)[3]]) for j in Y_v])
Amplitude_Module_Grid = np.array([np.array(Array.loc[Array['bdeg']==j,list(Array.columns)[4]]) for j in Y_v])
Period_MA_Grid = np.array([np.array(Array.loc[Array['bdeg']==j,list(Array.columns)[5]]) for j in Y_v])
Amplitude_MA_Grid = np.array([np.array(Array.loc[Array['bdeg']==j,list(Array.columns)[6]]) for j in Y_v])
           
X,Y = np.meshgrid(np.array(X_v),np.array(Y_v))
Period_Module_Grid[Period_Module_Grid == 0] = np.nan
Period_MA_Grid[Period_MA_Grid == 0] = np.nan

cs = fig_ax5.contourf(X,Y,Period_Module_Grid,levels=np.linspace(0,100,50),extend='max')
cb = fig.colorbar(cs,ax=fig_ax5,shrink=0.9,ticks=[0,50,100],location='bottom')
cb.ax.set_xlabel('Cdk1 Period (min)')
cb.ax.xaxis.set_label_position('top')

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)

cs = fig_ax6.contourf(X,Y,Period_MA_Grid,levels=np.linspace(0,100,50),extend='max')
cb = fig.colorbar(cs,ax=fig_ax6,shrink=0.9,ticks=[0,50,100],location='bottom')
cb.ax.set_xlabel('Cdk1 Period (min)')
cb.ax.xaxis.set_label_position('top')

for i in cs.collections:
    i.set_edgecolor('face')
cb.solids.set_rasterized(True)


# =============================================================================
#     PARAMETER SET 2
# =============================================================================
# Reduce all mass action rate constants to reduce time scale separation
params['kpe'] =  params['kpe']/16.5
params['kpg'] =  params['kpg']/16.5
params['kdg'] =  params['kdg']/16.5
params['kass1'] =  params['kass1']/16.5
params['kdis1'] =  params['kdis1']/16.5
params['kcat1'] =  params['kcat1']/16.5
params['kact'] =  params['kact']/16.5
params['kinact'] =  params['kinact']/16.5


# MASS ACTION MODEL
#######################
# Time traces
t_start = 0
t_end = 700
y0 = [1e-6,1e-6,1e-6,1e-6,1e-6]    
def g(t,y): return DBA.PP2A_GWL_ENSA_Osc(y,t,bsyn,bdeg,params['kpg'],params['kdg'],params['kpe'],params['kass1'],params['kdis1'],params['kcat1'],params['kact'],params['kinact'],params['GWLtot'],params['ENSAtot'],params['PP2Atot']) 
sol = solve_ivp(g,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9)
Cdk1_sol = sol.y[0,:]
Apc_sol = sol.y[4,:]
t_v = sol.t
# Plot time traces
t_new = DBA.remove_transient(Apc_sol,t_v,2)[1]
Cdk1_sol = sol.y[0,:][t_v >= t_new[0]]
Apc_sol = sol.y[4,:][t_v >= t_new[0]]
t_new = t_new - t_new[0]
fig_ax8.plot(t_new,Cdk1_sol,color='k',alpha=0.75,linestyle='-',label='Mass action',zorder=3)
fig_ax7.plot(Cdk1_sol,Apc_sol,color='k',alpha=0.75,linestyle='-',label='Mass action',zorder=3)

# Response curve
Model = DBA.PP2A_GWL_ENSA(**params)
g = Continuation(Model,3)
g.compute_responsecurve(0.05,0,40,xstart=[0,0,0],t_out=10)

PP2A = params['PP2Atot'] - g.uv[:,1] 
APC = g.uv[:,-1]/(g.uv[:,-1] + params['kinact']*PP2A/params['kact'])

fig_ax7.plot(g.uv[:,-1],APC,color='k',linestyle='--',alpha=0.5,label='Mass action',zorder=3) 
fig_ax7.set_xlim([16,26])
fig_ax7.set_ylim([0.15,0.8])

# PIECEWISE FIT
#######################
# As all parameters were scaled by the same factor, same response curve as parameter set 1 was obtained
# for mass action model
# Thus, also plot same piecewise fit
fig_ax7.plot(CDK_v*K,APC_v,color=COL_v[9],linestyle='--',alpha=0.75,label='Piecewise fit')

# Time traces
eps = 0.53
t_start = 0
t_end = 70
y0 = [1e-6,1e-6]
def f(t,y): return DBA.bist_2d_piecewise_asymmetric(y,t,c,eps,params['n'],APC_RF,APC_LF,Xi_RF,Xi_LF,lim) 
sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9)
Cdk1_sol = sol.y[0,:]
Apc_sol = sol.y[1,:]
t_v = sol.t

# Plot time traces
t_new = DBA.remove_transient(Apc_sol,t_v,2)[1]
Cdk1_sol = sol.y[0,:][t_v >= t_new[0]]
Apc_sol = sol.y[1,:][t_v >= t_new[0]]
t_new = t_new - t_new[0]
fig_ax8.plot(t_new/bdeg,Cdk1_sol*K,color=COL_v[9],label='Piecewise fit, '+r'$\epsilon$ = '+str(round(1/eps,2)),alpha=1)

fig_ax7.plot(Cdk1_sol*K,Apc_sol,color=COL_v[9],label='Piecewise fit, '+r'$\epsilon$ = '+str(round(1/eps,2)),alpha=1)

fig_ax8.set_ylim([10,26])
fig_ax8.set_xlim([0,400])
fig_ax8.legend(frameon=False)  



# =============================================================================
#     PARAMETER SET 3
# =============================================================================
# Divide parameterset by 6 to obtain original parameterset divided by ~100 (6*16.5)
params['kpe'] =  params['kpe']/6
params['kpg'] =  params['kpg']/6
params['kdg'] =  params['kdg']/6
params['kass1'] =  params['kass1']/6
params['kdis1'] =  params['kdis1']/6
params['kcat1'] =  params['kcat1']/6
params['kact'] =  params['kact']/6
params['kinact'] =  params['kinact']/6

# MASS ACTION MODEL
#######################
# Time traces
t_start = 0
t_end = 900
y0 = [1e-6,1e-6,1e-6,1e-6,1e-6]    
def g(t,y): return DBA.PP2A_GWL_ENSA_Osc(y,t,bsyn,bdeg,params['kpg'],params['kdg'],params['kpe'],params['kass1'],params['kdis1'],params['kcat1'],params['kact'],params['kinact'],params['GWLtot'],params['ENSAtot'],params['PP2Atot']) 
sol = solve_ivp(g,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9)
fig_ax9.plot(sol.t,sol.y[0,:],color='k',alpha=0.75,linestyle='-',label='Mass action',zorder=3)

# PIECEWISE FIT
#######################
# Again, same fit as before can be used
# Time traces
for eps,col in [[0.5,COL_v[2]],[0.1,COL_v[6]]]:
    t_start = 0
    t_end = 90
    y0 = [1e-6,1e-6]
    def f(t,y): return DBA.bist_2d_piecewise_asymmetric(y,t,c,eps,params['n'],APC_RF,APC_LF,Xi_RF,Xi_LF,lim) 
    sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9)
    # Plot time traces
    fig_ax9.plot(sol.t/bdeg,sol.y[0,:]*K,color=col,linestyle='-',label='Piecewise fit, '+r'$\epsilon$ = '+str(round(1/eps,2)),alpha=1)

fig_ax9.set_xlim([0,900])
fig_ax9.legend(frameon=False)  

# plt.savefig('Figure_4_Mass_Action.pdf', dpi=300) 
