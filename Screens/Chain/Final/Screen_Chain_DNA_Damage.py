# =============================================================================
# SCRIPT TO RUN PARAMETER SCANS FOR DNA DAMAGE
# Related to Figure 8 in the main text
# =============================================================================
from __future__ import division
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.integrate import solve_ivp

# Function to find period and amplitudes of oscillations
def Oscillator_analysis(y_osc,t_osc,n_max=6,tol=0.005):
    # n_max: number of maxima needed to validate oscillations
    y_osc = np.array(y_osc)
    t_osc = np.array(t_osc)
    # Calculate derivative of solution
    dy_osc = y_osc[1:]-y_osc[0:-1]
    # Replace zeros by small positive number to prevent the product of neighbouring
    # elements to equal zero, which would impose a problem for subsequent analysis
    # of finding the extrema
    dy_osc[dy_osc == 0] = 1e-12
    # Minimal number of required extrema = 2*(number of maxima)
    min_extr = 2*n_max
    # Extrema are located where product of two neighbouring elements in dy is 
    # negative, i.e. sign change
    if len(y_osc[1:-1][(dy_osc[0:-1]*dy_osc[1:] < 0)]) >= min_extr:
        # Maximum if sign change for derivative from + to - 
        y_max = y_osc[1:-1][(dy_osc[0:-1]*dy_osc[1:] < 0) & (dy_osc[0:-1] > 0)]
        # Minimum if sign change for derivative from - to + 
        y_min = y_osc[1:-1][(dy_osc[0:-1]*dy_osc[1:] < 0) & (dy_osc[0:-1] < 0)]
 
        t_max = t_osc[1:-1][(dy_osc[0:-1]*dy_osc[1:] < 0) & (dy_osc[0:-1] > 0)]
        t_min = t_osc[1:-1][(dy_osc[0:-1]*dy_osc[1:] < 0) & (dy_osc[0:-1] < 0)]
        
        amplitudes = np.array(abs(y_max[0:min(len(y_max),len(y_min))] - y_min[0:min(len(y_max),len(y_min))]))
        # Only consider extrema if amplitudes are larger than tol
        y_max = y_max[0:len(amplitudes)][amplitudes > tol]
        y_min = y_min[0:len(amplitudes)][amplitudes > tol]
        t_max = t_max[0:len(amplitudes)][amplitudes > tol]
        t_min = t_min[0:len(amplitudes)][amplitudes > tol]
        amplitudes = amplitudes[amplitudes > tol]
        
        # Damped oscillations if each amplitude is decreased by 100*tol% compared to the previous one
        if np.all(amplitudes[1:] < (amplitudes[0:-1]-tol*amplitudes[0:-1])):
            amplitude = 0
            period = 0
            out = y_max,y_min,t_max,t_min,period,amplitude
        else:
            amplitude = np.median(amplitudes)   # Use the median to neglect the effect of potential outliers
            period = np.median(list(abs(t_max[1:]-t_max[0:-1])) + list(abs(t_min[1:]-t_min[0:-1])))
            out = y_max,y_min,t_max,t_min,period,amplitude          
    else:
        out = 'Oscillator could not be validated; not enough or no oscillations in the given time frame'
        
    return(out)


# =============================================================================
# DEFINE SYSTEM EQUATIONS FOR INTERLINKED SWITCHES
# No non-dimensionalization was performed here
# =============================================================================
def bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb):
    CycD,E2F,CycB,Cdk1,Apc = y
    dy = [0,0,0,0,0] 
    
    if t < t_rp:
        dsyn = dsyn
    else:
        dsyn = 0.1*dsyn

    if t_g1 < t < t_g1 + delta_t_g1:
        deld = 3*deld
    else:
        deld = deld
        
    if t_g2 < t < t_g2 + delta_t_g2:
        a_cdk = 30
    else:
        a_cdk = a_cdk
        
    Xi_apc = 1 + a_apc*Apc*(Apc - 1)*(Apc - r)        
    Xi_cdk = 1 + a_cdk*(Cdk1/CycB)*((Cdk1/CycB) - 1)*((Cdk1/CycB) - r)        
    Xi_e2f = 1 + a_e2f*E2F*(E2F - 1)*(E2F - r)

    dy[0] = dsyn - ddeg*CycD*(Apc + deld)
    dy[1] = eps_e2f*(CycD**n/(CycD**n + (Kd*Xi_e2f)**n) - E2F)
    dy[2] = bsyn*E2F - bdeg*CycB*(Apc + delb)
    dy[3] = eps_cdk*(CycB**(n+1)/(CycB**n + (Kb*Xi_cdk)**n) - Cdk1)
    dy[4] = eps_apc*(Cdk1**n/(Cdk1**n + (Kcdk*Xi_apc)**n) - Apc)
    
    return np.array(dy)


# =============================================================================
# PERFORM THE PARAMETER SCAN
# DNA damage in G1 phase
# =============================================================================
# Standard set of parameter values (see table 1 in main text)
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

t_start = 0
t_end = 20000
t_eval = np.linspace(t_start,t_end,t_end+1)

dt_max = 10
y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]

# First calculate cell cycle phases for unperturbed  cell cycle
t_rp = t_end
t_g1 = t_end
t_g2 = t_end
delta_t_g1 = 0
delta_t_g2 = 0

def f(t,y): return bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)

sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=dt_max,rtol=1e-6,atol=1e-9)

CycD_sol = sol.y[0,:]
E2F_sol = sol.y[1,:]
CycB_sol = sol.y[2,:]
Cdk_sol = sol.y[3,:]
Apc_sol = sol.y[4,:]
t_v = sol.t
 
# Determine different cell cycle phases       
out_cyc = Oscillator_analysis(CycB_sol,t_v,n_max=4)  
if (type(out_cyc) != str) and (out_cyc[-1] != 0):          
    G1 = t_v[(E2F_sol < 0.95) & (Apc_sol < 0.95)]
    SG2 = t_v[(E2F_sol > 0.95) & (Apc_sol < 0.95)]
    M = t_v[Apc_sol > 0.95]
            
    if len(G1) > 0:
        G1 = np.split(G1,np.where(G1[1:]-G1[0:-1] > 1.5*dt_max)[0]+1)
    if len(SG2) > 0:
        SG2 = np.split(SG2,np.where(SG2[1:]-SG2[0:-1] > 1.5*dt_max)[0]+1)
    if len(M) > 0:
        M = np.split(M,np.where(M[1:]-M[0:-1] > 1.5*dt_max)[0]+1)
    
    # Determine time points of G1 phase for second cycle (i.e. G1[1])
    if (len(G1) > 1):
        t_g1_v = np.linspace(G1[1][0],G1[1][-1],50)
        delta_t_g1_v = np.linspace(0,500,50)

Array = np.zeros((len(t_g1_v)*len(delta_t_g1_v),4))
Array = pd.DataFrame(Array,columns=['Meta','t_g1','delta_t_g1','Delay_M'])
    
i_row = 0
for t_g1 in t_g1_v:
    for delta_t_g1 in delta_t_g1_v:

        Array.iloc[i_row,1:3] = t_g1,delta_t_g1
                
        print(i_row)
                
        def f(t,y): return bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)
     
        sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=dt_max,rtol=1e-6,atol=1e-9)
        
        CycD_sol = sol.y[0,:]
        E2F_sol = sol.y[1,:]
        CycB_sol = sol.y[2,:]
        Cdk_sol = sol.y[3,:]
        Apc_sol = sol.y[4,:]
        t_v = sol.t

        out_cyc = Oscillator_analysis(CycB_sol,t_v,n_max=4)  
        if (type(out_cyc) != str) and (out_cyc[-1] != 0): 
            M = t_v[Apc_sol > 0.95]  
            if len(M) > 0:
                M = np.split(M,np.where(M[1:]-M[0:-1] > 1.5*dt_max)[0]+1)           
                if (len(M) > 1):
                    Array.iloc[i_row,-1] = M[1][0]-M[0][0]
                
        i_row = i_row + 1 
                  
for ar in [Array]:
    ar.loc[0,'Meta'] = 'n = '+str(n)
    ar.loc[1,'Meta'] = 'r = '+str(r)
    ar.loc[2,'Meta'] = 'a_apc = '+str(a_apc)
    ar.loc[3,'Meta'] = 'a_cdk = '+str(a_cdk)
    ar.loc[4,'Meta'] = 'a_e2f = '+str(a_e2f)
    ar.loc[5,'Meta'] = 'deld = '+str(deld)
    ar.loc[6,'Meta'] = 'delb = '+str(delb)
    ar.loc[7,'Meta'] = 'eps_apc = '+str(eps_apc)
    ar.loc[8,'Meta'] = 'eps_cdk = '+str(eps_cdk)
    ar.loc[9,'Meta'] = 'eps_e2f = '+str(eps_e2f)
    ar.loc[10,'Meta'] = 'Kd = '+str(Kd)
    ar.loc[11,'Meta'] = 'Kb = '+str(Kb)
    ar.loc[12,'Meta'] = 'Kcdk = '+str(Kcdk)
    ar.loc[13,'Meta'] = 'dsyn = '+str(dsyn)
    ar.loc[14,'Meta'] = 'ddeg = '+str(ddeg)
    ar.loc[15,'Meta'] = 'bsyn = '+str(bsyn)
    ar.loc[16,'Meta'] = 'bdeg = '+str(bdeg)
    ar.loc[17,'Meta'] = 'tstart = '+str(t_start)
    ar.loc[18,'Meta'] = 'tend = '+str(t_end)
    ar.loc[19,'Meta'] = 'dt_max = '+str(dt_max)
               
time = datetime.now()
timestr = time.strftime("%d%m%Y-%H%M%S%f")
filename = 'Screen_Chain_DNA_Damage_G1_'+str(timestr)+'.csv'
Array.to_csv(filename)

  
#%%
# =============================================================================
# PERFORM THE PARAMETER SCAN
# DNA damage in G2 phase
# =============================================================================
# Standard set of parameter values (see table 1 in main text)
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

t_start = 0
t_end = 20000
t_eval = np.linspace(t_start,t_end,t_end+1)

dt_max = 10
y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]

# First calculate cell cycle phases for unperturbed  cell cycle
t_rp = t_end
t_g1 = t_end
t_g2 = t_end
delta_t_g1 = 0
delta_t_g2 = 0

def f(t,y): return bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)

sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=dt_max,rtol=1e-6,atol=1e-9)

CycD_sol = sol.y[0,:]
E2F_sol = sol.y[1,:]
CycB_sol = sol.y[2,:]
Cdk_sol = sol.y[3,:]
Apc_sol = sol.y[4,:]
t_v = sol.t
     
# Determine different cell cycle phases          
out_cyc = Oscillator_analysis(CycB_sol,t_v,n_max=4)  
if (type(out_cyc) != str) and (out_cyc[-1] != 0):          
    G1 = t_v[(E2F_sol < 0.95) & (Apc_sol < 0.95)]
    SG2 = t_v[(E2F_sol > 0.95) & (Apc_sol < 0.95)]
    M = t_v[Apc_sol > 0.95]
            
    if len(G1) > 0:
        G1 = np.split(G1,np.where(G1[1:]-G1[0:-1] > 1.5*dt_max)[0]+1)
    if len(SG2) > 0:
        SG2 = np.split(SG2,np.where(SG2[1:]-SG2[0:-1] > 1.5*dt_max)[0]+1)
    if len(M) > 0:
        M = np.split(M,np.where(M[1:]-M[0:-1] > 1.5*dt_max)[0]+1)
 
    # Determine time points of G2 phase for second cycle (i.e. G2[1])
    if (len(SG2) > 1):
        t_g2_v = np.linspace(SG2[1][0],SG2[1][-1],50)
        delta_t_g2_v = np.linspace(0,500,50)

Array = np.zeros((len(t_g2_v)*len(delta_t_g2_v),4))
Array = pd.DataFrame(Array,columns=['Meta','t_g2','delta_t_g2','Delay_M'])
    
i_row = 0
for t_g2 in t_g2_v:
    for delta_t_g2 in delta_t_g2_v:

        Array.iloc[i_row,1:3] = t_g2,delta_t_g2
                
        print(i_row)
                
        def f(t,y): return bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)
     
        sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=dt_max,rtol=1e-6,atol=1e-9)
        
        CycD_sol = sol.y[0,:]
        E2F_sol = sol.y[1,:]
        CycB_sol = sol.y[2,:]
        Cdk_sol = sol.y[3,:]
        Apc_sol = sol.y[4,:]
        t_v = sol.t

        out_cyc = Oscillator_analysis(CycB_sol,t_v,n_max=4)  
        if (type(out_cyc) != str) and (out_cyc[-1] != 0): 
            M = t_v[Apc_sol > 0.95]  
            if len(M) > 0:
                M = np.split(M,np.where(M[1:]-M[0:-1] > 1.5*dt_max)[0]+1)           
                if (len(M) > 1):
                    Array.iloc[i_row,-1] = M[1][0]-M[0][0]
                
        i_row = i_row + 1 
 
for ar in [Array]:
    ar.loc[0,'Meta'] = 'n = '+str(n)
    ar.loc[1,'Meta'] = 'r = '+str(r)
    ar.loc[2,'Meta'] = 'a_apc = '+str(a_apc)
    ar.loc[3,'Meta'] = 'a_cdk = '+str(a_cdk)
    ar.loc[4,'Meta'] = 'a_e2f = '+str(a_e2f)
    ar.loc[5,'Meta'] = 'deld = '+str(deld)
    ar.loc[6,'Meta'] = 'delb = '+str(delb)
    ar.loc[7,'Meta'] = 'eps_apc = '+str(eps_apc)
    ar.loc[8,'Meta'] = 'eps_cdk = '+str(eps_cdk)
    ar.loc[9,'Meta'] = 'eps_e2f = '+str(eps_e2f)
    ar.loc[10,'Meta'] = 'Kd = '+str(Kd)
    ar.loc[11,'Meta'] = 'Kb = '+str(Kb)
    ar.loc[12,'Meta'] = 'Kcdk = '+str(Kcdk)
    ar.loc[13,'Meta'] = 'dsyn = '+str(dsyn)
    ar.loc[14,'Meta'] = 'ddeg = '+str(ddeg)
    ar.loc[15,'Meta'] = 'bsyn = '+str(bsyn)
    ar.loc[16,'Meta'] = 'bdeg = '+str(bdeg)
    ar.loc[17,'Meta'] = 'tstart = '+str(t_start)
    ar.loc[18,'Meta'] = 'tend = '+str(t_end)
    ar.loc[19,'Meta'] = 'dt_max = '+str(dt_max)
               
time = datetime.now()
timestr = time.strftime("%d%m%Y-%H%M%S%f")
filename = 'Screen_Chain_DNA_Damage_G2_'+str(timestr)+'.csv'
Array.to_csv(filename)

  