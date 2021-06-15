# =============================================================================
# SCRIPT TO RUN PARAMETER SCANS FOR CHAIN OF INTERLINKED CUBIC SWITCHES
# Related to Figure 7 in the main text
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
            amplitude_mean = 0
            period = 0
            out = y_max,y_min,t_max,t_min,period,amplitude_mean,amplitude
        else:
            amplitude = np.median(amplitudes)
            # Here, also include the average amplitude to identify oscillations with irregular amplitudes (Grey areas in Fig7)
            amplitude_mean = np.mean(amplitudes)     
            period = np.median(list(abs(t_max[1:]-t_max[0:-1])) + list(abs(t_min[1:]-t_min[0:-1])))
            out = y_max,y_min,t_max,t_min,period,amplitude_mean,amplitude            
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
    
    if t < t_rp:    # t_rp = time from which CycD synthesis drops (i.e. to test for passage of the restriction point)
        dsyn = dsyn
    else:
        dsyn = 0.1*dsyn

    if t_g1 < t < t_g1 + delta_t_g1:    # t_g1 = time from which CycD degradation increases due to DNA damage in G1
        deld = 3*deld
    else:
        deld = deld
        
    if t_g2 < t < t_g2 + delta_t_g2:    # t_g2 = time from which Cdk activation threshold shifts to higher CycB levels due to DNA damage in G2
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
# Synthesis and degradation rate of CycD
# =============================================================================
# Standard set of parameter values (see table 1 in main text)
r = 0.5
n = 15
a_cdk = 5
a_apc = 5
a_e2f = 5
deld = 0.05
delb = 0.05
eps_apc = 100   # Note how eps is defined as the inverse of eps in the text
eps_cdk = 100
eps_e2f = 100

Kd = 120    # Kcycde2f
Kb = 40     # Kcycbcdk
Kcdk = 20   # Kcdkapc

bsyn = 0.03
bdeg = 0.003

t_start = 0
t_end = 20000
dt_max = 10
y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]

dsyn_v = np.linspace(0.05,0.3,100)
ddeg_v = np.linspace(0.005,0.015,100)

Array = np.zeros((len(dsyn_v)*len(ddeg_v),9))
Array = pd.DataFrame(Array,columns=['Meta','dsyn','ddeg','Period','G1','SG2','M','E2F Amplitude','APC Amplitude'])
    
i_row = 0
for dsyn in dsyn_v:
    for ddeg in ddeg_v:

        Array.iloc[i_row,1:3] = dsyn,ddeg
                
        print(i_row)

        t_rp = t_end    # CycD synthesis rate always constant at standard level; no effect of the RP is tested
        t_g1 = t_end    # CycD degradation rate always constant at standard level; no effect of DNA damage
        t_g2 = t_end    # Cdk activation threshold always constant at standard level; no effect of DNA damage
        delta_t_g1 = 0
        delta_t_g2 = 0
                
        def f(t,y): return bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)
     
        sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=dt_max,rtol=1e-6,atol=1e-9)
        
        CycD_sol = sol.y[0,:]
        E2F_sol = sol.y[1,:]
        CycB_sol = sol.y[2,:]
        Cdk_sol = sol.y[3,:]
        Apc_sol = sol.y[4,:]
        t_v = sol.t

        # Store average E2F and APC amplitudes
        out_e2f = Oscillator_analysis(E2F_sol,t_v,n_max=3)
        if (type(out_e2f) != str) and (out_e2f[-1] != 0):
            Array.iloc[i_row,7] = out_e2f[-2]

        out_apc = Oscillator_analysis(Apc_sol,t_v,n_max=3)
        if (type(out_apc) != str) and (out_apc[-1] != 0):
            Array.iloc[i_row,8] = out_apc[-2]
 
        # Store period of the oscillator (here based CycB)
        out_cyc = Oscillator_analysis(CycB_sol,t_v,n_max=3)  
        if (type(out_cyc) != str) and (out_cyc[-1] != 0):
            Array.iloc[i_row,3] = out_cyc[-3]
 
            # Determine different cell cycle phases
            G1 = t_v[(E2F_sol < 0.95) & (Apc_sol < 0.95)]
            SG2 = t_v[(E2F_sol > 0.95) & (Apc_sol < 0.95)]
            M = t_v[Apc_sol > 0.95]
        
            if len(G1) > 0:
                G1 = np.split(G1,np.where(G1[1:]-G1[0:-1] > 1.5*dt_max)[0]+1)
                G1_length = np.median([i[-1]-i[0] for i in G1]) 
                Array.iloc[i_row,4] = G1_length
            if len(SG2) > 0:
                SG2 = np.split(SG2,np.where(SG2[1:]-SG2[0:-1] > 1.5*dt_max)[0]+1)
                SG2_length = np.median([i[-1]-i[0] for i in SG2]) 
                Array.iloc[i_row,5] = SG2_length
            if len(M) > 0:
                M = np.split(M,np.where(M[1:]-M[0:-1] > 1.5*dt_max)[0]+1)
                M_length = np.median([i[-1]-i[0] for i in M]) 
                Array.iloc[i_row,6] = M_length
     
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
    ar.loc[13,'Meta'] = 'bsyn = '+str(bsyn)
    ar.loc[14,'Meta'] = 'bdeg = '+str(bdeg)
    ar.loc[15,'Meta'] = 'tstart = '+str(t_start)
    ar.loc[16,'Meta'] = 'tend = '+str(t_end)
    ar.loc[17,'Meta'] = 'dt_max = '+str(dt_max)
               
time = datetime.now()
timestr = time.strftime("%d%m%Y-%H%M%S%f")
filename = 'Screen_Chain_dsyn_ddeg_'+str(timestr)+'.csv'
Array.to_csv(filename)


#%%
# =============================================================================
# PERFORM THE PARAMETER SCAN
# Synthesis and degradation rate of CycB
# =============================================================================
# Standard set of parameter values (see table 1 in main text)
r = 0.5
n = 15
a_cdk = 5
a_apc = 5
a_e2f = 5
deld = 0.05
delb = 0.05
eps_apc = 100   # Note how eps is defined as the inverse of eps in the text
eps_cdk = 100
eps_e2f = 100

Kd = 120    # Kcycde2f
Kb = 40     # Kcycbcdk
Kcdk = 20   # Kcdkapc

dsyn = 0.15
ddeg = 0.009

t_start = 0
t_end = 20000
dt_max = 10
y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]   

bsyn_v = np.linspace(0,0.05,100)
bdeg_v = np.linspace(0,0.005,100)

Array = np.zeros((len(bsyn_v)*len(bdeg_v),10))
Array = pd.DataFrame(Array,columns=['Meta','bsyn','bdeg','Period','G1','SG2','M','E2F Amplitude','APC Amplitude','RP'])
    
i_row = 0
for bsyn in bsyn_v:
    for bdeg in bdeg_v:

        Array.iloc[i_row,1:3] = bsyn,bdeg
                
        print(i_row)

        t_rp = t_end    # CycD synthesis rate always constant at standard level; no effect of the RP is tested
        t_g1 = t_end    # CycD degradation rate always constant at standard level; no effect of DNA damage
        t_g2 = t_end    # Cdk activation threshold always constant at standard level; no effect of DNA damage
        delta_t_g1 = 0
        delta_t_g2 = 0
        
        def f(t,y): return bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)
        
        sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=dt_max,rtol=1e-6,atol=1e-9)
        
        CycD_sol = sol.y[0,:]
        E2F_sol = sol.y[1,:]
        CycB_sol = sol.y[2,:]
        Cdk_sol = sol.y[3,:]
        Apc_sol = sol.y[4,:]
        t_v = sol.t

        out_e2f = Oscillator_analysis(E2F_sol,t_v,n_max=3)
        if (type(out_e2f) != str) and (out_e2f[-1] != 0):
            Array.iloc[i_row,7] = out_e2f[-2]

        out_apc = Oscillator_analysis(Apc_sol,t_v,n_max=3)
        if (type(out_apc) != str) and (out_apc[-1] != 0):
            Array.iloc[i_row,8] = out_apc[-2]
                        
        out_cyc = Oscillator_analysis(CycB_sol,t_v,n_max=3)  
        if (type(out_cyc) != str) and (out_cyc[-1] != 0):
            Array.iloc[i_row,3] = out_cyc[-3]
            
            G1 = t_v[(E2F_sol < 0.95) & (Apc_sol < 0.95)]
            SG2 = t_v[(E2F_sol > 0.95) & (Apc_sol < 0.95)]
            M = t_v[Apc_sol > 0.95]
        
            if len(G1) > 0:
                G1 = np.split(G1,np.where(G1[1:]-G1[0:-1] > 1.5*dt_max)[0]+1)
                G1_length = np.median([i[-1]-i[0] for i in G1]) 
                Array.iloc[i_row,4] = G1_length
            if len(SG2) > 0:
                SG2 = np.split(SG2,np.where(SG2[1:]-SG2[0:-1] > 1.5*dt_max)[0]+1)
                SG2_length = np.median([i[-1]-i[0] for i in SG2]) 
                Array.iloc[i_row,5] = SG2_length
            if len(M) > 0:
                M = np.split(M,np.where(M[1:]-M[0:-1] > 1.5*dt_max)[0]+1)
                M_length = np.median([i[-1]-i[0] for i in M]) 
                Array.iloc[i_row,6] = M_length
     
            # If more than 1 SG2 phase and M phase was found, also check if restriction point is passed
            # In the end, this information was not used in the main text
            if (len(SG2) > 1) and (len(M) > 1):
                # Define timepoint at which CycD synthesis drops; after E2F switch is passed, i.e. in SG2 phase
                t_rp = SG2[1][0]+(SG2[1][-1]-SG2[1][0])/5
        
                def f(t,y): return bist_5d_chain(y,t,t_rp,t_g1,delta_t_g1,t_g2,delta_t_g2,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)
                sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=dt_max,rtol=1e-6,atol=1e-9)
                 
                Apc_sol = sol.y[4,:]
                t_v = sol.t
                # Started cell cycle is completed when cycle passes trough M-phase, i.e.
                # APC > 0.95 where unperturbed cell cycle was in M-phase
                if np.any(Apc_sol[(t_v > M[1][0]) & (t_v < M[1][-1])] > 0.95):
                    Array.iloc[i_row,-1] = 1
                else:
                    Array.iloc[i_row,-1] = 0

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
    ar.loc[15,'Meta'] = 'tstart = '+str(t_start)
    ar.loc[16,'Meta'] = 'tend = '+str(t_end)
    ar.loc[17,'Meta'] = 'dt_max = '+str(dt_max)
               
time = datetime.now()
timestr = time.strftime("%d%m%Y-%H%M%S%f")
filename = 'Screen_Chain_bsyn_bdeg_'+str(timestr)+'.csv'
Array.to_csv(filename)