from __future__ import division

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

    
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
# DEFINE NONDIMENSIONALIZED SYSTEM EQUATIONS FOR PIECEWISE XI
# =============================================================================
def MA(y,t,bsyn,bdeg,kpg,kdg,kpe,kass1,kdis1,kcat1,kact,kinact,GWLtot,ENSAtot,PP2Atot):
    Cdk1,Gp,C1,Ep,Apc = y
    dy = [0,0,0,0,0]
            
    dy[0] = bsyn - bdeg*Cdk1*Apc
    dy[1] = kpg*(GWLtot - Gp)*Cdk1 - kdg*Gp*(PP2Atot - C1)
    dy[2] = kass1*Ep*(PP2Atot - C1) - (kdis1 + kcat1)*C1
    dy[3] = kdis1*C1 + kpe*(ENSAtot - Ep - C1)*Gp - kass1*Ep*(PP2Atot - C1)
    dy[4] = kact*Cdk1*(1 - Apc) - kinact*Apc*(PP2Atot - C1)
    
    return dy
    
def bist_2d_piecewise_asymmetric(y,t,c,eps,n,x_max,x_min,xi_max,xi_min,lim):
    Cdk1,Apc = y

    dy = [0,0]

    if (Apc <= x_max):
        Xi = (xi_max-1)/(x_max)*Apc+1
    elif (x_max < Apc <= x_min):
        Xi = (xi_min-xi_max)/(x_min-x_max)*(Apc - x_max) + xi_max
    elif (x_min < Apc):
        Xi = (1-xi_min)/(1-x_min)*(Apc - x_min) + xi_min

    dy[0] = c - Cdk1*Apc
    dy[1] = eps*(lim*Cdk1**n/(Cdk1**n + Xi**n) - Apc)
    
    return np.array(dy)


# =============================================================================
# PERFORM THE PARAMETER SCAN
# =============================================================================
t_start = 0
    
CDKtot = 40
GWLtot = 40
PP2Atot = 40
ENSAtot = 200
n = 5
kpe = 10**0.6
kpg = 10**(-1.15)
kdg = 10**0.85
kass1 = 10**0.7
kdis1 = 10**1.45
kcat1 = 10**1.2
kact = 10**(-0.2)
kinact = 10**0.2
lim = 0.95
eps = 500
# Obtain folds by first plotting the response curve
CDK_LF = 17.34
APC_LF = 0.60
CDK_RF = 23.9
APC_RF = 0.36
K = CDK_LF + (CDK_RF - CDK_LF)/2

# Calculate Xi such that CDK of the folds of piecewise and S coincide
Xi_LF = (1/(APC_LF/(lim - APC_LF))**(1/n))*CDK_LF/K
Xi_RF = (1/(APC_RF/(lim - APC_RF))**(1/n))*CDK_RF/K

bsyn_v = np.linspace(0.25,1.25,70)
bdeg_v = np.linspace(0.05,0.15,70)

# Initialize arrays to store results for Cdk1 and APC oscillations
Cdk1_array = np.zeros((len(bsyn_v)*len(bdeg_v),7))
Cdk1_array = pd.DataFrame(Cdk1_array,columns=['Meta','bsyn','bdeg','Cdk1_Period_Module','Cdk1_Amplitude_Module','Cdk1_Period_MA','Cdk1_Amplitude_MA'])
Apc_array = np.zeros((len(bsyn_v)*len(bdeg_v),7))
Apc_array = pd.DataFrame(Apc_array,columns=['Meta','bsyn','bdeg','Apc_Period_Module','Apc_Amplitude_Module','Apc_Period_MA','Apc_Amplitude_MA'])

i_row = 0
for bsyn in bsyn_v:
   for bdeg in bdeg_v:
        print(i_row)
        
        Cdk1_array.iloc[i_row,1] = bsyn
        Apc_array.iloc[i_row,1] = bsyn
        Cdk1_array.iloc[i_row,2] = bdeg
        Apc_array.iloc[i_row,2] = bdeg
        
        ### Solve module system
        y0 = [0,0]
        t_end = 50
        c = bsyn/(bdeg*K)
        def f(t,y): return bist_2d_piecewise_asymmetric(y,t,c,eps,n,APC_RF,APC_LF,Xi_RF,Xi_LF,lim)   
        sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9)  
        Cdk1_sol = sol.y[0,:]
        Apc_sol = sol.y[1,:]
        t_v = sol.t      
        out_cdk = Oscillator_analysis(Cdk1_sol,t_v)
        out_apc = Oscillator_analysis(Apc_sol,t_v) 
                
        # Store amplitude and period of oscillations. If no oscillations, 0 from initialization remains.
        if (type(out_cdk) != str) and (out_cdk[-1] != 0):
            Cdk1_array.iloc[i_row,-4] = out_cdk[-2]/bdeg
            Cdk1_array.iloc[i_row,-3] = out_cdk[-1]*K
        if (type(out_apc) != str) and (out_apc[-1] != 0):
            Apc_array.iloc[i_row,-4] = out_apc[-2]/bdeg
            Apc_array.iloc[i_row,-3] = out_apc[-1]

        ### Solve mass action system
        y0 = [0,0,0,0,0]
        t_end = 50/bdeg
        def h(t,y): return MA(y,t,bsyn,bdeg,kpg,kdg,kpe,kass1,kdis1,kcat1,kact,kinact,GWLtot,ENSAtot,PP2Atot)
        sol = solve_ivp(h,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9)  
        Cdk1_sol = sol.y[0,:]
        Apc_sol = Cdk1_sol/(Cdk1_sol + kinact*(PP2Atot - sol.y[2,:])/kact)
        t_v = sol.t      
        out_cdk = Oscillator_analysis(Cdk1_sol,t_v)
        out_apc = Oscillator_analysis(Apc_sol,t_v) 

        # Store amplitude and period of oscillations. If no oscillations, 0 from initialization remains.
        if (type(out_cdk) != str) and (out_cdk[-1] != 0):
            Cdk1_array.iloc[i_row,-2] = out_cdk[-2]
            Cdk1_array.iloc[i_row,-1] = out_cdk[-1]           
        if (type(out_apc) != str) and (out_apc[-1] != 0):
            Apc_array.iloc[i_row,-2] = out_apc[-2]
            Apc_array.iloc[i_row,-1] = out_apc[-1]
                
        i_row = i_row + 1 
         
for ar in [Cdk1_array,Apc_array]:
    ar.loc[0,'Meta'] = 'CDK_LF = '+str(CDK_LF)
    ar.loc[1,'Meta'] = 'CDK_RF = '+str(CDK_RF)
    ar.loc[2,'Meta'] = 'APC_LF = '+str(APC_LF)
    ar.loc[3,'Meta'] = 'APC_RF = '+str(APC_RF)
    ar.loc[4,'Meta'] = 'APC_lim = '+str(lim)
    ar.loc[5,'Meta'] = 'n = '+str(n)
    ar.loc[6,'Meta'] = 'GWLtot = '+str(GWLtot)
    ar.loc[7,'Meta'] = 'PP2Atot = '+str(PP2Atot)
    ar.loc[8,'Meta'] = 'ENSAtot = '+str(ENSAtot)
    ar.loc[9,'Meta'] = 'kpe = '+str(kpe)
    ar.loc[10,'Meta'] = 'kpg = '+str(kpg)
    ar.loc[11,'Meta'] = 'kdg = '+str(kdg)
    ar.loc[12,'Meta'] = 'kass1 = '+str(kass1)
    ar.loc[13,'Meta'] = 'kdis1 = '+str(kdis1)
    ar.loc[14,'Meta'] = 'kcat1 = '+str(kcat1)
    ar.loc[15,'Meta'] = 'kact = '+str(kact)
    ar.loc[16,'Meta'] = 'kinact = '+str(kinact)
    ar.loc[17,'Meta'] = 'eps = '+str(eps)
    ar.loc[18,'Meta'] = 'tstart = '+str(t_start)
    ar.loc[19,'Meta'] = 'tend = '+str(t_end)

time = datetime.now()
timestr = time.strftime("%d%m%Y-%H%M%S%f")
filename = 'Screen_Mass_Action_APC_'+str(timestr)+'.csv'
Apc_array.to_csv(filename)
filename = 'Screen_Mass_Action_Cdk1_'+str(timestr)+'.csv'
Cdk1_array.to_csv(filename)

