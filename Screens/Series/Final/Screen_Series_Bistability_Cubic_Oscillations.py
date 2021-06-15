# =============================================================================
# SCRIPT TO RUN PARAMETER SCANS FOR SYSTEM III (TWO CUBIC SWITCHES)
# Related to Figure 6 in the main text
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
# DEFINE NONDIMENSIONALIZED SYSTEM EQUATIONS FOR TWO CUBIC XI
# =============================================================================
def bist_2d_cubic_series(y,t,c,d,eps_apc,eps_cdk,r,n,a_apc,a_cdk1):
    Cyc,Cdk1,Apc = y
    dy = [0,0,0] 
    
    Xi_apc = 1 + a_apc*Apc*(Apc - 1)*(Apc - r)        
    Xi_cdk1 = 1 + a_cdk1*(Cdk1/(d*Cyc))*((Cdk1/(d*Cyc)) - 1)*((Cdk1/(d*Cyc)) - r)
    
    dy[0] = c/d - Cyc*Apc
    dy[1] = eps_cdk*(d*Cyc**(n+1)/(Cyc**n + Xi_cdk1**n) - Cdk1)
    dy[2] = eps_apc*(Cdk1**n/(Cdk1**n + Xi_apc**n) - Apc)
    
    return np.array(dy)


# =============================================================================
# PERFORM THE PARAMETER SCAN: EFFECT OF BISTABLE WIDTH OF CDK-APC SWITCH AND C (Fig 6A-B)
# =============================================================================
n = 15
r = 0.5
d = 2       # Kcyccdk/Kcdkapc
eps_apc = 100
eps_cdk = 100

t_start = 0
y0 = [1e-12,1e-12,1e-12]    # Initial conditions not exactly zero to prevent division by zero

# Calculate maximal value of alpha for which the left fold still has positive x values
xmin = (1 + r + np.sqrt((1 + r)**2 - 3*r))/3 
a_max = -1/(xmin*(xmin - r)*(xmin - 1)) 
 
# Width of Cdk-APC switch will be screened (via a_apc), while keeping the width of
# the Cyc-Cdk switch constant (i.e. a_cdk). Two situations were tested:
# (1) a_cdk = 0: width Cyc-Cdk switch = 0 nM (i.e. ultrasensitive response)
# (2) a_cdk = 9.48: width Cyc-Cdk switch ~30 nM (for n = 15, r = 0.5)
a_cdk = 9.48  # OR 0

A_apc_v = np.concatenate((np.linspace(0,4,50,endpoint=False),np.linspace(4,a_max-1e-3,50))) # Screen relatively more points for narrow switch
C_v = np.linspace(0,1.5,200)

# Initialize arrays to store results for Cyc, Cdk1 and APC oscillations
Cyc_array = np.zeros((len(C_v)*len(A_apc_v),5))
Cyc_array = pd.DataFrame(Cyc_array,columns=['Meta','c','a_apc','Cyc_Period','Cyc_Amplitude'])
Cdk1_array = np.zeros((len(C_v)*len(A_apc_v),5))
Cdk1_array = pd.DataFrame(Cdk1_array,columns=['Meta','c','a_apc','Cdk1_Period','Cdk1_Amplitude'])
Apc_array = np.zeros((len(C_v)*len(A_apc_v),5))
Apc_array = pd.DataFrame(Apc_array,columns=['Meta','c','a_apc','Apc_Period','Apc_Amplitude'])
    
i_row = 0
for a_apc in A_apc_v:
    for c in C_v:
        # Choose time for simulation depending on width of bistable switch 
        if a_cdk == 9.48:
            t_end = max(min(15/c,1000),60)  # 60 is mostly fine, except for small c where simulations take longer (larger period)
        elif a_cdk == 0:
            t_end = 15      # For ultrasensitive Cyc-Cdk response, period of oscillations is small so use shorter time for simulations

        Cyc_array.iloc[i_row,1:3] = c,a_apc
        Cdk1_array.iloc[i_row,1:3] = c,a_apc
        Apc_array.iloc[i_row,1:3] = c,a_apc
                
        print(i_row)
                
        def f(t,y): return bist_2d_cubic_series(y,t,c,d,eps_apc,eps_cdk,r,n,a_apc,a_cdk)    
        sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9)
        
        Cyc_sol = sol.y[0,:]
        Cdk1_sol = sol.y[1,:]
        Apc_sol = sol.y[2,:]
        t_v = sol.t
           
        out_cyc = Oscillator_analysis(Cyc_sol,t_v,n_max=4)
        out_cdk = Oscillator_analysis(Cdk1_sol,t_v,n_max=4)
        out_apc = Oscillator_analysis(Apc_sol,t_v,n_max=4)
        
        # Store amplitude and period of oscillations. If no oscillations 0 from initialization remains.
        if (type(out_cyc) != str) and (out_cyc[-1] != 0):
            Cyc_array.iloc[i_row,-2] = out_cyc[-2]
            Cyc_array.iloc[i_row,-1] = out_cyc[-1]
        if (type(out_cdk) != str) and (out_cdk[-1] != 0):
            Cdk1_array.iloc[i_row,-2] = out_cdk[-2]
            Cdk1_array.iloc[i_row,-1] = out_cdk[-1]           
        if (type(out_apc) != str) and (out_apc[-1] != 0):
            Apc_array.iloc[i_row,-2] = out_apc[-2]
            Apc_array.iloc[i_row,-1] = out_apc[-1]
        
        i_row = i_row + 1 
                  
    for ar in [Cyc_array,Cdk1_array,Apc_array]:
        ar.loc[0,'Meta'] = 'n = '+str(n)
        ar.loc[1,'Meta'] = 'r = '+str(r)
        ar.loc[2,'Meta'] = 'd = '+str(d)
        ar.loc[3,'Meta'] = 'a_cdk = '+str(a_cdk)
        ar.loc[4,'Meta'] = 'eps_apc = '+str(eps_apc)
        ar.loc[5,'Meta'] = 'eps_cdk = '+str(eps_cdk)
        ar.loc[6,'Meta'] = 'tstart = '+str(t_start)
        ar.loc[7,'Meta'] = 'tend = '+str(t_end)
    
time = datetime.now()
timestr = time.strftime("%d%m%Y-%H%M%S%f")
filename = 'Screen_Series_Bistability_Cubic_Apc_aapc_c_acdk_'+str(a_cdk)+'_'+str(timestr)+'.csv'
Apc_array.to_csv(filename)
filename = 'Screen_Series_Bistability_Cubic_Cdk1_aapc_c_acdk_'+str(a_cdk)+'_'+str(timestr)+'.csv'
Cdk1_array.to_csv(filename)
filename = 'Screen_Series_Bistability_Cubic_Cyc_aapc_c_acdk_'+str(a_cdk)+'_'+str(timestr)+'.csv'
Cyc_array.to_csv(filename)


#%%
# =============================================================================
# SECOND SCREEN FOR THE EFFECT OF CHANGING K
# =============================================================================
def width_cubic(r,n,a):
    # 0 cannot be raised to a negative power: use 1e-12 as start
    # cannot divide by 0: use 1-1e-12 as end
    # Define x-axis and corresponding Xi function
    # X always defined between 0 and 1
    X = np.linspace(1e-12,1-1e-12,100000)
    Xi = (1 + a*X*(X - 1)*(X - r))
    dXi = a*(3*X**2 - 2*(1+r)*X + r)
    # Calculate derivative of inverted bistable response curve: 
    dI = dXi*(X/(1-X))**(1/n) + (1/(n*(1-X)**2))*(X/(1-X))**((1-n)/n)*Xi
    # Remove 0 from vector to prevent problems with subsequent calculations
    # to find roots
    dI[dI == 0] = 1e-12
    # Calculate extrema from I: X_max is X value at which I reaches maximum
    # If no extrema are found, return empty arrays
    # Extrema are located where derivative equals zero
    X_max = X[1:][(dI[0:-1]*dI[1:] < 0) & (dI[0:-1] > 0)]
    X_min = X[1:][(dI[0:-1]*dI[1:] < 0) & (dI[0:-1] < 0)]
    # Calculate values of the inverted bistable response at extrema
    I_max = (1 + a*X_max*(X_max - 1)*(X_max - r))*(X_max/(1-X_max))**(1/n)
    I_min = (1 + a*X_min*(X_min - 1)*(X_min - r))*(X_min/(1-X_min))**(1/n)

    if (len(I_max) == 1) and (len(I_min) == 1) and (I_max[0] >= 0) and (I_min[0] >= 0):
        W = abs(I_max - I_min)[0]       
    elif (len(I_max) == 1) and (len(I_min) == 1) and (I_max[0] >= 0) and (I_min[0] < 0):
        W = abs(I_max - 0)[0]
    elif (len(I_max) == 0) and (len(I_min) == 0):
        W = 0
    return(X_max,X_min,I_max,I_min,W)


# =============================================================================
# PERFORM THE PARAMETER SCAN: EFFECT OF THRESHOLD K (Fig 6C-E)
# Three different conditions were scanned:
# (1) Cyc-Cdk switch ultrasensitive, Cdk-APC switch bistable 
# (2) Cyc-Cdk switch bistable, Cdk-APC switch ultrasensitive 
# (3) Cyc-Cdk switch bistable, Cdk-APC switch bistable 
# =============================================================================
n = 15 
r = 0.5
c = 0.5

eps_apc = 100
eps_cdk = 100

t_start = 0
# Depending on the width of the bistable Cyc-Cdk switch, use different times for simulation
t_end = 60  # OR 15 if Cyc-Cdk switch is ultrasensitive

y0 = [1e-12,1e-12,1e-12]

K_cyc_v = np.arange(17,50.5,0.5) 
K_cdk_v = np.arange(17,50.5,0.5) 

# As explained in supplemental information, the width of the switch is affected by
# changing K. To compensate, divide alpha by K. The standard value of alpha was chosen
# to result in similar widths of the bistable regions as before, resulting in the
# three conditions also mentioned above:
# (1) Cyc-Cdk switch ultrasensitive, Cdk-APC switch bistable: a_cdk = 0, a_apc = 189.6
# This results in a width of the Cdk-APC switch ~ 15 nM for n = 15, Kcdkapc = 20, r = 0.5
# (2) Cyc-Cdk switch bistable, Cdk-APC switch ultrasensitive: a_cdk = 379.2, a_apc = 0
# This results in a width of the Cyc-Cdk switch ~ 30 nM for n = 15, Kcyccdk = 40, r = 0.5 (similar to screen for width vs c, as a_apc/Kcyccdk = 9.48)
# (3) Cyc-Cdk switch bistable, Cdk-APC switch bistable: a_cdk = 379.2, a_apc = 189.6
for a_apc,a_cdk in [[189.6,379.2],[0,379.2],[189.6,0]]:
    Cyc_array = np.zeros((len(K_cyc_v)*len(K_cdk_v),7))
    Cyc_array = pd.DataFrame(Cyc_array,columns=['Meta','K_cyc','K_cdk','W_cyc','W_cdk','Cyc_Period','Cyc_Amplitude'])
    Cdk1_array = np.zeros((len(K_cyc_v)*len(K_cdk_v),7))
    Cdk1_array = pd.DataFrame(Cdk1_array,columns=['Meta','K_cyc','K_cdk','W_cyc','W_cdk','Cdk1_Period','Cdk1_Amplitude'])
    Apc_array = np.zeros((len(K_cyc_v)*len(K_cdk_v),7))
    Apc_array = pd.DataFrame(Apc_array,columns=['Meta','K_cyc','K_cdk','W_cyc','W_cdk','Apc_Period','Apc_Amplitude'])
    
    i_row = 0
    for K_cyc in K_cyc_v:
        for K_cdk in K_cdk_v:

            d = K_cyc/K_cdk
            
            Cyc_array.iloc[i_row,1:3] = K_cyc,K_cdk
            Cdk1_array.iloc[i_row,1:3] = K_cyc,K_cdk
            Apc_array.iloc[i_row,1:3] = K_cyc,K_cdk
             
            # Divide a by K and multiply width by K to convert to original dimensions
            Cyc_array.iloc[i_row,3:5] = width_cubic(r,n,a_cdk/K_cyc)[-1]*K_cyc,width_cubic(r,n,a_apc/K_cdk)[-1]*K_cdk
            Cdk1_array.iloc[i_row,3:5] = width_cubic(r,n,a_cdk/K_cyc)[-1]*K_cyc,width_cubic(r,n,a_apc/K_cdk)[-1]*K_cdk
            Apc_array.iloc[i_row,3:5] = width_cubic(r,n,a_cdk/K_cyc)[-1]*K_cyc,width_cubic(r,n,a_apc/K_cdk)[-1]*K_cdk
            
            print(i_row)
                
            def f(t,y): return bist_2d_cubic_series(y,t,c,d,eps_apc,eps_cdk,r,n,a_apc/K_cdk,a_cdk/K_cyc)    
            sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9) 
                    
            Cyc_sol = sol.y[0,:]
            Cdk1_sol = sol.y[1,:]
            Apc_sol = sol.y[2,:]
            t_v = sol.t
               
            out_cyc = Oscillator_analysis(Cyc_sol,t_v,n_max=4)
            out_cdk = Oscillator_analysis(Cdk1_sol,t_v,n_max=4)
            out_apc = Oscillator_analysis(Apc_sol,t_v,n_max=4)
            
            if (type(out_cyc) != str) and (out_cyc[-1] != 0):
                Cyc_array.iloc[i_row,-2] = out_cyc[-2]
                Cyc_array.iloc[i_row,-1] = out_cyc[-1]
            if (type(out_cdk) != str) and (out_cdk[-1] != 0):
                Cdk1_array.iloc[i_row,-2] = out_cdk[-2]
                Cdk1_array.iloc[i_row,-1] = out_cdk[-1]           
            if (type(out_apc) != str) and (out_apc[-1] != 0):
                Apc_array.iloc[i_row,-2] = out_apc[-2]
                Apc_array.iloc[i_row,-1] = out_apc[-1]
            
            i_row = i_row + 1 
                  
    for ar in [Cyc_array,Cdk1_array,Apc_array]:
        ar.loc[0,'Meta'] = 'n = '+str(n)
        ar.loc[1,'Meta'] = 'r = '+str(r)
        ar.loc[2,'Meta'] = 'c = '+str(c)
        ar.loc[3,'Meta'] = 'a_apc = '+str(a_apc)
        ar.loc[4,'Meta'] = 'a_cdk = '+str(a_cdk)
        ar.loc[5,'Meta'] = 'eps_apc = '+str(eps_apc)
        ar.loc[6,'Meta'] = 'eps_cdk = '+str(eps_cdk)
        ar.loc[7,'Meta'] = 'tstart = '+str(t_start)
        ar.loc[8,'Meta'] = 'tend = '+str(t_end)
    
    time = datetime.now()
    timestr = time.strftime("%d%m%Y-%H%M%S%f")
    filename = 'Screen_Series_Bistability_Cubic_Apc_K_aapc_'+str(a_apc)+'_acdk_'+str(a_cdk)+'_'+str(timestr)+'.csv'
    Apc_array.to_csv(filename)
    filename = 'Screen_Series_Bistability_Cubic_Cdk1_K_aapc_'+str(a_apc)+'_acdk_'+str(a_cdk)+'_'+str(timestr)+'.csv'
    Cdk1_array.to_csv(filename)
    filename = 'Screen_Series_Bistability_Cubic_Cyc_K_aapc_'+str(a_apc)+'_acdk_'+str(a_cdk)+'_'+str(timestr)+'.csv'
    Cyc_array.to_csv(filename)
