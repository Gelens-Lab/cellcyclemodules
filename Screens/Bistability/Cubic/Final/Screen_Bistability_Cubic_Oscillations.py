# =============================================================================
# SCRIPT TO RUN PARAMETER SCANS FOR SYSTEM I (CUBIC BISTABILITY)
# Related to Figure 3 in the main text
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

# Function to calculate width of the bistable response for a cubic scaling function 
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
# DEFINE NONDIMENSIONALIZED SYSTEM EQUATIONS FOR CUBIC XI
# =============================================================================
def bist_2d_cubic(y,t,c,eps,r,n,a):
    Cdk1,Apc = y

    dy = [0,0]

    Xi = 1 + a*Apc*(Apc - 1)*(Apc - r) 
            
    dy[0] = c - Cdk1*Apc
    dy[1] = eps*(Cdk1**n/(Cdk1**n + Xi**n) - Apc)
    
    return np.array(dy)

# =============================================================================
# PERFORM THE PARAMETER SCAN
# =============================================================================
NR_v = [[15,0.5],[5,0.5],[1,0.5],[300,0.5],[15,0.75],[15,0.25]] # Different combinations of n and r to test
eps = 100

t_start = 0
t_end = 100
y0 = [0,0]

C_v = np.linspace(0,1.2,150)
for n,r in NR_v: 
    # Calculate maximal value of alpha for which the left fold still has positive x values
    xmin = (1 + r + np.sqrt((1 + r)**2 - 3*r))/3 
    a_max = -1/(xmin*(xmin - r)*(xmin - 1))
    
    A_v = np.linspace(0,a_max,150)

    # Initialize arrays to store results for Cdk1 and APC oscillations
    Cdk1_array = np.zeros((len(A_v)*len(C_v),7))
    Cdk1_array = pd.DataFrame(Cdk1_array,columns=['Meta','c','a','N_ss','SS_pos','Cdk1_Period','Cdk1_Amplitude'])
    Apc_array = np.zeros((len(A_v)*len(C_v),7))
    Apc_array = pd.DataFrame(Apc_array,columns=['Meta','c','a','N_ss','SS_pos','Apc_Period','Apc_Amplitude'])

    i_row = 0
    for c in C_v:
        for a in A_v:

            Cdk1_array.iloc[i_row,1:3] = c,a
            Apc_array.iloc[i_row,1:3] = c,a
            print(i_row)

            def f(t,y): return bist_2d_cubic(y,t,c,eps,r,n,a)    
            sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',max_step=0.1,rtol=1e-6,atol=1e-9)
        
            Cdk1_sol = sol.y[0,:]
            Apc_sol = sol.y[1,:]
            t_v = sol.t
                
            out_cdk = Oscillator_analysis(Cdk1_sol,t_v)
            out_apc = Oscillator_analysis(Apc_sol,t_v) 
            
            # Store amplitude and period of oscillations. If no oscillations 0 from initialization remains.
            if (type(out_cdk) != str) and (out_cdk[-1] != 0):
                Cdk1_array.iloc[i_row,-2] = out_cdk[-2]
                Cdk1_array.iloc[i_row,-1] = out_cdk[-1]           
            if (type(out_apc) != str) and (out_apc[-1] != 0):
                Apc_array.iloc[i_row,-2] = out_apc[-2]
                Apc_array.iloc[i_row,-1] = out_apc[-1]
 
            # Calculate the number and position of steady states
            Apc = np.linspace(1e-12,1+1e-12,100000)
            Cdk1 = c/Apc
            Xi_v = 1 + a*Apc*(Apc - 1)*(Apc - r) 
            F = eps*(Cdk1**n/(Cdk1**n + Xi_v**n) - Apc)     # Nullclines intersect where F = 0
            F[F == 0] = 1e-12                               # Change 0 by small positive number to prevent finding no sign changes in next line
            Apc_roots = Apc[1:][F[0:-1]*F[1:]<0]            # Determine APC levels for which F = 0 (i.e. where F changes sign; i.e. if product of subsequent elements < 0)
            N_roots = len(Apc_roots)                        # Determine number of roots
            Cdk1_array.iloc[i_row,-4] = N_roots
            Apc_array.iloc[i_row,-4] = N_roots
            if N_roots == 1:
                X_max,X_min,Y_max,Y_min,W = width_cubic(r,n,a)
                if Apc_roots[0] < X_max:                    # Stable steady state on bottom branch if APC coordinate < right fold
                    Cdk1_array.iloc[i_row,-3] = 'Bottom'
                    Apc_array.iloc[i_row,-3] = 'Bottom'
                elif X_max < Apc_roots[0] < X_min:
                    Cdk1_array.iloc[i_row,-3] = 'Middle'
                    Apc_array.iloc[i_row,-3] = 'Middle'
                elif Apc_roots[0] > X_min:                  # Stable steady state on top branch if APC coordinate > right fold
                    Cdk1_array.iloc[i_row,-3] = 'Top'
                    Apc_array.iloc[i_row,-3] = 'Top'
            else:   # If multiple roots, check if only one is located in between folds
                unstable = np.array([1 if ((i > X_max) & (i < X_min)) else 0 for i in Apc_roots])  # Add 1 for each APC root between the folds
                if np.sum(unstable) == 2:   # If multiple steady states between folding points, check the location of third steady state and store it
                    if Apc_roots[unstable == 0] < X_max:
                        Cdk1_array.iloc[i_row,-3] = 'Bottom'
                        Apc_array.iloc[i_row,-3] = 'Bottom'
                    elif Apc_roots[unstable == 0] > X_min:
                        Cdk1_array.iloc[i_row,-3] = 'Top'
                        Apc_array.iloc[i_row,-3] = 'Top'
                # If only one unstable steady state is found, system is bistable and 0 from initializatioon remains                        
            i_row = i_row + 1 
             
    for ar in [Cdk1_array,Apc_array]:
        ar.loc[0,'Meta'] = 'n = '+str(n)
        ar.loc[1,'Meta'] = 'r = '+str(r)
        ar.loc[2,'Meta'] = 'eps = '+str(eps)
        ar.loc[3,'Meta'] = 'tstart = '+str(t_start)
        ar.loc[4,'Meta'] = 'tend = '+str(t_end)

    time = datetime.now()
    timestr = time.strftime("%d%m%Y-%H%M%S%f")
    filename = 'Screen_Bistability_Cubic_Apc_a_c_n_'+str(n)+'_r_'+str(r)+'_'+str(timestr)+'.csv'
    Apc_array.to_csv(filename)
    filename = 'Screen_Bistability_Cubic_Cdk1_a_c_n_'+str(n)+'_r_'+str(r)+'_'+str(timestr)+'.csv'
    Cdk1_array.to_csv(filename)
