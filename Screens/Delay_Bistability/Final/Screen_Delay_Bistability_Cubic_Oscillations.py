# =============================================================================
# SCRIPT TO RUN PARAMETER SCANS FOR SYSTEM II (DELAYED CUBIC BISTABILITY)
# Related to Figure 5 in the main text
# =============================================================================
from __future__ import division
import numpy as np
import pandas as pd
from datetime import datetime

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
# DEFINE NONDIMENSIONALIZED SYSTEM EQUATIONS FOR CUBIC XI WITH DELAY
# =============================================================================
def delay_bist_2d_cubic(y,t,tau,c,eps,r,n,a):
    Cdk1 = y(0)
    Apc = y(1)
    Cdk1_hist = y(0,t-tau)
    Apc_hist = y(1,t-tau)

    Xi = 1 + a*Apc*(Apc - 1)*(Apc - r) 
            
    f = [c - Cdk1*Apc,
        eps*(Cdk1_hist**n/(Cdk1_hist**n + Xi**n) - Apc)] 
    return(f)

   
# =============================================================================
# PERFORM THE PARAMETER SCAN
# =============================================================================
eps = 100
N_v = [15]
R_v = [0.5]

C_v = [0.2,0.5,0.8]

Tau_v = np.linspace(0,5,100)

t_start = 0
t_end = 200
dt = 0.01
y0 = [0,0]

for n in N_v:
    for r in R_v:
        for c in C_v:
        
            xmin = (1 + r + np.sqrt((1 + r)**2 - 3*r))/3 
            a_max = -1/(xmin*(xmin - r)*(xmin - 1))
            
            A_v = np.linspace(0,a_max,150)

            # Initialize arrays to store results for Cdk1 and APC oscillations
            Cdk1_array = np.zeros((len(Tau_v)*len(A_v),5))
            Cdk1_array = pd.DataFrame(Cdk1_array,columns=['Meta','tau','a','Cdk1_Period','Cdk1_Amplitude'])
            Apc_array = np.zeros((len(Tau_v)*len(A_v),5))
            Apc_array = pd.DataFrame(Apc_array,columns=['Meta','tau','a','Apc_Period','Apc_Amplitude'])

            i_row = 0
            for tau in Tau_v:
                for a in A_v:
            
                    Cdk1_array.iloc[i_row,1:3] = tau,a
                    Apc_array.iloc[i_row,1:3] = tau,a
                          
                    print(i_row)

                    from jitcdde import jitcdde, y, t                  
                    f = delay_bist_2d_cubic(y,t,tau,c,eps,r,n,a)
        
                    DDE = jitcdde(f,verbose=False)
                    DDE.set_integration_parameters(max_step=0.01)
                    DDE.constant_past(y0)
                    DDE.step_on_discontinuities()
                
                    sol = []
                    t_v = []
                    for time in np.arange(DDE.t, DDE.t+t_end, dt):
                        t_v.append(time)
                        sol.append(DDE.integrate(time))
                    sol = np.array(sol)    
                    t_v = np.array(t_v)  
                    
                    Cdk1_sol = sol[:,0]
                    Apc_sol = sol[:,1]
                                   
                    out_cdk = Oscillator_analysis(Cdk1_sol,t_v)
                    out_apc = Oscillator_analysis(Apc_sol,t_v) 

                    # Store amplitude and period of oscillations. If no oscillations 0 from initialization remains.                    
                    if (type(out_cdk) != str) and (out_cdk[-1] != 0):
                        Cdk1_array.iloc[i_row,-2] = out_cdk[-2]
                        Cdk1_array.iloc[i_row,-1] = out_cdk[-1]           
                    if (type(out_apc) != str) and (out_apc[-1] != 0):
                        Apc_array.iloc[i_row,-2] = out_apc[-2]
                        Apc_array.iloc[i_row,-1] = out_apc[-1]
                            
                    i_row = i_row + 1 
                          
            for ar in [Cdk1_array,Apc_array]:
                ar.loc[0,'Meta'] = 'n = '+str(n)
                ar.loc[1,'Meta'] = 'r = '+str(r)
                ar.loc[2,'Meta'] = 'c = '+str(c)
                ar.loc[3,'Meta'] = 'eps = '+str(eps)
                ar.loc[4,'Meta'] = 'tstart = '+str(t_start)
                ar.loc[5,'Meta'] = 'tend = '+str(t_end)
                ar.loc[6,'Meta'] = 'dt = '+str(dt)
                
            time = datetime.now()
            timestr = time.strftime("%d%m%Y-%H%M%S%f")
            filename = 'Screen_Delay_Bistability_Cubic_Apc_tau_a_c_'+str(c)+'_n_'+str(n)+'_r_'+str(r)+'_'+str(timestr)+'.csv'
            Apc_array.to_csv(filename)
            filename = 'Screen_Delay_Bistability_Cubic_Cdk1_tau_a_c_'+str(c)+'_n_'+str(n)+'_r_'+str(r)+'_'+str(timestr)+'.csv'
            Cdk1_array.to_csv(filename)

