# =============================================================================
# SCRIPT TO RUN PARAMETER SCANS FOR ARNOLD TONGUES
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

# Define function to find repeating pattern in array A, with all elements in A being positive numbers
def pattern_recognition(A,err):
    import numpy as np
    if type(A) != np.ndarray:
        A = np.array(A)
    Repeats = []
    A_max = np.max(A)
    for i in range(len(A)):
        # Calculate difference of list with time-shifted version of itself
        D = abs(A - np.concatenate(((i+1)*[0],A[0:-(i+1)])))
        # print(D)
        # Where difference is sufficiently small (theoretically it should equal zero),
        # elements are considered equal. Replace element by -1
        D[D < err*A_max] = -1
        # Difference list must end in one long repeat of sufficiently small differences
        # to ensure that not just a subset of a longer repeating pattern was detected
        # As A only has positive elements and elements of repeating pattern have been 
        # changed to -1, just look for the number of sign changes in D which should equal 1.
        if (D[-1] == -1) & (len(D[1:][D[0:-1]*D[1:] < 0]) == 1):
            Repeats.append(A[D == -1])
    # print(Repeats)
    if len(Repeats) > 1:
        R = Repeats[0][0:len(Repeats[0])-len(Repeats[1])]
    else:
        R = 0
    return(R)
            

# =============================================================================
# DEFINE SYSTEM EQUATIONS FOR INTERLINKED SWITCHES WITH ADJUSTABLE CYC-CDK SWITCH
# No non-dimensionalization was performed here
# =============================================================================
def bist_5d_chain_sin(y,t,omega,A_a,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb):
    CycD,E2F,CycB,Cdk1,Apc = y
    dy = [0,0,0,0,0] 
    
    Xi_apc = 1 + a_apc*Apc*(Apc - 1)*(Apc - r)        
    Xi_cdk = 1 + (a_cdk + A_a + A_a*np.sin(omega*t))*(Cdk1/CycB)*((Cdk1/CycB) - 1)*((Cdk1/CycB) - r)        
    Xi_e2f = 1 + a_e2f*E2F*(E2F - 1)*(E2F - r)

    dy[0] = dsyn - ddeg*CycD*(Apc + deld)
    dy[1] = eps_e2f*(CycD**n/(CycD**n + (Kd*Xi_e2f)**n) - E2F)
    dy[2] = bsyn*E2F - bdeg*CycB*(Apc + delb)
    dy[3] = eps_cdk*(CycB**(n+1)/(CycB**n + (Kb*Xi_cdk)**n) - Cdk1)
    dy[4] = eps_apc*(Cdk1**n/(Cdk1**n + (Kcdk*Xi_apc)**n) - Apc)
    
    return np.array(dy)


# =============================================================================
# PERFORM THE PARAMETER SCAN
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

# First calculate cell cycle frequency for unforced cell cycle
t_start = 0
t_end = 30000
t_eval = np.linspace(t_start,t_end,t_end)
t_incr_max = 10
y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]
def f(t,y): return bist_5d_chain_sin(y,t,0,0,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)
sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=t_incr_max,rtol=1e-6,atol=1e-9)
t_v = sol.t
Cdk_sol = sol.y[3,:]
period_0 = Oscillator_analysis(Cdk_sol,t_v)[-2]     # Unforced cell cycle period
if (type(period_0) != str) and (period_0 != 0):
    omega_0 = 2*np.pi/period_0                      # Unforced cell cycle frequency


# Define forcing frequency as fractions of unforced cell cycle frequency
omega_v = omega_0*np.arange(0.3,3,0.005)
A_a_v = np.linspace(0,10,50)   # As this screen takes a very long time, the A_v values can be separated over multiple vectors and simulated in parallel

Array = np.zeros((len(omega_v)*len(A_a_v),6))
Array = pd.DataFrame(Array,columns=['Meta','Omega','A_a','subcycles_osc','p/q','Locking'])

# Determine wich p:q ratios should be detected
pmax = 5
pq_v = []
for p in range(1,pmax+1):
    for q in range(1,pmax+1):
        if p/q not in [j[1] for j in pq_v]:
            pq_v.append([str(p)+'/'+str(q),p/q])


i_row = 0
for omega in omega_v:
    for A_a in A_a_v:
        print(i_row)
        # Period of forcing oscillator
        period = 2*np.pi/omega

        Array.iloc[i_row,1:3] = omega,A_a

        t_start = 0
        t_end = 45000
        t_incr_max = 10
        t_eval = np.linspace(t_start,t_end,t_end)
        y0 = [1e-12,1e-12,1e-12,1e-12,1e-12]
        def f(t,y): return bist_5d_chain_sin(y,t,omega,A_a,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb)
        sol = solve_ivp(f,(t_start,t_end),y0,method='Radau',t_eval=t_eval,max_step=t_incr_max,rtol=1e-6,atol=1e-9)
        t_v = sol.t
        Cdk_sol = sol.y[3,:]

        # Remove first third of solution to only consider converged solution
        y_osc = np.array(Cdk_sol[t_v > t_end//3])
        t_osc = np.array(t_v[t_v > t_end//3])
        # Calculate derivative of solution
        dy_osc = y_osc[1:]-y_osc[0:-1]
        # Replace zeros by small positive number to prevent not finding any sign change in subsequent line
        dy_osc[dy_osc == 0] = 1e-12
        # Extrema are located where product of two neighbouring elements in dy is negative, i.e.
        # sign change
        t_max = t_osc[1:-1][(dy_osc[0:-1]*dy_osc[1:] < 0) & (dy_osc[0:-1] > 0)]
        if len(t_max) > 1:
            # Time differences between maxima
            dt_max = t_max[1:]-t_max[0:-1]

            # Find repeating pattern in time differences between maxima
            p_err = 0.005
            pattern = pattern_recognition(dt_max,p_err)
            period_osc = np.sum(pattern)  # Overall period of forced cell cycle equals sum of detected pattern
            if (period_osc != 0):
                for i in pq_v:
                    if (abs(i[1] - period_osc/period) < p_err):
                        Array.iloc[i_row,-1] = i[1]
                        Array.iloc[i_row,-2] = i[0]
                        Array.iloc[i_row,-3] = len(pattern)
                        break

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
    ar.loc[17,'Meta'] = 't_incr_max = '+str(t_incr_max)
    ar.loc[18,'Meta'] = 'pmax = '+str(pmax)
    ar.loc[19,'Meta'] = 'p_err = '+str(p_err)
    # ar.loc[20,'Meta'] = 'omega_err = '+str(omega_err)

time = datetime.now()
timestr = time.strftime("%d%m%Y-%H%M%S%f")
filename = 'Arnold_Tongues_'+str(timestr)+'.csv'
Array.to_csv(filename)
