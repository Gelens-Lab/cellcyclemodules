#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A modular approach for modeling the cell cycle based on functional response curves

DEFINE SOME FUNCTIONS USED FOR ANALYSING DATA AND GENERATING FIGURES
"""
import numpy as np
import os
import pandas as pd

# =============================================================================
# DEFINE FUNCTION TO FIND EXTREMA, PERIOD AND AMPLITUDE OF OSCILLATOR
# =============================================================================
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
            amplitude = np.median(amplitudes)
            period = np.median(list(abs(t_max[1:]-t_max[0:-1])) + list(abs(t_min[1:]-t_min[0:-1])))
            out = y_max,y_min,t_max,t_min,period,amplitude          
    else:
        out = 'Oscillator could not be validated; not enough or no oscillations in the given time frame'
        
    return(out)


# =============================================================================
# DEFINE SYSTEM EQUATIONS WITH DELAY (jitcdde solver)
# Note how eps is defined as the inverse of eps in the text
# =============================================================================
# Delay equations for bistable module; for a = 0, delay system for the
# ultrasensitive module are obtained
def delay_bist_2d_cubic(y,t,tau,c,eps,r,n,a):
    Cdk1 = y(0)
    Apc = y(1)
    Cdk1_hist = y(0,t-tau)
    Apc_hist = y(1,t-tau)

    Xi = 1 + a*Apc*(Apc - 1)*(Apc - r) 
            
    f = [c - Cdk1*Apc,
        eps*(Cdk1_hist**n/(Cdk1_hist**n + Xi**n) - Apc)] 
    return(f)

def delay_bist_2d_piecewise(y,t,tau,c,eps,n,x_max,x_min,xi_max,xi_min):
    Cdk1 = y(0)
    Apc = y(1)
    Cdk1_hist = y(0,t-tau)
    Apc_hist = y(1,t-tau)

    if (Apc <= x_max):
        Xi = (xi_max-1)/(x_max)*Apc+1
    elif (x_max < Apc <= x_min):
        Xi = (xi_min-xi_max)/(x_min-x_max)*(Apc - x_max) + xi_max
    elif (x_min < Apc):
        Xi = (1-xi_min)/(1-x_min)*(Apc - x_min) + xi_min
                    
    f = [c - Cdk1*Apc,
        eps*(Cdk1_hist**n/(Cdk1_hist**n + Xi**n) - Apc)] 
    
    return(f)
    
    
# State dependent delays for the bistable module
def delay_bist_2d_cubic_state_dependent(y,t,tau1,tau2,c,eps,r,n,a):
    Cdk1 = y(0)
    Apc = y(1)
  
    p = 5
    tau = tau1 + (tau2 - tau1)*(Apc**p/(0.5**p + Apc**p))
  
    Cdk1_hist = y(0,t-tau)
    Apc_hist = y(1,t-tau)

    Xi = 1 + a*Apc*(Apc - 1)*(Apc - r) 
            
    f = [c - Cdk1*Apc,
        eps*(Cdk1_hist**n/(Cdk1_hist**n + Xi**n) - Apc)] 
    return(f)
             

# =============================================================================
# DEFINE SYSTEM EQUATIONS WITHOUT DELAY (scipy solve_ivp solver)
# Note how eps is defined as the inverse of eps in the text
# =============================================================================
# System i
def bist_2d_cubic(y,t,c,eps,r,n,a):
    Cdk1,Apc = y

    dy = [0,0]

    Xi = 1 + a*Apc*(Apc - 1)*(Apc - r) 
            
    dy[0] = c - Cdk1*Apc
    dy[1] = eps*(Cdk1**n/(Cdk1**n + Xi**n) - Apc)
    
    return np.array(dy)

# Piecewise approximation
def bist_2d_piecewise_asymmetric(y,t,c,eps,n,x_max,x_min,xi_max,xi_min,lim=1):
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

# System iii
def bist_2d_cubic_series(y,t,c,d,eps_apc,eps_cdk,r,n,a_apc,a_cdk1):
    Cyc,Cdk1,Apc = y
    # Cdk1 = Cdk1/Kcdk
    dy = [0,0,0] 
    
    Xi_apc = 1 + a_apc*Apc*(Apc - 1)*(Apc - r)        
    Xi_cdk1 = 1 + a_cdk1*(Cdk1/(d*Cyc))*((Cdk1/(d*Cyc)) - 1)*((Cdk1/(d*Cyc)) - r)
    
    dy[0] = c/d - Cyc*Apc
    dy[1] = eps_cdk*(d*Cyc**(n+1)/(Cyc**n + Xi_cdk1**n) - Cdk1)
    dy[2] = eps_apc*(Cdk1**n/(Cdk1**n + Xi_apc**n) - Apc)
    
    return np.array(dy)

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

def bist_5d_chain_sin(y,t,omega,A_a,dsyn,ddeg,bsyn,bdeg,Kd,Kb,Kcdk,eps_apc,eps_cdk,eps_e2f,r,n,a_apc,a_cdk,a_e2f,deld,delb):
    CycD,E2F,CycB,Cdk1,Apc = y
    dy = [0,0,0,0,0] 
    
    Xi_apc = 1 + a_apc*Apc*(Apc - 1)*(Apc - r)        
    # Modulate a_cdk by a sine wave
    Xi_cdk = 1 + (a_cdk + A_a + A_a*np.sin(omega*t))*(Cdk1/CycB)*((Cdk1/CycB) - 1)*((Cdk1/CycB) - r)        

    Xi_e2f = 1 + a_e2f*E2F*(E2F - 1)*(E2F - r)

    dy[0] = dsyn - ddeg*CycD*(Apc + deld)
    dy[1] = eps_e2f*(CycD**n/(CycD**n + (Kd*Xi_e2f)**n) - E2F)
    dy[2] = bsyn*E2F - bdeg*CycB*(Apc + delb)
    dy[3] = eps_cdk*(CycB**(n+1)/(CycB**n + (Kb*Xi_cdk)**n) - Cdk1)
    dy[4] = eps_apc*(Cdk1**n/(Cdk1**n + (Kcdk*Xi_apc)**n) - Apc)
    
    return np.array(dy)

# =============================================================================
# DEFINE SYSTEM EQUATIONS FOR MASS ACTION MODEL
# =============================================================================
# Model to perform numerical continuation
class PP2A_GWL_ENSA(object):
    def __init__(self,**params):
        self.kpg,self.kdg,self.kpe,self.kass1,self.kdis1,self.kcat1,self.n,self.kact,self.kinact,self.GWLtot,self.ENSAtot,self.PP2Atot = params['kpg'],params['kdg'],params['kpe'],params['kass1'],params['kdis1'],params['kcat1'],params['n'],params['kact'],params['kinact'],params['GWLtot'],params['ENSAtot'],params['PP2Atot']     
    
    def __call__(self,y,bp):
        kpg,kdg,kpe,kass1,kdis1,kcat1,n,kact,kinact,GWLtot,ENSAtot,PP2Atot,CDKtot =         self.kpg,self.kdg,self.kpe,self.kass1,self.kdis1,self.kcat1,self.n,self.kact,self.kinact,self.GWLtot,self.ENSAtot,self.PP2Atot,bp

        Gp,C1,Ep = y
        dy = [0,0,0]
            
        dy[0] = kpg*(GWLtot - Gp)*CDKtot - kdg*Gp*(PP2Atot - C1)
        dy[1] = kass1*Ep*(PP2Atot - C1) - (kdis1 + kcat1)*C1
        dy[2] = kdis1*C1 + kpe*(ENSAtot - Ep - C1)*Gp - kass1*Ep*(PP2Atot - C1)
    
        return dy
    
# Oscillator model for time simulations
def PP2A_GWL_ENSA_Osc(y,t,bsyn,bdeg,kpg,kdg,kpe,kass1,kdis1,kcat1,kact,kinact,GWLtot,ENSAtot,PP2Atot):
        Cdk1,Gp,C1,Ep,Apc = y
        dy = [0,0,0,0,0]
                
        dy[0] = bsyn - bdeg*Cdk1*Apc
        dy[1] = kpg*(GWLtot - Gp)*Cdk1 - kdg*Gp*(PP2Atot - C1)
        dy[2] = kass1*Ep*(PP2Atot - C1) - (kdis1 + kcat1)*C1
        dy[3] = kdis1*C1 + kpe*(ENSAtot - Ep - C1)*Gp - kass1*Ep*(PP2Atot - C1)
        dy[4] = kact*Cdk1*(1 - Apc) - kinact*Apc*(PP2Atot - C1)
        
        return dy


# =============================================================================
# CONVERSION FROM A TO BISTABLE WIDTH
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
# FIND REPEATING PATTERNS
# All elements in A should be positive numbers
# =============================================================================
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
# REMOVE TRANSIENT SOLUTION FROM OSCILLATIONS
# =============================================================================
def remove_transient(Sol,t_v,n_min=3):
    dSol = Sol[1:]-Sol[0:-1]
    dSol[dSol == 0] = 1e-12
    t_min = t_v[1:-1][(dSol[0:-1]*dSol[1:] < 0) & (dSol[0:-1] < 0)]
    if len(t_min) != 0:
        t_start = t_min[n_min]
        Sol_new = Sol[t_v > t_start]
        T_new = t_v[t_v > t_start]
    else:
        Sol_new = Sol
        T_new = t_v     
    return(Sol_new,T_new)


# =============================================================================
# LOAD AND CONVERT DATA FROM PARAMETER SCANS
# =============================================================================
# Bistable Xi or Piecewise approximation 
def find_csv_bistability(n,r,prot,xi):
    path = os.path.dirname(__file__)+'/Screens/Bistability/'+str(xi)+'/Final/' 
    csv_list = [i for i in os.listdir(path) if (i.endswith('.csv')) and ('Screen_Bistability_'+str(xi)+'_'+str(prot)+'_a_c_n_'+str(n)+'_r_'+str(r)+'_' in i)]     
    return(path,csv_list)        

def csv_to_array_bistability(CSV,path,xi,r,n):
    # CSV = list with csv files 
    # Check if only one file with the unique file name is found
    if len(CSV) == 1:
        csv = CSV[0] 
        Array = pd.read_csv(path+csv,index_col=0)
        C_v = Array['c'].unique()
        A_v = Array['a'].unique()
        # X-axis = c, Y-axis = a; data from last column           
        Period_Grid = np.array([np.array(Array.loc[Array['a']==j,list(Array.columns)[-2]]) for j in A_v])
        Amplitude_Grid = np.array([np.array(Array.loc[Array['a']==j,list(Array.columns)[-1]]) for j in A_v])
        SS_Grid = np.array([np.array(Array.loc[Array['a']==j,list(Array.columns)[-4]]) for j in A_v])
        SS_Pos_Grid = np.array([np.array(Array.loc[Array['a']==j,list(Array.columns)[-3]]) for j in A_v])
         
        # Adapt a to width of bistable region
        W_v = [] 
        if xi == 'Cubic':
            for a in A_v:
                W_v.append(width_cubic(r,n,a)[4])  
        elif xi == 'Piecewise':
            for a in A_v:
                x_max = (1+r-np.sqrt((1+r)**2-3*r))/3
                xi_max = (1 + a*x_max*(x_max - 1)*(x_max - r))
                x_min = (1+r+np.sqrt((1+r)**2-3*r))/3
                xi_min = (1 + a*x_min*(x_min - 1)*(x_min - r))
                # For large n, the width can be approximated well by the difference between the extrema of Xi
                W_v.append(xi_max-xi_min)  

    return(Array,Period_Grid,Amplitude_Grid,SS_Grid,SS_Pos_Grid,A_v,C_v,W_v)


# Cubic bistability in combination with delay
def find_csv_delay_bistability(n,r,c,prot):
    path = os.path.dirname(__file__)+'/Screens/Delay_Bistability/Final/' 
    csv_list = [i for i in os.listdir(path) if (i.endswith('.csv')) and ('Screen_Delay_Bistability_Cubic_'+str(prot)+'_tau_a_c_'+str(c)+'_n_'+str(n)+'_r_'+str(r)+'_' in i)]   
    return(path,csv_list)        

def csv_to_array_delay_bistability(CSV,path,r,n):
    # csv = list with csv files (output from find_csv_delay_bistability)
    if len(CSV) == 1:
        csv = CSV[0] 
        Array = pd.read_csv(path+csv,index_col=0)
        Tau_v = Array['tau'].unique()
        A_v = Array['a'].unique()
        # X-axis = a, Y-axis = tau; data from last column           
        Period_Grid = np.array([np.array(Array.loc[Array['tau']==j,list(Array.columns)[-2]]) for j in Tau_v])
        Amplitude_Grid = np.array([np.array(Array.loc[Array['tau']==j,list(Array.columns)[-1]]) for j in Tau_v])
        
        # Adapt a to width of cubic bistable region
        W_v = []   
        for a in A_v:
            W_v.append(width_cubic(r,n,a)[4])  
                
    return(Array,Period_Grid,Amplitude_Grid,A_v,Tau_v,W_v)


# Two cubic bistable switches combined: width of Cdk-APC switch vs relative synthesis c
def find_csv_series_bistability_aapc_c(a_cdk,prot):
    path = os.path.dirname(__file__)+'/Screens/Series/Final/' 
    csv_list = [i for i in os.listdir(path) if (i.endswith('.csv')) and ('Screen_Series_Bistability_Cubic_'+str(prot)+'_aapc_c_acdk_'+str(a_cdk)+'_' in i)]   
    return(path,csv_list)        

def csv_to_array_series_bistability_aapc_c(CSV,path,r,n):
    # csv = list with csv files (output from find_csv_series_bistability_aapc_c)
    if len(CSV) == 1:
        csv = CSV[0] 
        Array = pd.read_csv(path+csv,index_col=0)
        A_apc_v = Array['a_apc'].unique()
        C_v = Array['c'].unique()
        # X-axis = c, Y-axis = a_apc; data from last column
        Period_Grid = np.array([np.array(Array.loc[Array['a_apc']==j,list(Array.columns)[-2]]) for j in A_apc_v])
        Amplitude_Grid = np.array([np.array(Array.loc[Array['a_apc']==j,list(Array.columns)[-1]]) for j in A_apc_v])
        
        # Adapt a to width of bistable region
        W_cdk_v = []  
        for a_apc in A_apc_v:
            W_cdk_v.append(width_cubic(r,n,a_apc)[4])  
               
    return(Array,Period_Grid,Amplitude_Grid,A_apc_v,C_v,W_cdk_v)


# Two cubic bistable switches combined: effect of changing threshold K
def find_csv_series_bistability_K(prot,a_apc,a_cdk):
    path = os.path.dirname(__file__)+'/Screens/Series/Final/' 
    csv_list = [i for i in os.listdir(path) if (i.endswith('.csv')) and ('Screen_Series_Bistability_Cubic_'+str(prot)+'_K_aapc_'+str(a_apc)+'_acdk_'+str(a_cdk)+'_' in i)]   
    return(path,csv_list)        

def csv_to_array_series_bistability_K(CSV,path):
    # csv = list with csv files (output from find_csv_series_bistability_K)
    if len(CSV) == 1:
        csv = CSV[0] 
        Array = pd.read_csv(path+csv,index_col=0)
        K_cyc_v = Array['K_cyc'].unique()
        K_cdk_v = Array['K_cdk'].unique()
        # X-axis = K_cyc, Y-axis = K_cdk; data from last column
        Period_Grid = np.array([np.array(Array.loc[Array['K_cdk']==j,list(Array.columns)[-2]]) for j in K_cdk_v])
        Amplitude_Grid = np.array([np.array(Array.loc[Array['K_cdk']==j,list(Array.columns)[-1]]) for j in K_cdk_v])
        
        # W_cyc = np.median(Array['W_cyc'])
        # W_cdk = np.median(Array['W_cdk'])
        W_cyc_v = Array['W_cyc']
        W_cdk_v = Array['W_cdk']

    return(Array,Period_Grid,Amplitude_Grid,K_cyc_v,K_cdk_v,W_cyc_v,W_cdk_v)



def find_csv_chain(x,y):
    path = os.path.dirname(__file__)+'/Screens/Chain/Final/' 
    csv_list = [i for i in os.listdir(path) if (i.endswith('.csv')) and ('Screen_Chain_'+str(x)+'_'+str(y)+'_' in i)]   
    return(path,csv_list)        

def csv_to_array_chain(CSV,path,x,y):
    if len(CSV) == 1:
        csv = CSV[0] 
        Array = pd.read_csv(path+csv,index_col=0)
        X_v = Array[x].unique()
        Y_v = Array[y].unique()
        
        Period_Grid = np.array([np.array(Array.loc[Array[y]==j,list(Array.columns)[3]]) for j in Y_v])
        G1_Grid = np.array([np.array(Array.loc[Array[y]==j,list(Array.columns)[4]]) for j in Y_v])
        SG2_Grid = np.array([np.array(Array.loc[Array[y]==j,list(Array.columns)[5]]) for j in Y_v])
        M_Grid = np.array([np.array(Array.loc[Array[y]==j,list(Array.columns)[6]]) for j in Y_v])
        E2F_Grid = np.array([np.array(Array.loc[Array[y]==j,list(Array.columns)[7]]) for j in Y_v])
        APC_Grid = np.array([np.array(Array.loc[Array[y]==j,list(Array.columns)[8]]) for j in Y_v])              

        if (x == 'acdk') and (y == 'ae2f'):
            W_cyccdk_v = []  
            for a in X_v:
                W_cyccdk_v.append(width_cubic(0.5,15,a)[4]) # c=0.5, n=15
            W_cyce2f_v = []  
            for a in Y_v:
                W_cyce2f_v.append(width_cubic(0.5,15,a)[4]) 
            out = Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,X_v,Y_v,W_cyccdk_v,W_cyce2f_v
        elif (x != 'acdk') and (y == 'ae2f'):
            W_cyce2f_v = []  
            for a in Y_v:
                W_cyce2f_v.append(width_cubic(0.5,15,a)[4]) 
            out = Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,X_v,Y_v,W_cyce2f_v
        elif (y == 'acdk') and (x != 'ae2f'):
            W_cyccdk_v = []  
            for a in Y_v:
                W_cyccdk_v.append(width_cubic(0.5,15,a)[4]) 
            out = Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,X_v,Y_v,W_cyccdk_v
        else:
            out = Array,Period_Grid,G1_Grid,SG2_Grid,M_Grid,E2F_Grid,APC_Grid,X_v,Y_v
    return(out)



def find_csv_chain_dna_damage(G):
    path = os.path.dirname(__file__)+'/Screens/Chain/Final/' 
    csv_list = [i for i in os.listdir(path) if (i.endswith('.csv')) and ('Screen_Chain_DNA_Damage_'+str(G) in i)]   
    return(path,csv_list)        

def csv_to_array_chain_dna_damage(CSV,path,G):
    # csv = list with csv files (output from find_csv_chain_dna_damage)
    if len(CSV) == 1:
        csv = CSV[0] 
        Array = pd.read_csv(path+csv,index_col=0)
        if G == 'G1':
            t_g_v = Array['t_g1'].unique()
            delta_t_g_v = Array['delta_t_g1'].unique()
        elif G == 'G2':
            t_g_v = Array['t_g2'].unique()
            delta_t_g_v = Array['delta_t_g2'].unique()

        # X-axis = t_in, Y-axis = delta_t; data from last column
        if G == 'G1':
            Delay_M_Grid = np.array([np.array(Array.loc[Array['delta_t_g1']==j,list(Array.columns)[-1]]) for j in delta_t_g_v])
        elif G == 'G2':
            Delay_M_Grid = np.array([np.array(Array.loc[Array['delta_t_g2']==j,list(Array.columns)[-1]]) for j in delta_t_g_v])
              
    return(Array,Delay_M_Grid,t_g_v,delta_t_g_v)



def find_csv_delay(n,prot):
    path = os.path.dirname(__file__)+'/Screens/Delay/Final/' 
    csv_list = [i for i in os.listdir(path) if (i.endswith('.csv')) and ('Screen_Delay_'+str(prot)+'_tau_c_n_'+str(n)+'_' in i)]   
    return(path,csv_list)        

def csv_to_array_delay(CSV,path):
    # csv = list with csv files (output from find_csv_delay)
    if len(CSV) == 1:
        csv = CSV[0] 
        Array = pd.read_csv(path+csv,index_col=0)
        C_v = Array['c'].unique()
        Tau_v = Array['tau'].unique()
        # X-axis = c, Y-axis = tau; data from last column           
        Period_Grid = np.array([np.array(Array.loc[Array['tau']==j,list(Array.columns)[-2]]) for j in Tau_v])
        Amplitude_Grid = np.array([np.array(Array.loc[Array['tau']==j,list(Array.columns)[-1]]) for j in Tau_v])
                
    return(Array,Period_Grid,Amplitude_Grid,Tau_v,C_v)


def find_csv_arnold():
    path = os.path.dirname(__file__)+'/Screens/Arnold_Tongues/Final/' 
    # Screening results for Arnold tongues are divided over separate files
    csv_list = np.sort([i for i in os.listdir(path) if (i.endswith('.csv'))])   
    return(path,csv_list) 

def csv_to_array_arnold(CSV,path):
    # Multiple files were generated during screen, so concatenate 
    A_a_v = [] 
    Locking_Grid = [] 
    Arrays = []
    for i in CSV:
        Array = pd.read_csv(path+i,index_col=0)
        Arrays.append(Array)
        omega_v = Array['Omega'].unique()
        A_a = Array['A_a'].unique()
        A_a_v.append(A_a)
        # Convert dataframe columns to 2D array; X-axis = omega, Y-axis = A_a; data from last column
        Locking_Grid.append(np.array([np.array(Array.loc[Array['A_a']==j,list(Array.columns)[-1]]) for j in A_a]))
    
    A_a_v = np.concatenate(A_a_v)
    Locking_Grid = np.vstack(Locking_Grid)
    
    return(Arrays,Locking_Grid,omega_v,A_a_v)

