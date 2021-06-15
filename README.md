# A modular approach for modeling the cell cycle based on functional response curves’
This folder contains the code and data to reproduce the figures in the paper ‘A modular approach for modeling the cell cycle based on functional response curves’.

* Delay_Bistability_Analysis_Final.py contains various functions used for generating the figures (e.g. determination of period and amplitude of oscillator, definitions of the ODE systems, and functions to load the csv files containing the data from parameter scans).

* Figure_X.py containes the code that produces the figures as they appear in the paper.

* The 'Screens' folder contains the data from the different parameter scans, as well as the scripts to run them.  

Dependencies: all the code was written and tested using Spyder4. Running the code requires numpy, pandas, matplotlib, scipy, and JITCDDE. Documentation for the JITCDDE package can be found on: https://jitcde-common.readthedocs.io/en/stable/
