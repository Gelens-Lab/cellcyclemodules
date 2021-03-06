# A modular approach for modeling the cell cycle based on functional response curves’
This folder contains the code and data to reproduce the figures in the paper ‘A modular approach for modeling the cell cycle based on functional response curves’.

* Delay_Bistability_Analysis_Final.py contains various functions used for generating the figures (e.g. determination of period and amplitude of oscillator, definitions of the ODE systems, and functions to load the csv files containing the data from parameter scans).

* Figure_X.py contains the code that produces the figures as they appear in the paper.

* The 'Screens' folder contains the data from the different parameter scans, as well as the scripts to run them.  

To reproduce the figures, download all the code as a single zip file, extract, and run the Figure_X.py files from within the extracted folder in Spyder.

Dependencies: all the code was written and tested using Spyder4. Running the code requires numpy, pandas, matplotlib, scipy, and JITCDDE. Documentation for the JITCDDE package can be found on: https://jitcde-common.readthedocs.io/en/stable/
