"""
The dataset used in this project is the NASA Battery Dataset, created and made available \
online by the NASA Ames Research Center. It consists of charge/discharge data for different \
batteries, in .mat format. The first step is to convert the .mat files to \
.csv file format for Exploratory Data Analysis and Downstream processing before actually training ML models. 
"""

import scipy.io as sio # for reading the .mat data
import pandas as pd # to create and manipulate the dataframes
import numpy as np # for reading, creating and manipulating arrays
from datetime import datetime # to add proper formatted timestamps for records

## <-> 
# Utility functions 
## <-> 

##Util001 - convert matlab datevec to python datetime
def matlab_dv_to_pyth_dt(vec):
    """
    Convert MATLAB datevec [Y M D h m s] to Python datetime format
    """
    if vec is None: #empty datevec 
        return None
    try:
        Y, M, D, h, m, s = vec 
        return datetime(int(y), int(m), int(d), int(H), int(M), int(S))
    except:
        return None

