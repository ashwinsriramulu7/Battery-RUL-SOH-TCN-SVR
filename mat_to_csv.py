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
    Convert a MATLAB datevec [Y M D h m s] into a Python datetime object.
    Fractional seconds, if present, are truncated to achieve second-level resolution.
    """
    if vec is None:
        return None

    try:
        Year, Month, Day, hour, minute, second = vec
        return datetime(
            int(Year),
            int(Month),
            int(Day),
            int(hour),
            int(minute),
            int(second)
        )
    except Exception:
        return None


