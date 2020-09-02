#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:35:26 2019

@author: emmomp

"""

# Scaling for ADJ files (duration of one ctrl period / duration of one timestep) 
ADJ_scale=1209600./3600.;
# Cost function scaling (mult_gencost in data.ecco). Set to 1 if not used.
fc_scale=1.0e9;
# adxx iter number. Iteration label for adxx files. i.e. expects files of the form adxx_[var].0000000012.data/meta
adxx_it=12;
'''
Define information for each adjoint sensitivity field to be used. Here the 
'sensitivity variable' refers to the sensitivity field, wheras the 'root variable'
is the variable the sensitivity is with respect to, 
i.e. [sensitivity variable]= dJ/d[root variable] where J is the user-defined
 objective function.

Standard elements: 
    Name in dictionary should be called the same as the first element of the filename, eg 'ADJtheta' or 'adxx_qnet'.
    'adjtype': Should be either ADJ [one field per timestep] or adxx [all timesteps in one file].
    'varlabel': Label for root variable, used in figures.
    'vartype' : Must be either grid location, list of dimensions, or [ADJ variables only] left empty if in available_diagnostics.log
                If location, must be one of:
                    'c' is a central variable like temperature, tracer
                    'w' is a west variable like zonal velocity
                    's' is a south variable like meridional velocity
                    'z' is a corner variable like vorticity
                    See help(ecco.llc_tiles_to_xda) for more info
                Or if a list must be 2D or 3D, and follow the ECCO_v4_py dimension labelling format:
                    e.g. ['k','j','i'] is a central 3D variable like a tracer
                    e.g. ['j_g','i'] is a south 2D variable like a surface meridional velocity
                    See https://ecco-v4-python-tutorial.readthedocs.io/ for more info  
                NB info in available_diagnostics.log will take precedence over vartype for ADJ variables
    'ndims': Number of dimensions of root variable. 2 for surface fields, 3 for full depth fields.                 
    
Optional elements:    
    'longname' : Descriptive name of root variable. 
    'units' : Units of root variable.
    'fact' : Scalar factor to multiply by when calculating stats.
    'ECCOname': Name of root variable in ECCO.  s

'''

adj_dict = \
    {
     'ADJtheta':
         {'adjtype':'ADJ',
          'varlabel':'$\Theta$',
          'ECCOname':'THETA',
          'vartype':'c',
          'ndims':3,
          'longname':'Potential Temperature',
          'units':'Degree C',
          'fact':0.3,
          },
         
     'ADJsalt':
         {'adjtype':'ADJ',
          'varlabel':'$S$',
          'ECCOname':'SALT',
          'vartype':'c',
          'ndims':3,
          'longname':'Salinity',
          'units':'psu',
          'fact':0.07,
          },
         
     'ADJaqh':
         {'adjtype':'ADJ',
          'varlabel':'$q$',
          'ECCOname':'EXFaqh',
          'vartype':['j','i'],
          'ndims':2,
          'longname':'Specific Humidity',
          'units':'kg/kg',
          'fact':1.,
          },
         
     'ADJqnet':
         {'adjtype':'ADJ',
          'varlabel':'$Q_{net}$',
          'ECCOname':'EXFqnet',
          'vartype':'c',
          'ndims':2,
          'longname':'Net Heat Flux',
          'units':'W/m^2',
          'fact':60.,
          },

     'ADJqnet':
         {'adjtype':'ADJ',
          'varlabel':'$Q_{net}$',
          'ECCOname':'EXFqnet',
          'vartype':['j','i'],
          'ndims':2,
          'longname':'Net Heat Flux',
          'units':'W/m^2',
          'sig0':60.,
          },
    
     'adxx_empmr':
         {'adjtype':'adxx',
          'varlabel':'E-P-R',
          'ECCOname':'EXFempmr',
          'vartype':'c',
          'ndims':2,
          'longname':'Evaporation-Precipitation-Runoff',
          'units':'m/s',
          'fact':2.0e-8,
          },
         
     'adxx_qnet':
         {'adjtype':'adxx',
          'varlabel':'$Q_{net}$',
          'ECCOname':'EXFqnet',
          'vartype':'c',
          'ndims':2,
          'longname':'Net Heat Flux',
          'units':'W/m^2',
          'fact':60.,
          },
         
     'adxx_tauu':
         {'adjtype':'adxx',
          'varlabel':'$\\tau_U$',
          'ECCOname':'EXFtaue',
          'vartype':'w',
#          'vartype':['j','i_g'],
          'ndims':2,
          'longname':'Zonal Wind Stress',
          'units':'N/m^2',
          'fact':0.08,
          },
         
     'adxx_tauv':
         {'adjtype':'adxx',
          'varlabel':'$\\tau_V$',
          'ECCOname':'EXFtaun',
          'vartype':'s',
          'ndims':2,
          'longname':'Meridional Wind Stress',
          'units':'N/m^2',
          'fact':0.06,
          },
         
     'adxx_uwind':
         {'adjtype':'adxx',
          'varlabel':'$U_{wind}$',
          'ECCOname':'EXFuwind',
          'vartype':'w',
          'ndims':2,
          'longname':'Zonal Wind',
          'units':'m/s',
          'fact':1.6,
          },
         
     'adxx_vwind':
         {'adjtype':'adxx',
          'varlabel':'$V_{wind}$',
          'ECCOname':'EXFvwind',
          'vartype':'s',
          'ndims':2,
          'longname':'Meridional Wind',
          'units':'m/s',
          'fact':1.6,
          }     
     
     }


