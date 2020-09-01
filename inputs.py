#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:35:26 2019

@author: emmomp

Define directories for inputs and outputs
"""
mydirs = \
    {
     # Where grid data can be found
     'grid_dir':'../orchestra/grid2/',
     # Where experiment folders can be found
     'exp_dirs':'experiments/',
     # Where plots saved
     'plot_dir':'plots/',
     # Where stats saved
     'stats_dir':'data_out/',
     # Where 3d stdev fields are binary llc format
     '3dstdev_dir': '../orchestra/other_data/ecco_stdvs_anoms/'
    }
# Scaling for ADJ files (duration of one ctrl period / duration of one timestep) 
ADJ_scale=1209600/3600;
# Cost function scaling
fc_scale=1e9;

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
    'vartype' : One of 'c','w','s' or 'z' to describe grid location of variable. 
                'c' is a central variable like temperature, tracer
                'w' is a west variable like zonal velocity
                's' is a south variable like meridional velocity
                'z' is a corner variable like vorticity
                See help(ecco.llc_tiles_to_xda) for more info
    'ndims': Number of dimensions of root variable. E.g. 2 for surface fields, 3 for full depth fields.
    'ECCOname': OPTIONAL Name of root variable in ECCO, used to copy dimensions of field
                If not defined, you must add details of the variable to your copy of available_diagnostics.log
    
Optional elements:    
    'longname' : Descriptive name of root variable.
    'units' : Units of root variable.
    'fact' : Scalar factor to multiply by when calculating stats.


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
          'sig0':0.3,
          },
         
     'ADJsalt':
         {'adjtype':'ADJ',
          'varlabel':'$S$',
          'ECCOname':'SALT',
          'vartype':'c',
          'ndims':3,
          'longname':'Salinity',
          'units':'psu',
          'sig0':0.07,
          },
         
     'adxx_empmr':
         {'adjtype':'adxx',
          'varlabel':'E-P-R',
          'ECCOname':'EXFempmr',
          'vartype':'c',
          'ndims':2,
          'longname':'Evaporation-Precipitation-Runoff',
          'units':'m/s',
          'sig0':2.0e-8,
          },
         
     'adxx_qnet':
         {'adjtype':'adxx',
          'varlabel':'$Q_{net}$',
          'ECCOname':'EXFqnet',
          'vartype':'c',
          'ndims':2,
          'longname':'Net Heat Flux',
          'units':'W/m^2',
          'sig0':60.,
          },
         
     'adxx_tauu':
         {'adjtype':'adxx',
          'varlabel':'$\\tau_U$',
          'ECCOname':'EXFtaue',
          'vartype':'w',
          'ndims':2,
          'longname':'Zonal Wind Stress',
          'units':'N/m^2',
          'sig0':0.08,
          },
         
     'adxx_tauv':
         {'adjtype':'adxx',
          'varlabel':'$\\tau_V$',
          'ECCOname':'EXFtaun',
          'vartype':'s',
          'ndims':2,
          'longname':'Meridional Wind Stress',
          'units':'N/m^2',
          'sig0':0.06,
          },
         
     'adxx_uwind':
         {'adjtype':'adxx',
          'varlabel':'$U_{wind}$',
          'ECCOname':'EXFuwind',
          'vartype':'w',
          'ndims':2,
          'longname':'Zonal Wind',
          'units':'m/s',
          'sig0':1.6,
          },
         
     'adxx_vwind':
         {'adjtype':'adxx',
          'varlabel':'$V_{wind}$',
          'ECCOname':'EXFvwind',
          'vartype':'s',
          'ndims':2,
          'longname':'Meridional Wind',
          'units':'m/s',
          'sig0':1.6,
          }     
     
     }


