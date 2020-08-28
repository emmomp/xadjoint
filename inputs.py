#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:35:26 2019

@author: emmomp
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

# Scalar ECCO standard deviations
keySet = ['ADJtheta',
          'ADJsalt',
          'adxx_empmr',
          'adxx_qnet',
          'adxx_tauu',
          'adxx_tauv',
           'adxx_uwind', #From https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008JD011696
           'adxx_vwind',
          'NONE']
valueSet = [0.3,
            0.07,
            2.0e-08,
            60.0,
            0.08,
            0.06,
            1.6,
            1.6,
            1.0]
FSigSingle=dict(zip(keySet,valueSet))

# 3-D ECCO standard deviation filenames     
keySet = ['ADJtheta',
          'ADJsalt',
          'adxx_empmr',
          'adxx_qnet',
          'adxx_tauu',
          'adxx_tauv',
           'adxx_uwind', 
           'adxx_vwind']
valueSet=['THETA','SALT','EXFempmr','EXFqnet','EXFtaue','EXFtaun','EXFuwind','EXFvwind']
FSig3D=dict(zip(keySet,valueSet))

# CMIP filenames & varnames    
CMIPnames = {'ADJtheta':['consist_cmip_sst_std.nc','thetao'],
          'ADJsalt':['consist_cmip_sss_std.nc','so'],
          'adxx_qnet':['consist_cmip_net_std.nc','hfds'],
          'adxx_tauu':['consist_cmip_taux_std.nc','tauu'],
          'adxx_tauv':['consist_cmip_tauy_std.nc','tauv']}

varlabs = {'ADJtheta':'$\Theta$',
          'ADJsalt':'S',
          'adxx_empmr':'E-P-R',
          'adxx_qnet':'$Q_{net}$',
          'adxx_tauu':'$\\tau_U$',
          'adxx_tauv':'$\\tau_V$',
           'adxx_uwind':'$U_{wind}$', 
           'adxx_vwind':'$V_{wind}$'}