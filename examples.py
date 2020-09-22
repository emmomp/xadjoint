#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example uses of xadjoint experiment class
Created on Thu Sep  3 12:06:05 2020

@author: emmomp
"""
import xadjoint as xad
import matplotlib.pyplot as plt
import ecco_v4_py as ecco
import xarray as xr
 
rootdir = '/data/smurphs/emmomp/orchestra/'
griddir = rootdir+'grid2/'

expdir = rootdir+'experiments/run_ad.8yr.SOpv3.00.atl/'
startdate='1993-01-01'
lag0='2000-07-01'
         
myexp = xad.Experiment(griddir,expdir,start_date=startdate,lag0=lag0)

#myexp = Exp('smurphs','run_ad.CORE2.5yr.1mosssrelax_k500_mergesss')
#myexp.find_results()
myexp.load_vars(['ADJqnet','adxx_tauu','adxx_tauv','adxx_qnet','ADJsalt'])
print(myexp)
myexp.load_vars(['adxx_tauv',])

# load grid and plot via ecco-v4-python 

# If you have one of the grid netcdfs open it here
grid_ds=xr.open_dataset(griddir+'ECCOv4r3_grid_with_masks.nc')
# or if you only have grid meta/data files use xmitgcm
#grid_ds = xmitgcm.open_mdsdataset(iters=None,read_grid=True,geometry='llc',data_dir=expdir,grid_dir=griddir)

tmp_plt = myexp.data.ADJqnet.isel(time=1)

plt.figure(figsize=(12,6), dpi= 90)
ecco.plot_proj_to_latlon_grid(grid_ds.XC, \
                              grid_ds.YC, \
                              tmp_plt, \
                              plot_type = 'pcolormesh', \
                              dx=2,\
                              dy=2, \
                              projection_type = 'robin',
                              less_output = True);
plt.show()

# Write to netcdf
myexp.to_nctiles(['ADJqnet',])


