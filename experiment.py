#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:35:41 2019

@author: emmomp
"""
import os
import numpy as np
import glob
import xmitgcm
from inputs import mydirs
import xarray as xr

class Exp(object):
    def __init__(self,rootdir,exp_dir,start_date,lag0):
        self.root_dir = rootdir
        # assign expt directory
        self.exp_dir = rootdir+mydirs['exp_dirs']+exp_dir
            # Check experiment dir exists
        if not os.path.isdir(self.exp_dir):
            raise ValueError('Experiment directory '+self.exp_dir+' not found')
                        
        self.start_date=start_date
        self.lag0=lag0
        
        # Find its
        with open(self.exp_dir+'/its_ad.txt') as f:
            itin = f.readlines()
        self.nits = len(itin)    
        self.its = np.zeros(self.nits)
        self.dates=np.empty(self.nits,dtype='datetime64[D]')
        self.lag_days=np.empty(self.nits)
        self.lag_years=np.empty(self.nits)
        for i in range(self.nits):
            self.its[i]=int(itin[i])
            self.dates[i]=start_date+np.timedelta64(int(self.its[i]/24),'D')
            self.lag_days[i]=(self.dates[i]-lag0)/np.timedelta64(1,'D')
            self.lag_years[i]=self.lag_days[i]/365.25
        del itin             
        
    
    # Look for ADJ and adxx files
    def find_results(self):
        # find all ADJ meta files at first it
        self.ADJ_vars=[]
        allADJ = [os.path.basename(x) for x in glob.glob(self.exp_dir+'/ADJ*'+'{:010.0f}'.format(self.its[0])+'.meta')]
        for item in allADJ:
            #i1 = item.find('ADJ')
            i2 = item.find('.')
            self.ADJ_vars.append(item[:i2])
        del allADJ
        print('Found {:d} ADJ variables'.format(len(self.ADJ_vars)))
        
        # find all adxx meta files
        all_vars=[]
        alladxx = [os.path.basename(x) for x in glob.glob(self.exp_dir+'/adxx*012.meta')]
        for item in alladxx:
            #i1 = item.find('adxx')
            i2 = item.find('.')
            all_vars.append(item[:i2])
        varset = set(all_vars)
        self.adxx_vars=list(varset)
        del alladxx,varset
        print('Found {:d} adxx variables'.format(len(self.adxx_vars)))
        
        
    def load_all(self):
    #Loads all found variables
        if not hasattr(self,'ADJ_vars'):
            self.find_results()
        self.load_vars([*self.ADJ_vars,*self.adxx_vars])    
        
    # Load adjoint files (assumes nz=1 for adxx vars)    
    def load_vars(self,var_list):
        # Check if find_results has been run
        if not hasattr(self,'ADJ_vars'):
            self.find_results()
        # Loop through and read variables
        datasets = []
        for var in var_list:
            print('Reading in '+var)
            if var in self.ADJ_vars:
                var_ds= xmitgcm.open_mdsdataset(data_dir=self.exp_dir,grid_dir=self.root_dir+mydirs['grid_dir'],prefix=[var,],geometry='llc')
                datasets.append(var_ds)
                del var_ds
            elif var in self.adxx_vars:
                var_data= xmitgcm.utils.read_3d_llc_data(fname=self.exp_dir+'/'+var+'.0000000012.data',nz=1,nx=90,nrecs=self.nits,dtype='>f4') 
                grid_ds = xmitgcm.open_mdsdataset(iters=None, read_grid=True,geometry='llc',prefix=var,data_dir=self.exp_dir,grid_dir=self.root_dir+mydirs['grid_dir'])
                var_1 = xmitgcm.open_mdsdataset(data_dir=self.exp_dir,grid_dir=self.root_dir+mydirs['grid_dir'],prefix=[var,],geometry='llc')
                vardims = var_1[var].dims
                newcords = {k: grid_ds[k] for k in vardims[1:]} #exclude time here
                newcords['time']=self.its
                var_ds = xr.Dataset(data_vars={var:(vardims,var_data)},coords=newcords)
                var_ds = xr.combine_by_coords([grid_ds,var_ds])
                datasets.append(var_ds)
                del var_ds,var_data,grid_ds,newcords,vardims,var_1
            else:
                print('variable '+var+' not found in '+self.exp_dir)
        
        # At to existing data or create new attr
        if hasattr(self,'data'):
            self.data = xr.combine_by_coords([self.data,]+datasets)
        else:
            self.data = xr.combine_by_coords(datasets)
        del datasets
    
    # Calculate stats with optional sigma multiplier    
    def calc_stats(self,sigma=None,sigma_type=None): 
        # sigma should be dictionary with keys equal to variable names
        # sigma_type should be 1D or 3D, if sigma provided
        
        # Check if data exists
        if not hasattr(self,'data'):
            raise AttributeError('No data found in this experiment - run .load_vars first')
        # Check what sigma we have
        if sigma is None:
            print('No sigma provided - raw sensitivity stats only')
        elif type(sigma) is dict:
            print('Using provided sigma dictionary for multiplier')
            if sigma_type == '1D':
                if type(list(sigma.values())[0]) is float: 
                    print('Found 1D sigmas')
                else:
                    raise TypeError('sigma dict should contain floats')
            elif sigma_type == '3D':
                if type(list(sigma.values())[0]) is str:
                    print('Found 3D sigmas')
                    
                    #TO DO: ADD STATS
                    
                     #exfqnet = xr.open_mfdataset(glob.glob('EXFqnet*nc'),concat_dim='tile')  
                    
                else:
                    raise TypeError('sigma dict should contain filenames')
            else:
                raise ValueError('sigma_type should be 1D or 3D')
        else:
            raise TypeError('sigma should be a dictionary')
        
# Tests            
#myexp = Exp('/data/smurphs/emmomp/orchestra/','run_ad.8yr.SOpv.00.atl',start_date=np.datetime64('1993-01-01'),lag0=np.datetime64('2000-07-01'))
#print(vars(myexp))
#myexp = Exp('murphs','run')
#myexp = Exp('smurphs','run_ad.CORE2.5yr.1mosssrelax_k500_mergesss')
#myexp.find_results()
#myexp.load_vars(['ADJqnet','adxx_tauu'])
#myexp.load_vars(['adxx_tauu',])