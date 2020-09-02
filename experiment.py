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
from inputs import adxx_it
from inputs import adj_dict
import xarray as xr
import ecco_v4_python as ecco

class Exp(object):
    '''
    Representation of specific MITgcm adjoint experiment run in ECCOv4
    '''
    def __init__(self,grid_dir,exp_dir,start_date,lag0,deltat=3600.):
        '''
        Initialise Exp object based on user data

        Parameters
        ----------
        grid_dir : string
            Location of grid data.
        exp_dir : string
            Location of adjoint sensitivities.
        start_date : string
            Start date of simulation in 'YYYY-MM-DD' format.
        lag0 : string
            Lag 0 for cost function definedin *_maskT file
        deltat : double, optional
            Time step of forward model in seconds. The default is 3600. (one day).        

        '''
        
        #Assign directories
        self.grid_dir = grid_dir
        self.exp_dir = exp_dir      

        #Assign time info                  
        self.start_date=np.datetime64(start_date)
        self.lag0=np.datetime64(lag0)
        self.deltat=deltat
        
        # Generate various time dimensions
        self.time_data=_get_time_data(self.exp_dir,self.start_date,self.lag0,self.deltat)
        
    
    # Look for ADJ and adxx files
    def find_results(self):
        '''
        Finds and prints all ADJ and adxx in experiment,
        adds them to lists self.ADJ_vars and self.adxx_vars.

        '''
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
        alladxx = [os.path.basename(x) for x in glob.glob(self.exp_dir+'/adxx_*'+'{010.0f}'.format(adxx_it)+'.meta')]
        for item in alladxx:
            #i1 = item.find('adxx')
            i2 = item.find('.')
            all_vars.append(item[:i2])
        varset = set(all_vars)
        self.adxx_vars=list(varset)
        del alladxx,varset
        print('Found {:d} adxx variables'.format(len(self.adxx_vars)))
        
    def load_all(self):
        '''
        Loads all ADJ and adxx variables found in experiment.

        '''
        if not hasattr(self,'ADJ_vars'):
            self.find_results()
        self.load_vars([*self.ADJ_vars,*self.adxx_vars])    
        
    # Load adjoint files (assumes nz=1 for adxx vars)    
    def load_vars(self,var_list='ALL'):
        '''
        Load user specified list of variables into xarray DataSet.

        Parameters
        ----------
        var_list : list of strings. Default 'ALL'
            Names of data variables to be loaded. If 'ALL', all will be loaded

        Returns
        -------
        self.data is an xarray DataSet with variables loaded.

        '''
        if var_list=='ALL':
            self.load_all()
        else:       
            # Check if find_results has been run
            if not hasattr(self,'ADJ_vars'):
                self.find_results()
            # Loop through and read variables
            datasets = []
            for var in var_list:
                print('Reading in '+var)
                dims=_parse_vartype(adj_dict[var]['vartype'])
                
                if adj_dict[var]['adjtype'] == 'ADJ':
                    
                    if dims is not None:
                        extra_variable=dict(dims=dims,attrs=dict(standard_name=var,long_name='Sensitivity to '+adj_dict[var]['longname'],units='[J]/'+adj_dict[var]['units']))
                        var_ds= xmitgcm.open_mdsdataset(data_dir=self.exp_dir,grid_dir=self.root_dir+mydirs['grid_dir'],
                                                    prefix=[var,],geometry='llc',delta_t=self.deltat,ref_date=self.start_date,
                                                    extra_variable=extra_variable,read_grid=False)
                    else:
                        var_ds= xmitgcm.open_mdsdataset(data_dir=self.exp_dir,grid_dir=self.root_dir+mydirs['grid_dir'],
                                                    prefix=[var,],geometry='llc',delta_t=self.deltat,ref_date=self.start_date,
                                                    read_grid=False)                    
                    datasets.append(var_ds)
                    del var_ds
                elif var in self.adxx_vars:
                    var_data= xmitgcm.utils.read_3d_llc_data(fname=self.exp_dir+'/'+var+'.0000000012.data',nz=1,nx=90,nrecs=self.nits,dtype='>f4') 
                    if adj_dict[var]['ndims']==3:
                        var_ds=ecco.llc_tiles_to_xda(var_data, var_type=adj_dict[var]['vartype'],dim4='depth', dim5='time')
                    elif adj_dict[var]['ndims']==2:
                        var_ds=ecco.llc_tiles_to_xda(var_data, var_type=adj_dict[var]['vartype'],dim4='time')
                    else:
                        raise ValueError('Ndims of variables must be 2 or 3')
                    #grid_ds = xmitgcm.open_mdsdataset(iters=None, read_grid=True,geometry='llc',prefix=var,data_dir=self.exp_dir,grid_dir=self.root_dir+mydirs['grid_dir'])
                    #var_1 = xmitgcm.open_mdsdataset(data_dir=self.exp_dir,grid_dir=self.root_dir+mydirs['grid_dir'],prefix=[var,],geometry='llc')
                    #vardims = var_1[var].dims
                    #newcords = {k: grid_ds[k] for k in vardims[1:]} #exclude time here
                    #newcords['time']=self.its
                    #var_ds = xr.Dataset(data_vars={var:(vardims,var_data)},coords=newcords)
                    #var_ds = xr.combine_by_coords([grid_ds,var_ds])
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
            
def _get_time_data(exp_dir,start_date,lag0,deltat) :   
    
    tdata={}
    with open(exp_dir+'/its_ad.txt') as f:
        itin = f.readlines()
    nits=len(itin)
    tdata['nits'] = nits   
    tdata['its'] = np.zeros(nits)
    tdata['dates']=np.empty(nits,dtype='datetime64[D]')
    tdata['lag_days']=np.empty(nits)
    tdata['lag_years']=np.empty(nits)
    for i in range(nits):
        tdata['its'][i]=int(itin[i])
        tdata['dates'][i]=start_date+np.timedelta64(int(itin[i])*deltat,'s')
        tdata['lag_days'][i]=(tdata['dates'][i]-lag0)/np.timedelta64(1,'D')
        tdata['lag_years'][i]=tdata['lag_days'][i]/365.25
    del itin,nits
    return tdata     

def _parse_vartype(vartype):

    if vartype=='c':
        if adj_dict[var]['ndims']==3:
            dims=['k','j','i']
        elif adj_dict[var]['ndims']==2:
            dims=['j','i']
        else:
            raise ValueError('Ndims of variables must be 2 or 3')
    elif vartype=='w':
        if adj_dict[var]['ndims']==3:
            dims=['k','j','i_g']
        elif adj_dict[var]['ndims']==2:
            dims=['j','i_g']
        else:
            raise ValueError('Ndims of variables must be 2 or 3')
    elif vartype=='s':
        if adj_dict[var]['ndims']==3:
            dims=['k','j_g','i']
        elif adj_dict[var]['ndims']==2:
            dims=['j_g','i']
        else:
            raise ValueError('Ndims of variables must be 2 or 3')
    elif vartype=='z':
        if adj_dict[var]['ndims']==3:
            dims=['k','j_g','i_g']
        elif adj_dict[var]['ndims']==2:
            dims=['j_g','i_g']
        else:
            raise ValueError('Ndims of variables must be 2 or 3')
    else:
        print('No vartype found, variable must be defined in available_diagnostics.log')
        dims=[]
    return dims
        
# Tests   
rootdir = '/data/emmomp/orchestra/'
griddir = rootdir+'grid2/'

expdir = rootdir+'experiments/run_ad.8yr.SOpv3.00.atl/'
startdate='1993-01-01'
lag0='2000-07-01'
         
myexp = Exp(griddir,expdir,start_date=startdate,lag0=lag0)
print(vars(myexp))

#myexp = Exp('smurphs','run_ad.CORE2.5yr.1mosssrelax_k500_mergesss')
#myexp.find_results()
myexp.load_vars(['ADJqnet','adxx_tauu'])
#myexp.load_vars(['adxx_tauu',])
