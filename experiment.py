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
import ecco_v4_py as ecco

class Experiment(object):
    '''
    Representation of specific MITgcm adjoint experiment run in ECCOv4
    '''
    def __init__(self,grid_dir,exp_dir,start_date,lag0,deltat=3600):
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
            Lag 0 for cost function defined in *_maskT file
        deltat : int, default 3600
            Time step of forward model in seconds. The default is 3600 (one day).        

        '''
        
        #Assign directories
        self.grid_dir = grid_dir
        self.exp_dir = exp_dir      

        #Assign time info                  
        self.start_date=np.datetime64(start_date)
        self.lag0=np.datetime64(lag0)
        self.deltat=int(deltat)
        
        # Generate various time dimensions
        self.time_data=_get_time_data(self.exp_dir,self.start_date,self.lag0,self.deltat)
        self._find_results()
        
    def __repr__(self):
        out_str = '<xadjoint.Experiment> \n Directories: \n\t experiment = {} \n\t grid = {}'.format(self.exp_dir,self.grid_dir)     
        out_str = out_str+'\n Time Data: \n\t Start Date {}, Lag Zero {} \n\t {} timesteps, deltat = {}'.format(str(self.start_date),str(self.lag0),str(self.time_data['nits']),str(self.deltat))
        for td in ['its','dates','lag_days','lag_years']:
            out_str = out_str+'\n\t {} from {} to {}'.format(td,str(self.time_data[td][0]),str(self.time_data[td][-1]))
        out_str = out_str+'\n Adjoint Variables: \n\t ADJ type {} \n\t adxx type {}'.format(str(self.ADJ_vars),str(self.adxx_vars))
        if 'data' in vars(self):
            out_str = out_str+'\n Data loaded: '+str(self.data)
        else: 
            out_str = out_str+'\n No data loaded. Use [].load_vars() to load variables'
        return out_str
    
    def _find_results(self):
        '''
        Finds and prints all ADJ and adxx in experiment,
        adds them to lists self.ADJ_vars and self.adxx_vars.
    
        '''
        # find all ADJ meta files at first it
        self.ADJ_vars=[]
        allADJ = [os.path.basename(x) for x in glob.glob(self.exp_dir+'ADJ*'+'{:010.0f}'.format(self.time_data['its'][0])+'.meta')]
        for item in allADJ:
            #i1 = item.find('ADJ')
            i2 = item.find('.')
            self.ADJ_vars.append(item[:i2])
        del allADJ
        print('Found {:d} ADJ variables'.format(len(self.ADJ_vars)))
        
        # find all adxx meta files
        all_vars=[]
        alladxx = [os.path.basename(x) for x in glob.glob(self.exp_dir+'adxx_*'+'{:010.0f}'.format(adxx_it)+'.meta')]
        for item in alladxx:
            #i1 = item.find('adxx')
            i2 = item.find('.')
            all_vars.append(item[:i2])
        varset = set(all_vars)
        self.adxx_vars=list(varset)
        del alladxx,varset
        print('Found {:d} adxx variables'.format(len(self.adxx_vars)))
        
    # Load adjoint files (assumes nz=1 for adxx vars)    
    def load_vars(self,var_list='ALL'):
        '''
        Load user specified list of variables into xarray DataSet placed in self.data.
        Will overwrite any previously loaded variables with the same name.

        Parameters
        ----------
        var_list : list of strings. Default 'ALL'
            Names of data variables to be loaded. If 'ALL', all will be loaded            

        '''
        if var_list=='ALL':
            var_list=[*self.ADJ_vars,*self.adxx_vars]       

        # Loop through and read variables
        datasets = []
        for var in var_list:
            print('Reading in '+var)
            if var not in adj_dict.keys():
                raise ValueError('{} not found in adj_dict. Please add details of variable to inputs.py'.format(var))
            
            if 'vartype' in adj_dict[var].keys():
                dims=_parse_vartype(adj_dict[var]['vartype'],adj_dict[var]['ndims'])  
            else:
                dims=None
            if 'attrs' in adj_dict[var].keys():
                attrs=adj_dict[var]['attrs']
            else:
                attrs={}
            
            if adj_dict[var]['adjtype'] == 'ADJ':                    
                if dims is None:
                    var_ds= xmitgcm.open_mdsdataset(data_dir=self.exp_dir,grid_dir=self.grid_dir,
                                                prefix=[var,],geometry='llc',delta_t=self.deltat,ref_date=self.start_date,
                                                read_grid=False)   
                    var_ds=var_ds.rename({'face':'tile'})
                else:                             
                    extra_variable={var:dict(dims=dims,attrs=attrs)}
                    var_ds= xmitgcm.open_mdsdataset(data_dir=self.exp_dir,grid_dir=self.grid_dir,
                                                prefix=[var,],geometry='llc',delta_t=self.deltat,ref_date=self.start_date,
                                                extra_variables=extra_variable,read_grid=False)
                    var_ds=var_ds.rename({'face':'tile'})

            elif var in self.adxx_vars:
                if adj_dict[var]['ndims']==3:
                    var_data= xmitgcm.utils.read_3d_llc_data(fname=self.exp_dir+'/'+var+'.'+'{:010.0f}'.format(adxx_it)+'.data',nz=50,nx=90,nrecs=self.time_data['nits'],dtype='>f4') 
                elif adj_dict[var]['ndims']==2:
                    var_data= xmitgcm.utils.read_3d_llc_data(fname=self.exp_dir+'/'+var+'.'+'{:010.0f}'.format(adxx_it)+'.data',nz=1,nx=90,nrecs=self.time_data['nits'],dtype='>f4') 
                else:
                    raise ValueError('Ndims of variables must be 2 or 3')
                if isinstance(adj_dict[var]['vartype'],str):
                    if adj_dict[var]['ndims']==3:
                        var_ds=ecco.llc_tiles_to_xda(var_data, var_type=adj_dict[var]['vartype'],dim4='depth', dim5='time')
                    elif adj_dict[var]['ndims']==2:
                        var_ds=ecco.llc_tiles_to_xda(var_data, var_type=adj_dict[var]['vartype'],dim4='time')
                    else:
                        raise ValueError('Ndims of variables must be 2 or 3')
                    var_ds=xr.Dataset(data_vars={var:var_ds},coords=var_ds.coords)
                elif dims is None:
                    raise ValueError('Vartype must be defined for adxx fields')
                else:
                    grid_ds = xmitgcm.open_mdsdataset(iters=None,read_grid=True,geometry='llc',prefix=var,data_dir=self.exp_dir,grid_dir=self.grid_dir)
                    dims=['face',]+dims
                    newcoords = {k: grid_ds[k] for k in dims}
                    dims=['time',]+dims 
                    newcoords['time']=self.time_data['dates']
                    var_ds=xr.Dataset(data_vars={var:(dims,var_data)},coords=newcoords)
                    var_ds=var_ds.rename({'face':'tile'})
                    del newcoords,grid_ds
                var_ds[var].attrs=attrs
                var_ds=_add_time_coords(var_ds,self.time_data)
                        
                #var_ds = xr.Dataset(data_vars={var:(vardims,var_data)},coords=newcords)
                #var_ds = xr.combine_by_coords([grid_ds,var_ds])
            else:
                print('variable '+var+' not found in '+self.exp_dir)

            datasets.append(var_ds)                
            del var_ds       
        # At to existing data or create new attr
        if hasattr(self,'data'):
            self.data = xr.combine_by_coords([self.data,]+datasets)
        else:
            self.data = xr.combine_by_coords(datasets)
        del datasets
 
    def to_nctiles(self,var_list=None,out_dir=None):
        '''
        Writes data to nctiles format netcdf files, one per timestep matching ECCOv4r4 format
        Loads any variables not already read in.
        Output netcdfs readable in python using xarray.open_dataset,
        or in matlab with gcmfaces toolbox fn read_nctiles.

        Parameters
        ----------
        var_list : list, optional
            List of variables to be written to file. The default is to write the variables
            found in <experiment>.data. 
            Can also be 'ALL' which writes all variables found in experiment.
        out_dir : str, optional
            Where files are to be written. The default is the experiment directory.

        Returns
        -------
        None.

        '''
        if out_dir is None:
            out_dir=self.exp_dir
            
        print('Preparing to write netcdf to '+out_dir)
        
        if 'data' not in vars(self):
            print('No data found, reading in data first')
            self.load_vars(var_list)
        else:
            if var_list is None:
                var_list=[var for var in self.data]
            else:
                if var_list == 'ALL':
                    var_list=[*self.ADJ_vars,*self.adxx_vars]                  
                load_list=[var for var in var_list if var not in self.data]
                if not load_list ==[]:
                    print('Reading in '+str(load_list))
                    self.load_vars(var_list)                

        print('All variables loaded, starting write')      
        for var in var_list:
            print('Writing '+var)
            for it in range(0,self.nits):
                file_name='{}.{:010.0f}.nc'.format(var,myexp.time_data['its'][0])
                self.data[var].isel(time=it).to_netcdf(path=out_dir+file_name)
        
        print('All files written to '+out_dir)        
    
    # Calculate stats with optional sigma multiplier    
#    def calc_stats(self,sigma=None,sigma_type=None): 
        # sigma should be dictionary with keys equal to variable names
        # sigma_type should be 1D or 3D, if sigma provided
        
        # # Check if data exists
        # if not hasattr(self,'data'):
        #     raise AttributeError('No data found in this experiment - run .load_vars first')
        # # Check what sigma we have
        # if sigma is None:
        #     print('No sigma provided - raw sensitivity stats only')
        # elif type(sigma) is dict:
        #     print('Using provided sigma dictionary for multiplier')
        #     if sigma_type == '1D':
        #         if type(list(sigma.values())[0]) is float: 
        #             print('Found 1D sigmas')
        #         else:
        #             raise TypeError('sigma dict should contain floats')
        #     elif sigma_type == '3D':
        #         if type(list(sigma.values())[0]) is str:
        #             print('Found 3D sigmas')
                    
        #             #TO DO: ADD STATS
                    
        #              #exfqnet = xr.open_mfdataset(glob.glob('EXFqnet*nc'),concat_dim='tile')  
                    
        #         else:
        #             raise TypeError('sigma dict should contain filenames')
        #     else:
        #         raise ValueError('sigma_type should be 1D or 3D')
        # else:
        #     raise TypeError('sigma should be a dictionary')
        

        
def _get_time_data(exp_dir,start_date,lag0,deltat) :   
    
    tdata={}
    with open(exp_dir+'its_ad.txt') as f:
        itin = f.readlines()
    nits=len(itin)
    tdata['nits'] = nits   
    tdata['its'] = np.zeros(nits,dtype='int')
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

def _parse_vartype(vartype,ndims):
    if isinstance(vartype, str):
        if vartype=='c':
            dims=['k','j','i']
        elif vartype=='w':
            dims=['k','j','i_g']
        elif vartype=='s':
            dims=['k','j_g','i'] 
        elif vartype=='z':
            dims=['k','j_g','i_g']
        else:
            raise ValueError('Unrecognized vartype, expecting c,w,s or z.')
        dims=dims[-ndims:]
    elif isinstance(vartype,list):
        dims=vartype
    else:
        print('Expecting a string or list for vartype')
    return dims

def _add_time_coords(var_ds,time_data):
    var_ds['time']=time_data['dates']
    var_ds=var_ds.assign_coords(lag_days=("time",time_data['lag_days']))
    var_ds=var_ds.assign_coords(lag_years=("time",time_data['lag_years']))
    return var_ds
        