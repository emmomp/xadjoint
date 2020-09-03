#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example uses of xadjoint experiment class
Created on Thu Sep  3 12:06:05 2020

@author: emmomp
"""
import experiment as ex
 
rootdir = '/data/smurphs/emmomp/orchestra/'
griddir = rootdir+'grid2/'

expdir = rootdir+'experiments/run_ad.8yr.SOpv3.00.atl/'
startdate='1993-01-01'
lag0='2000-07-01'
         
myexp = ex.Experiment(griddir,expdir,start_date=startdate,lag0=lag0)

#myexp = Exp('smurphs','run_ad.CORE2.5yr.1mosssrelax_k500_mergesss')
#myexp.find_results()
myexp.load_vars(['ADJqnet','adxx_tauu','adxx_tauv','adxx_qnet','ADJsalt'])
print(myexp)
myexp.load_vars(['adxx_tauv',])
myexp.to_nctiles(['ADJqnet',])
myexp.to_nctiles()
