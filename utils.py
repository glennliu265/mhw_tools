#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Marine Heat Wave Utitilies

Utilities/Functions for analysis of Marine Heatwaves in the MESACLIP simulations


List of Functions
-----------------

    get_paths      : Get path to lores/hires CESM simulations on glade, account for ensemble names
    get_nclist_ens : Get list of NetCDFs for a given simulation (all ensemble members), wrapper for [get_paths]

Created on Wed Aug  5 14:20:35 2026

@author: gliu
"""

import numpy as np
import glob

# Copied Path functions from crop_TP_MESACLIP
def get_paths(expname,ens,freq="month_1",scenario="BHIST",regrid=True,realm='atm'):
    # Need to add scenario to the data       
    # First get the path to the data
    # Scenarios Supported" BHIST, BRCP85
    # Copied from merge_anom_detrend_ncremap on 2026.08.05
    if expname == "hires": # Hi-Res Runs in Campaign
        dpath = "/glade/campaign/collections/cmip/CMIP6/CESM-HR/RDA/%s" % scenario
        gridname="ne120_t12"
    elif expname == "lores":
        gridname="ne30_g16"
        if ens <= 10:  # First 10 Ens Lo-Res
            dpath = "/glade/campaign/collections/cmip/CMIP6/CESM-HR/RDA/lowres/%s" % scenario
        else:   # Ens 11-40 by Cuong
            dpath = "/glade/campaign/cgd/cas/scsan/CESMLR_New_Run"
    else:
        print("expname must be [hires] or [lores]")
    
    
    if regrid: # Set path to data regridded by ncremap
        dpath = "/glade/derecho/scratch/glennliu/MESACLIP/regridded"
    
    # Next, get the experiment string (based on scenario)
    if scenario == "BHIST":
        if ens == 1:
            ystr = "cesm-ihesp-sehires38-1850-2005"
        else: 
            # Hi-Res Has Specific Numbers
            if expname == 'hires':
                if (ens > 1) and (ens < 4): # 2-3
                    ystr = "cesm-ihesp-hires1.0.30-1920-2005"
                elif (ens >= 4) and (ens < 6): # 4-5
                    ystr = "cesm-ihesp-hires1.0.44-1920-2005"
                elif ens == 6: # 6
                    ystr = "cesm-ihesp-hires1.0.45-1920-2005"
                elif (ens > 6) or (ens < 11): # 6-10
                    ystr = "cesm-ihesp-hires1.0.46-1920-2005"
                else:
                    print("Warning! ens not recognized. For Hi Res. Ens is between 1 and 10")  
            elif expname == 'lores':  # Lo-Res is either 42 or 46 ====================
                if ens <= 10:
                    ystr = "cesm-ihesp-hires1.0.42-1920-2005"
                else:
                    ystr = "cesm-ihesp-hires1.0.46-1920-2005"
        
        # Now Combine
        if regrid:
            exppath = "%s/b.e13.BHISTC5.%s.%s.%03i/%s/%s/" % (dpath,gridname,ystr,ens,realm,freq)
        else:
            exppath = "%s/b.e13.BHISTC5.%s.%s.%03i/%s/proc/tseries/%s" % (dpath,gridname,ystr,ens,realm,freq)
    elif scenario == "BRCP85":
        if ens == 1:
            ystr = "cesm-ihesp-sehires38-2006-2100"
        else:
            if expname == 'hires':
                if ens == 2:   # (Ens 2)
                    ystr = "cesm-ihesp-hires1.0.30-2006-2100"
                elif ens == 3: # (Ens 3)
                    ystr = "cesm-ihesp-hires1.0.31-2006-2100"
                elif (ens >= 4) and (ens < 6): # (Ens 4-5)
                    ystr = "cesm-ihesp-hires1.0.44-2006-2100"
                else:          # (Ens 6-10)
                    ystr = "cesm-ihesp-hires1.0.46-2006-2100"
            elif expname == "lores":
                if ens <= 10: # (Ens 1-10) on gdex
                    ystr = "cesm-ihesp-hires1.0.42-2006-2100"
                else:
                    ystr = "cesm-ihesp-hires1.0.46-2006-2100"
        # Now Combine
        if regrid:
            exppath = "%s/b.e13.BRCP85C5.%s.%s.%03i/%s/%s/" % (dpath,gridname,ystr,ens,realm,freq)
        else:
            exppath = "%s/b.e13.BRCP85C5.%s.%s.%03i/%s/proc/tseries/%s" % (dpath,gridname,ystr,ens,realm,freq)
    return exppath


def get_nclist_ens(expname,vname,freq,scenario,realm='atm',regrid=False,debug=False):
    # Convenience Function to get all NetCDFs
    # Needs to be updated once permissions are changed for Lo_Res Run
    # Copied from merge_anom_detrend_ncremap on 2026.08.05
    if expname == "hires":
        ensall = np.arange(1,11,1)
    else:
        ensall = np.arange(23,41,1) # np.arange(1,41,1)#
    
    ncall = []
    print("Searching for NetCDFs for %s" % expname)
    for ens in ensall:
        datpath  = get_paths(expname,ens,freq=freq,scenario=scenario,realm=realm,regrid=regrid)
        #print(datpath)
        ncsearch = "%s/*%s*.%s.*.nc" % (datpath,scenario,vname)
        if debug:
            print(ncsearch)
        nclist   = glob.glob(ncsearch)
        nclist.sort()
        nfiles   = len(nclist)
        print("\tFound %2i files for ens %03i..." % (nfiles,ens))
        #print(nclist)
        ncall.append(nclist)
    return ncall,ensall
