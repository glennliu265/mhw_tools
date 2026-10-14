#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Marine Heat Wave Utitilies

Utilities/Functions for analysis of Marine Heatwaves in the MESACLIP simulations


List of Functions
-----------------

    MHW Identification
    ------------------------
    arr2da                     : Transform output of [id_extremes_arr] from array into DataArray
    combine_consecutive_events : Merge events based on duration and intensity criteria
    cull_events_byduration     : Remove events shorter than the minimum duration
    id_extremes_arr            : Identify Events from exceedences of percentile, then merge and get key metrics
    retrieve_event_metrics_arr : Get event metrics given output of [combine_consecutive_events] within [id_extremes_arr]
    
    General Helper Functions
    ------------------------
    get_paths                  : Get path to lores/hires CESM simulations on glade, account for ensemble names
    get_nclist_ens             : Get list of NetCDFs for a given simulation (all ensemble members), wrapper for [get_paths]
    pad_nan                    : Add NaN to pad output to the same length, for DataArray creation



[2026.10.13] --> Copied many functions from `GOA_Event_Combine_Testing.ipynb

Created on Wed Aug  5 14:20:35 2026

@author: gliu

"""

import numpy as np
import glob
import xarray as xr



#%% MHW Indentification



def arr2da(metricsout):
    # Convert output of id_extremes_arr into DataArray
    # [2026.09.23] Updated to work with id_first and id_last
    outnames = ['id_max','event_max','duration','event_mean','cumulative_intensity','nevents','id_first','id_last']
    id_nevents = outnames.index('nevents')
    nevents  = metricsout[id_nevents]
    eventid  = np.arange(nevents)+1
    coords   = dict(eventid=eventid)
    
    nvars = len(outnames)
    dsout = []
    for ii in range(nvars):
        if outnames[ii] ==  "nevents":
            continue
        outvar = metricsout[ii][:nevents]
        if ii == 0:
            outvar = outvar.astype(int)
        da     = xr.DataArray(outvar,coords=coords,dims=coords,name=outnames[ii])
        dsout.append(da)
    dsout = xr.merge(dsout)
    dsout['nevents'] = nevents
    return dsout



def combine_consecutive_events(timeseries,event_indices,tol=1,verbose=True,efolding_cutoff_pct=0.40,percentile=None):
    """
    Given a timeseries of anomalies and indices of exceendences, merge each into coherent events using a duration tolerance [tol]
    and an optional integrated intensity if the [percentile] is provided.

    Parameters
    ----------
    timeseries : Array/Numeric [time]
        1-D Timeseries of an anomaly.
    event_indices : Array of int [time]
        Indices of each exceendence.
    tol : Int or Array, optional
        One of several options to merge events if [time to next exceedance] < tol
            - Int              --> Max allowable # of days/months to separate events.
            - Array of Numeric --> Tolerance thresholds for either monthly data (12 values) or daily data (365 values)
        The default is 1 (Int threshold).
    verbose : BOOL, optional 
        True to output verbose comments. The default is True.
    efolding_cutoff_pct : NUMERIC, optional
        Percent adjustment applied to e-folding timescale (currently not implemented, commented out). The default is 0.40.
    percentile : Array/Numeric [time], optional
        Percentile values used to identify the exceedances. If supplied, implements integrated intensity criteria. 
        Note that this has only been implemented for positive events. Need to adjust sign for negative check...
        The default is None.

    Returns
    -------
    event_combine : Nested list containing indices for each event, 
        List of lists, where each inner list contains the exceedances grouped together for 1 event.
    
    """
    # Make some adjustments based on the tolerance
    # Note: Asssumes timeseries has an even number of years or is evenly divisible...
    # Copied from [GOA_Event_Combine_Testing.ipynb] on 2026.10.13
    if type(tol) != int: # Tolerance is NOT just 1 value
        use_efolding = True
        if len(tol) == 12: # Monthly E-Folding Tolerance
            # Tolerance Applied by Month
            nyrs     = int(len(timeseries)/12)
            tolcheck = np.tile(np.arange(1,13,1),nyrs)
            if len(timeseries)%12 != 0:
                print("Warning, Timeseries is not evenly divisible by 12, indexing errors may result.")
        elif len(tol) == 365: # Daily E-folding Tolerance
            # Tolerance Applied by Day of Year
            nyrs     = int(len(timeseries)/365)
            tolcheck = np.tile(np.arange(1,366,1),nyrs)
            if len(timeseries)%365 != 0:
                print("Warning, Timeseries is not evenly divisible by 365, indexing errors may result.")
        else:
            print("Invalid size for Tol... must be either a constant or array of 12 (monthly) or 365 (doy)")
    else:
        use_efolding = False
        tolcheck = None
    
    if percentile is not None: # Get Difference from Percentile
        percentile_anom = timeseries - percentile
    
    # Given a tolerance level, merge consecutive events
    # Separate into discrete events
    # Copied from combine_events from enso event id
    nevents        = len(event_indices)
    if verbose:
        print("Original Starting Count: %i Events" % nevents)
        if not use_efolding:
            print("\tCombining Events <= [%i] timesteps apart" % tol)
    
    # Looping through events
    event_combine = []
    for ii in range(nevents+1): 
        
        if ii == (nevents): # Don't Perform Check for the last event
            # if verbose:
            #     print("Merging last event: %s" % event_merge)
            event_combine.append(event_merge) # This is defined later
            continue
        
        # Get the Index of the event
        ievent = event_indices[ii].item()
        
        if ii == 0: # Start Array with event to calculate Distances
            prev_id     = ievent
            event_merge = [ievent,]
            continue

        if tolcheck is None: # Tolerance is just a single value
            tol_in = tol
        else: # Tolerance is an Array (day of year, or month)
            
            event_category = tolcheck[ievent] # Check day or month of event
            tol_in         = tol[event_category-1]    # -1 for Python Indexing

        
        dt_since_last_exceedence = ievent - prev_id # Change in time between current index and last index
        
        if (dt_since_last_exceedence <= tol_in) and (len(event_combine) != 0): # Consecutive Event
        
            # # Version One ======================== Compare Event Duration to % of E-folding =========================================
            # # Note --> this doesnt work well because of small initial event durations, Maybe should wait until e-fold skips event duration
            # # Stop counting events if their total duration is shorter than a given
            # # percent of the efolding cutoff. Useful for loations with very long e-folding
            # # artificially merging events short events...
            # if (use_efolding) and (efolding_cutoff_pct is not None): # Apply Additional Criteria
            #     if len(event_merge) == 1:
            #         current_event_duration = 1
            #     else:
            #         current_event_duration = event_merge[-1] - event_merge[0] + 1 # Elapsed time since last event
            #     duration_cutoff        = efolding_cutoff_pct * tol_in
                
            #     #print("Time Since Last Exceedence: %2i vs. Current Event Duration: %2i" % (dt_since_last_exceedence,current_event_duration))
            #     if (current_event_duration < duration_cutoff) and (dt_since_last_exceedence > current_event_duration):
            #         if verbose:
            #             print("ievent = %i, event start = %i" % (ievent,event_merge[0]))
            #             #print("\tEvent at (i=%i) was ended because the current duration (%i) exceeds %.2f (%.2f of the E-Folding Time %.2f)" % (ievent,current_event_duration,duration_cutoff,efolding_cutoff_pct,tol_in))
                        
            #             print("\tEvent at (i=%i) was ended because of 2 reasons:" % ievent)
            #             print("\t\t1. Current duration (%i) < %.2f (%.2f of the E-Folding Time %.2f)" % (current_event_duration,duration_cutoff,efolding_cutoff_pct,tol_in))
            #             print("\t\t2. Time Elapsed since last event (%i) > current event duration (%i)" % (dt_since_last_exceedence,current_event_duration))
                    
            #         event_combine.append(event_merge) # End the previous Event and start anew
            #         event_merge = [ievent,] # Start a new event
            #         continue
            # # END Version One ======================== Compare Event Duration to % of E-folding =========================================
            
            # # Version Two ======================== Compare Time to Next Event to % of E-Folding =========================================
            # # Also called "Route 1, confusingly in my presentation slide#
            # if (use_efolding) and (efolding_cutoff_pct is not None): # Apply Additional Criteria
            #     if len(event_merge) == 1:
            #         current_event_duration = 1
            #     else:
            #         current_event_duration = event_merge[-1] - event_merge[0] + 1 # Elapsed time since last event
            #     duration_cutoff        = efolding_cutoff_pct * tol_in

            #     if (dt_since_last_exceedence > duration_cutoff) and (dt_since_last_exceedence > current_event_duration):
            #         if verbose:
            #             print("ievent = %i, event start = %i" % (ievent,event_merge[0]))
            #             #print("\tEvent at (i=%i) was ended because the current duration (%i) exceeds %.2f (%.2f of the E-Folding Time %.2f)" % (ievent,current_event_duration,duration_cutoff,efolding_cutoff_pct,tol_in))
                        
            #             print("\tEvent at (i=%i) was ended because of 2 reasons:" % ievent)
            #             print("\t\t1. Time Elapsed since last event (%i) > %.2f (%.2f of the E-Folding Time %.2f)" % (dt_since_last_exceedence,duration_cutoff,efolding_cutoff_pct,tol_in))
            #             print("\t\t2. Time Elapsed since last event (%i) > current event duration (%i)" % (dt_since_last_exceedence,current_event_duration))
                    
            #         event_combine.append(event_merge) # End the previous Event and start anew
            #         event_merge = [ievent,] # Start a new event
            #         continue
            # # End Version Two ======================== Compare Time to Next Event to % of E-Folding =========================================
            
            # # Version Three ======================== Compare Time to Next Event to Event Duration =========================================

            # if (use_efolding) and (efolding_cutoff_pct is not None): # Apply Additional Criteria
            #     if len(event_merge) == 1:
            #         current_event_duration = 1
            #     else:
            #         current_event_duration = event_merge[-1] - event_merge[0] + 1 # Elapsed time since last event
            #     duration_over_dt       = current_event_duration / dt_since_last_exceedence

            #     if ( duration_over_dt < efolding_cutoff_pct ) and (dt_since_last_exceedence > current_event_duration):
            #         if verbose:
            #             print("ievent = %i, event start = %i" % (ievent,event_merge[0]))
            #             #print("\tEvent at (i=%i) was ended because the current duration (%i) exceeds %.2f (%.2f of the E-Folding Time %.2f)" % (ievent,current_event_duration,duration_cutoff,efolding_cutoff_pct,tol_in))
            #             print("\tEvent at (i=%i) was ended because of 2 reasons:" % ievent)
            #             print("\t\t1. Ratio of [Event Duration, %i] / [Time to Next Event, %i] is %.2f < %.2f" % (current_event_duration,dt_since_last_exceedence,duration_over_dt,efolding_cutoff_pct))
            #             print("\t\t2. Time Elapsed since last event (%i) > current event duration (%i)" % (dt_since_last_exceedence,current_event_duration))
                    
            #         event_combine.append(event_merge) # End the previous Event and start anew
            #         event_merge = [ievent,] # Start a new event
            #         continue
            # # End Version Three ======================== Compare Time to Next Event to Event Duration =========================================


            # # Version Four ========================================= Consider the integrated percentile (abs value) =========================================
            # Integrate the current event and time to next event, and use this to decide whether or not to crop
            if percentile is not None:
                current_event_integr     = np.nansum(percentile_anom[event_merge[0]:(event_merge[-1]+1)]) # Integrated Event
                next_exceedance_integr   = np.nansum(percentile_anom[event_merge[-1]:(ievent+1)]) # Integrate to next exceedence
                if verbose:
                    print("Integral of Current Event (%.2f) vs. To Next Exceedence (%.2f)" % (current_event_integr,next_exceedance_integr))
                
                # Apply Additional check for next_exceedance integr based on sign of event
                if current_event_integr >= 0: # Positive Event, make sure following drop is negative
                    next_check = next_exceedance_integr < 0
                elif current_event_integr < 0: # Negative Event, make sure following drop is positive
                    next_check = next_exceedance_integr > 0
                
                if (np.abs(next_exceedance_integr) > np.abs(current_event_integr)) and (next_check): # Also Check for Negative Values
                    print("\tAbsolute Value of Next Exceedance Integral > Current Event, splitting...")
                    event_combine.append(event_merge) # End the previous Event and start anew
                    event_merge = [ievent,] # Start a new event
                    continue
            # End Version Four ========================================= Consider the integrated percentile (abs value) =========================================
                
            
            
            # Otherwise, Merge the Event
            event_merge.append(ievent)
            # if verbose:
            #     print("%i is consecutive to previous events (%s)" % (ievent,event_merge))
        else: # Otherwise, just add event and merge
            event_combine.append(event_merge)
            event_merge = [ievent,] # Make a new one
            # if verbose:
            #     print("Making new event sequence at %i" % (ievent))
        
        
        prev_id = ievent
    nevents_combined = len(event_combine)
    if verbose:
        print("\tCulled to %i events!" % nevents_combined)
    return event_combine


def cull_events_byduration(dsin,min_duration=5):
    
    ds = dsin.copy()
    
    print("Dropping events less than %i days" % min_duration)
    id_above      = np.where(ds.duration >= min_duration)[0]

    nevents     = ds.nevents.data.item()
    nevents_new = len(id_above)
    print("%i of %i events (%.2f %%) are >= duration (%s days)" % (nevents_new,
                                                                       nevents,
                                                                     nevents_new/nevents*100,
                                                                       min_duration))
    ds = ds.isel(eventid=id_above)
    ds['nevents'] = nevents_new
    return ds


def id_extremes_arr(timeseries,thres,positive,eventid_max=None,tol=1,
                    verbose=False,tname='max',efolding_cutoff_pct=None,integrated_threshold=False):
    # # [2026.09.23] Updated output work with id_first and id_last
    # # [2026.09.24] Added event cumulative intensity as output
    # # [2026.10.12] Added option to apply integrated threshold criterion
    if eventid_max is None:
        eventid_max = int(len(timeseries)*0.25)
    # If NaN, just Continue
    if np.any(np.isnan(timeseries)):
        #print("Skipping NaN")
        dummy=np.zeros(eventid_max) * np.nan
        #output = *[dummy,]*5
        return dummy,dummy,dummy,dummy,0
    
    # Use Thresholds to find Events 
    if positive == True:
        below      = False
        event_indices = np.where(timeseries > thres)[0]
    
    else:
        below      = True
        # if verbose:
        #     print("Looking for events below threshold")
        event_indices = np.where(timeseries < thres)[0]
    
    nevents         = len(event_indices)
    # if verbose:
    #     print("Identified %i Events" % nevents)

    if integrated_threshold:
        print("Using Integrated Threshold...")
        event_combine   = combine_consecutive_events(timeseries,event_indices,tol=tol,verbose=verbose,efolding_cutoff_pct=efolding_cutoff_pct,percentile=thres)
    else:
        event_combine   = combine_consecutive_events(timeseries,event_indices,tol=tol,verbose=verbose,efolding_cutoff_pct=efolding_cutoff_pct)
    metrics_out     = retrieve_event_metrics_arr(event_combine,timeseries,tname=tname)
    # #id_out,values_out,duration,event_mean,event_std,event_cumu = metrics_out
    
    # For Output Variables, Pad with NaN #Enter the Amount
    nevents         = len(metrics_out[0])
    #npad            = eventid_max-nevents
    metrics_out     = [pad_nan(arr,eventid_max) for arr in metrics_out]
    id_out,values_out,duration,event_mean,event_std,event_cumu,id_first,id_last = metrics_out
    
    return id_out,values_out,duration,event_mean,event_cumu,nevents,id_first,id_last



def retrieve_event_metrics_arr(event_combine,timeseries,tname='max'):
    """
    Processes merged events and calculates basic statistics (max, min, mean, stdev, duration, cumulative sum)
    
    Inputs
        event_combine : list of lists, where each element is an event and each inner list contains the indices corresponding to the event
        timeseries    : xr.DataArray , timeseries containing values

    Returns
        xr.DataSet Containing Time, Indices, and Summary Stats for each event, numbered by eventid.
    
    Updates ---
    [2026.09.23] - Updated to directly output id_first and id_last
    [2026.10.07] - Change Duration to be id_last - id_first, rather than length of event loop...
    
    See `develop_MHW_id_code.ipynb` for debgging script

    """
    # Given a list of lists containing indices of combined events 
    nevents_combined = len(event_combine)
    
    # Part (1): Perform A Loop through Events =================================
    # Event Timing
    duration      = np.zeros((nevents_combined)) * np.nan
    
    # Indexing
    id_center     = duration.copy()
    id_first      = duration.copy()
    id_last       = duration.copy()
    id_max        = duration.copy()
    id_min        = duration.copy()
    
    # Event Stats
    event_mean    = duration.copy()
    event_std     = duration.copy()
    event_cumu    = duration.copy()
    for ie in range(nevents_combined):
        
        eventid_loop = event_combine[ie]
        
        # Get Variables  ------------------------------------------------
        intensities    = timeseries[eventid_loop]
        nconsecutive   = len(eventid_loop)
        
        # Record Some Metrics -------------------------------------------
        # Determine Some Indices for Metrics
        # Index within event chunk
        idmax            = np.argmax(np.abs(intensities))
        idmin            = np.argmin(np.abs(intensities))
        idfirst          = 0 #eventid_loop[0]
        idlast           = -1 #eventid_loop[-1]
        idcenter         = nconsecutive // 2 # Middle is just divided by 2
        if not nconsecutive & 0x1: # If Evenn, shift earlier
            idcenter     = idcenter - 1
        
        # Index from full timeseries
        id_center[ie]    = eventid_loop[idcenter]
        id_first[ie]     = eventid_loop[idfirst]
        id_last[ie]      = eventid_loop[idlast]
        id_max[ie]       = eventid_loop[idmax]
        id_min[ie]       = eventid_loop[idmin]
        
        # Timing (Note this saves to unreadable number...)
        duration[ie]     = eventid_loop[idlast] - eventid_loop[idfirst] + 1 # Add 1 for inclusive boundary##nconsecutive            # Duration (Note assumes regular spacing...)
        
        # Statistics
        event_mean[ie]   = np.nanstd(intensities) # Mean
        event_std[ie]    = np.nanstd(intensities) # Standard Deviation
        event_cumu[ie]   = np.nansum(intensities) # Cumulative Values

    if tname == "max":
        id_out = id_max
    elif tname == "min":
        id_out = id_min
    elif tname == "center":
        id_out = id_center
    elif tname == "start":
        id_out = id_first
    elif tname == "end":
        id_out = id_last
    values_out = [timeseries[dd.astype(int)] for dd in id_out]
    #group2           = [duration,event_mean,event_std,event_cumu]
    
    return id_out,values_out,duration,event_mean,event_std,event_cumu,id_first,id_last

#%% General Helper Functions

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

def pad_nan(indata,nmax):
    ndata = len(indata)
    if ndata > nmax:
        print("Warning: # of identified events (%i) exceeds that of the maximum pad (%i). Please set nmax manually." % (ndata,nmax))
    return np.pad(indata,(0,nmax-ndata),'constant',constant_values=np.nan)
