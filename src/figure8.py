#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 13:59:44 2023

@author: Mo Cohen
"""

import iris
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def cloud_tseries(cubes, start=0, end=600, long1=36, long2=108, filtering=True,
               ice_threshold=0.1, liq_threshold=0.1, sim='trap',
               savedir='/exports/csce/datastore/geos/users/s1144983/papers/cloudproject/epsfigs/',
               save=False):
    
    """ Plot time series of ice and liquid cloud at the terminators for the
    input simulation.
    
    Inputs:
        cubes - Iris CubeList
        start - Start of period to be plotted (int)
        end - End of period to be plotted (int)
        long1 - Longitude/row of eastern terminator (int)
        long2 - Longitude/row of western terminator (int)
        filtering - Whether to filter out noise. Default True (Boolean)
        ice_threshold - Noise threshold for ice condensate. Default is 0.1, 
        which filters out cycles shorter than 10 days (float)
        liq_thtreshold - Noise threshold for liquid condensate. Default 0.1
        sim - Which simulation you are plotting, used to label saved plot
        (str)
        savedir - Directory to save output plot (str)
        save - Save plot or no (Boolean)
        
    Outputs:
        Time series of liquid and ice condensate on same axis
        Time series of liquid and ice condensate on same plot but different
        axes                                                        """

    for cube in cubes:
        if cube.standard_name == 'mass_fraction_of_cloud_ice_in_air':
            ice_condensate_raw = cube[start:end, :, :, :].copy()
        if cube.standard_name == 'mass_fraction_of_cloud_liquid_water_in_air':
            liquid_condensate_raw = cube[start:end, :, :, :].copy()
        # Extract datacubes to plot

    if long1 == 36 and long2 == 108:
        titleloc = 'Terminators'
    elif long1 == 143 and long2 == 0:
        titleloc = 'Substellar Point'
    else:
        titleloc = 'No automatic title generation'
    # Create string for plot title depending on input longitudes

    ice_condensate = ice_condensate_raw.data
    liquid_condensate = liquid_condensate_raw.data
    # Extract numpy array from cubes

    time_axis = np.arange(start, end+1)
    # Create x-axis of time coordinates
    heights = np.array(ice_condensate_raw.coord('level_height').points)
    # Get heights (used for weighting in mean calculation)
    # Levels are not evenly spaced, so we need weighting in our calc

    ice_east = np.mean(ice_condensate[:, :, :, long1], axis=2)
    ice_east = np.average(ice_east, axis=1, weights=heights)
    ice_west = np.mean(ice_condensate[:, :, :, long2], axis=2)
    ice_west = np.average(ice_west, axis=1, weights=heights)
    # For ice condensate, extract terminators separately and get
    # height-weighted mean

    liq_east = np.mean(liquid_condensate[:, :, :, long1], axis=2)
    liq_east = np.average(liq_east, axis=1, weights=heights)
    liq_west = np.mean(liquid_condensate[:, :, :, long2], axis=2)
    liq_west = np.average(liq_west, axis=1, weights=heights)
    # Repeat for liquid condensate

    total_ice = (ice_east + ice_west)/2
    total_liq = (liq_east + liq_west)/2
    # Take average to find overall mean around the limb for each
    # type of condensate

    if filtering == True:
        fft_ice = sp.fftpack.fft(total_ice)
        # Take Fourier transform of ice time series
        psd_ice = np.abs(fft_ice)**2
        # Calculate power spectral density
        freqs_ice = sp.fftpack.fftfreq(len(psd_ice), 1./1)
        # Get frequencies in the ice time series
        lowpass_ice = fft_ice.copy()
        lowpass_ice[np.abs(freqs_ice) > ice_threshold] = 0
        # Apply a low-pass filter, removing frequencies above the
        # threshold
        total_ice = np.real(sp.fftpack.ifft(lowpass_ice))
        # Transform back out of spectral space to get filtered time series

        fft_liq = sp.fftpack.fft(total_liq)
        psd_liq = np.abs(fft_liq)**2
        freqs_liq = sp.fftpack.fftfreq(len(psd_liq), 1./1)
        lowpass_liq = fft_liq.copy()
        lowpass_liq[np.abs(freqs_liq) > liq_threshold] = 0
        total_liq = np.real(sp.fftpack.ifft(lowpass_liq))
        # Repeat all steps for liquid condensate

    fig, ax1 = plt.subplots()
    # Create figure which will have two y-axes
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Ice condensate [kg/kg]')
    ax1.plot(time_axis, total_ice, color='b', label='Ice')
    ax1.tick_params(axis='y', labelcolor='b')
    # Plot ice condensate on one axis

    ax2 = ax1.twinx()
    # Duplicate the first axis
    ax2.set_ylabel('Liquid condensate [kg/kg]')
    ax2.plot(time_axis, total_liq, color='r', label='Liquid')
    ax2.tick_params(axis='y', labelcolor='r')
    # Plot liquid condensate on second axis

    plt.title('Mean Ice and Liquid Condensate at %s' % titleloc)
    fig.tight_layout()
    plt.show()
    # Show figure

    plt.plot(time_axis, total_ice, color='b', label='Ice')
    plt.plot(time_axis, total_liq, color='r', label='Liquid')
    plt.title('Mean Cloud Condensate at %s' % titleloc)
    plt.xlabel('Time [days]')
    plt.ylabel('Cloud condensate [kg/kg]')
    plt.legend()
    # This time, plot the two time series on the same axis

    if save == True:
        # Only the second figure is saved by default
        plt.savefig(savedir + 'cloudseries_%s_%s_%s.eps' %
                    (start, end, sim), format='eps', bbox_inches='tight')
    else:
        pass
    plt.show()
