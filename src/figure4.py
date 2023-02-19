#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 11:12:46 2023

@author: Mo Cohen
"""

import iris
import warnings
import matplotlib.pyplot as plt
import iris.coords
import numpy as np
from numpy import unravel_index
import windspharm

warnings.filterwarnings('ignore')


def rwave_velocity(datalist, start=500, end=600, nlat=90, nlon=144, level=8,
                   omega=1.19e-05, g=9.12, radius=5797818, lat=80, meaning=5,
                   savedir='/exports/csce/datastore/geos/users/s1144983/papers/cloudproject/epsfigs/',
                   save=False):
    """ This function calculates the Rossby wave phase speed (including 
    zonal wind) over time. At certain latitudes, the phase velocity alternates 
    between positive and negative. This is the region where the cyclonic 
    structure appears to travel back and forth periodically. 
    
    Inputs:
        datalist - List of Iris CubeLists. Should include Control and Dry
        TRAP-1e simulations, in that order.
        start - Start of period to be plotted (int)
        end - End of period to be plotted (int)
        nlat - Number of latitudes (int)
        nlon - Number of longitudes (int)
        level - Level being plotted (int)
        omega - Rotation rate in rads/sec. Default TRAPPIST-1e (float)
        g - Gravitational constant. Default TRAPPIST-1e (float)
        lat - Latitude where gyre is being tracked. Default 71N (row 80) (int)
        meaning - Meaning period for rolling mean. Default 5 days. (int)
        savedir - Directory to save output plot (str)
        save - Save plot or no (Boolean)
        
    Outputs:
        Plot of Rossby wave phase velocities at 71N on Control and Dry 
        TRAPPIST-1e showing oscillation across 0 for the former only.
        
        Plot of northeast gyre location (in longitude east) and Rossby wave
        phase speed for Control TRAP-1e showing matching periods.
        
        """

    cphase_list = []
    cphase_hlist = []
    gyrelon_list = []
    names = ['Control TRAP-1e', 'Dry TRAP-1e']
    # Names of the two simulations we are comparing

    for cubes in datalist:
        # Take CubeList for each simulation separately
        for cube in cubes:
            if cube.standard_name == 'x_wind':
                x_wind = cube[start:end, :, :, :].copy()
                longterm_x_wind = cube.copy()
            if cube.standard_name == 'y_wind':
                y_wind = cube[start:end, :, :, :].copy()
            if cube.standard_name == 'air_potential_temperature':
                theta = cube[start:end, :, :, :].copy()
            if cube.standard_name == 'air_pressure':
                pressure = cube[start:end, :, :, :].copy()
        # Extract datacubes we need for the calculation
        y_wind = y_wind.regrid(x_wind, iris.analysis.Linear())
        # Regrid y-wind onto coordinates of x-wind cube
        km_heights = np.round(x_wind.coord('level_height').points*1e-03, 2)
        # Extract heights in km for labelling plots
        latitudes = x_wind.coord('latitude').points
        longitudes = x_wind.coord('longitude').points
        # Extract latitudes and longitudes

        winds = windspharm.iris.VectorWind(
            x_wind[:, level, :, :], y_wind[:, level, :, :])
        # Create windspharm VectorWind data object containing u and v
        rel_vort = winds.vorticity()
        # Calculate relative vorticity
        rel_vort = np.flip(rel_vort.data, axis=1)
        # windspharm reverses latitudes, so we flip the data back to stay
        # south to north

        lat_deg = int(latitudes[lat])
        # Convert input row number to latitude in degrees north
        print(lat_deg)
        zmzw = x_wind[:, level, lat, :].collapsed(
            'longitude', iris.analysis.MEAN)
        zmzw = zmzw.data
        longterm_zmzw = longterm_x_wind[:, level, lat, :].collapsed(
            ['longitude', 'time'], iris.analysis.MEAN)
        longterm_zmzw = longterm_zmzw.data
        # Calculate zonal mean zonal wind

        lat_rad = lat_deg*(np.pi/180)
        # Convert input latitude to radians
        beta = 2*omega*np.cos(lat_rad)/radius
        # Beta factor
        circum = 2*np.pi*radius*np.cos(lat_rad)
        # Circumference in meters at input latitude
        x_num = 2*np.pi/circum
        # Zonal wavenumber in units of m^-1 at input latitde

        d_theta = iris.analysis.calculus.differentiate(theta, 'level_height')
        # Change in potential temperature with height
        bv_freq = np.mean(
            np.sqrt(np.abs((g/theta[:, :-1, :, :].data)*d_theta.data)), axis=-1)
        # Zonal mean Brunt-Vaisala frequency
        Ld = (bv_freq[:, level, lat]*6800)/(2*omega*np.sin(lat_rad))
        # Calculate Rossby radius of deformation using the BV frequency
        # for the height and latitude we are plotting
        # Scale height is fixed at 6800 for all simulations

        c_phase = (zmzw - longterm_zmzw) - ((beta +
                                             ((zmzw - longterm_zmzw)/(Ld**2)))/(x_num**2 + (1/Ld)**2))
        # Phase velocity as per Vallis 6.65
        cphase_list.append(c_phase)
        # Append to list
        c_phase_h = (zmzw - (beta/(x_num**2)))
        # Phase velocity not accounting for vertical effects
        cphase_hlist.append(c_phase_h)
        # Append to its own list

        core_lons = []
        for time in range(0, rel_vort.shape[0]):
            rossby_core = unravel_index(np.argmax(
                rel_vort[time, 70:, 0:72], axis=None), rel_vort[time, 70:, 0:72].shape)
            # Extract the indices of the maximum relative vorticity in each
            # time coordinate. To avoid catching the substellar region, we
            # limit the search region to the northeast quarter of the globe.
            core_lons.append(rossby_core[1])
            # Make list of the longitudes (E) of the northeast Rossby gyre core

        lon_deg = [int(longitudes[item]) for item in core_lons]
        # Get actual longitudes for our gyre indices
        lon_deg_meaned = np.convolve(np.array(lon_deg).flatten(),
                                     np.ones(meaning), 'valid')/meaning
        # Take rolling mean over the input meaning period (default 5 days)
        # This smooths out noise due to random areas of high vorticity in
        # daily snapshots
        gyrelon_list.append(lon_deg_meaned)

    # Figure 4a):

    plt.plot(cphase_list[0], color='r', label=names[0] + ', with kd')
    # Control TRAP-1e
    plt.plot(cphase_list[1], color='b', label=names[1] + ', with kd')
    # Dry TRAP-1e
    plt.plot(cphase_hlist[0], color='r',
             linestyle='dashed', label=names[0] + ', no kd')
    # Control TRAP-1e
    plt.plot(cphase_hlist[1], color='b',
             linestyle='dashed', label=names[1] + ', no kd')
    # Dry TRAP-1e

    plt.title('Rossby wave phase velocity at %s N' % (lat_deg))
    plt.xlabel('Time [days]')
    plt.ylabel('Velocity [m/s]')
    plt.legend(fontsize='small')

    if save == True:
        plt.savefig(savedir + 'cphase_%s_to_%s.eps' %
                    (start, end), format='eps', bbox_inches='tight')
    else:
        pass
    plt.show()

    # Figure 4b):
    time_axis = np.arange(start, end-meaning+1)
    # Create numbered time axis
    time_length = np.arange(0, time_axis.shape[0])
    # Just a list from 0 to the time axis length (used in loop below)
    c_phase_meaned = np.convolve(np.array(cphase_list[0]),
                                 np.ones(meaning), 'valid')/meaning
    # Take rolling mean of phase velocity to match gyre longitude array length
    # We are doing Control TRAP-1e only, hence we take c_phase[0]
    # phase_shift = np.mean(gyrelon_list[0])
    # print(phase_shift)
    # What's the mean longitude? This is the phase shift of the wave response
    # east of the substellar point.
    zero_ind = np.where(np.diff(np.sign(c_phase_meaned)))[0]
    # Find indices where the sign of the phase velocity changes

    zeroes = []
    for zc in zero_ind:
        t1 = time_length[zc]
        t2 = time_length[zc+1]
        p1 = c_phase_meaned[zc]
        p2 = c_phase_meaned[zc+1]
        interpolated_zero = t1 + (0-p1) * ((t2-t1)/(p2-p1))
        zeroes.append(interpolated_zero)
        # This code block interpolates points where the
        # phase velocity is 0

    fig, ax1 = plt.subplots()
    # Plot with gyre longitude location and phase velocity on the two y-axes
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Velocity [m/s]')
    ax1.plot(time_axis, c_phase_meaned, color='r', label='Phase vel')
    ax1.plot([item+start for item in zeroes],
             np.zeros_like(zeroes), 'o', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Longitude [deg E]')
    ax2.plot(time_axis, gyrelon_list[0], color='b', label='Longitude')
    ax2.plot(time_axis, np.ones_like(gyrelon_list[0])
             * 83, color='b', linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor='b')

    plt.title('Rossby wave phase velocity at %sN and gyre longitude' % lat_deg)
    fig.tight_layout()

    if save == True:
        plt.savefig(savedir + 'gyre_lon_cphase_%s_to_%s.eps' % (start, end),
                    format='eps', bbox_inches='tight')
    else:
        pass
    plt.show()
    
    plt.plot(time_axis, gyrelon_list[0])
    plt.title('Path travelled by northeast gyre')
    plt.xlabel('Time [days]')
    plt.ylabel('Longitude [deg E]')
    plt.show()
