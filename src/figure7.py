#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 12:54:45 2023

@author: Mo Cohen
"""

import iris
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import iris.coords
import numpy as np
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings('ignore')


def global_cphase(datalist, ndata=5, start=500, end=800, nlat=90, nlon=144,
                  level=8,
                  savedir='/exports/csce/datastore/geos/users/s1144983/papers/cloudproject/epsfigs_v2/',
                  save=False):
    """ This function calculates the Rossby wave phase speed (including 
    zonal wind) over time. At certain latitudes, the phase velocity alternates 
    between positive and negative. This is the region where the cyclonic 
    structure appears to travel back and forth periodically. The function
    outputs a time-latitude contourfill plot. 
    
    Inputs:
        datalist - Master ist of Iris CubeLists
        ndata - Number of CubeLists in the master list
        start - Start of period to be plotted (int)
        end - End of period to be plotted (int)
        nlat - Number of latitudes (int)
        nlon - Number of longitudes (int)
        level - Level being plotted (int)
        savedir - Directory to save output plot (str)
        save - Save plot or no (Boolean)
        
    Outputs:
        Time-latitude contourfill plots of Rossby wave phase velocity
        for all five input simulations.    """

    names = ['Control ProxB',
             'Warm ProxB', 'Control TRAP1-e', 'Warm TRAP1-e', 'Dry TRAP1-e']

    omegas = [6.46e-06, 7.93e-06, 1.19e-05, 1.75e-05, 1.19e-05]
    radii = [7160000, 7160000, 5797818, 5797818, 5797818]
    gravities = [10.9, 10.9, 9.12, 9.12, 9.12]

    redblu = mpl_cm.get_cmap('coolwarm')

    for i in range(ndata):

        data = datalist[i]
        name = names[i]
        omega = omegas[i]
        radius = radii[i]
        g = gravities[i]

        for cube in data:
            if cube.standard_name == 'x_wind':
                x_wind = cube[start:end, :, :, :].copy()
                longterm_x_wind = cube.copy()
            if cube.standard_name == 'air_potential_temperature':
                theta = cube[start:end, :, :, :].copy()
            if cube.standard_name == 'air_pressure':
                pressure = cube[start:end, :, :, :].copy()

        heights = np.round(x_wind.coord('level_height').points*1e-03, 2)
        # Extract height for labelling plots
        lat_deg = x_wind.coord('latitude').points
        # Extract latitudes

        zmzw = x_wind[:, level, :, :].collapsed(
            'longitude', iris.analysis.MEAN)
        zmzw = zmzw.data
        longterm_zmzw = longterm_x_wind[:, level, :, :].collapsed(
            ['longitude', 'time'], iris.analysis.MEAN)
        longterm_zmzw = longterm_zmzw.data
        # Calculate zonal mean zonal wind

        lat_rad = lat_deg*(np.pi/180)
        # Convert latitudes to radians
        beta = 2*omega*np.cos(lat_rad)/radius
        # Beta factor for all latitudes
        circum = 2*np.pi*radius*np.cos(lat_rad)
        # Circumference in meters for all latitudes
        x_num = 2*np.pi/circum
        # Zonal wavenumber in units of m^-1 for all latitudes

        d_theta = iris.analysis.calculus.differentiate(theta, 'level_height')
        # Change in potential temperature with height
        bv_freq = np.mean(
            np.sqrt(np.abs((g/theta[:, :-1, :, :].data)*d_theta.data)), axis=-1)
        # Zonal mean Brunt-Vaisala frequency
        Ld = (bv_freq[:, level, :]*6800)/(2*omega*np.sin(lat_rad))
        # Calculate Rossby radius of deformation using the BV frequency
        # for the height and latitude we are plotting
        # Scale height is fixed at 6800 for all simulations

        c_phase = (zmzw - longterm_zmzw) - ((beta +
                                             ((zmzw - longterm_zmzw)/(Ld**2)))/(x_num**2 + (1/Ld)**2))
        # Calculate phase velocity

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.contourf(c_phase.T, levels=np.arange(-220, 41, 10),
                     cmap=redblu, norm=TwoSlopeNorm(0))
        plt.title('Rossby wave phase velocity, h=%s km' % heights[level], fontsize=14)
        plt.xlabel('Time [days]', fontsize=14)
        plt.ylabel('Latitude [degrees]', fontsize=14)
        plt.yticks((0, 15, 30, 45, 60, 75, 90),
                   ('90S', '60S', '30S', '0', '30N', '60N', '90N'))
        mbar = plt.colorbar(pad=0.1)
        mbar.set_ticks(np.arange(-220, 41, 20))
        mbar.ax.set_title('m/s')

        if save == True:
            plt.savefig(savedir + 'cphase_hov_%s.eps' % name, format='eps',
                        bbox_inches='tight')
        else:
            pass
        plt.show()
