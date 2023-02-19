#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:23:00 2023

@author: Mo Cohen
"""

import iris
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import iris.coords
import numpy as np


def cloud_bubble(cubes, lat=45, lon=72, start=0, end=120,
                 savedir='/exports/csce/datastore/geos/users/s1144983/papers/cloudproject/epsfigs/',
                 save='no'):
    """ Creates a contourfill plot of the sum of ice and liquid cloud
    at the equator and at longitude 0 for the input Iris CubeList.
 
    Inputs:
        cubes - Iris CubeList
        lat - Latitude to plot. Default 45 (equator) (int)
        lon - Longitude to plot. Default 72 (substellar) (int)
        start - Start of meaning period
        end - End of meaning period
        savedir - Directory to save output plot (str)
        save - Save plot or no (Boolean)
        
    Outputs:
        Latitude-height slice plot through cloud condensate
        Longitude-height slice plot through cloud condensate              """

    for cube in cubes:
        if cube.standard_name == 'mass_fraction_of_cloud_ice_in_air':
            ice = cube[start:end, :, :, :].copy()
        if cube.standard_name == 'mass_fraction_of_cloud_liquid_water_in_air':
            liq = cube[start:end, :, :, :].copy()
     # Extract datacubes to plot

    ice = ice.collapsed('time', iris.analysis.MEAN)
    liq = liq.collapsed('time', iris.analysis.MEAN)
    # Find time means for ice and liquid condensate cubes

    heights = np.round(ice.coord('level_height').points*1e-03, 0)
    lons = ice.coord('longitude').points
    lats = ice.coord('latitude').points
    # Extract height, lon, and lat coordinates for plotting
    total_cloud = ice + liq
    # Add ice and liquid mass fractions to get total cloud mass fraction

    if len(heights) > 39:
        final_height = 35
    else:
        final_height = 25
    # To make sure all plots extend to the same height, we fix the top level
    # to be plotted. This differs for ProxB and TRAP-1e. We stop at 25 km.
    #  25 km = level 35 for ProxB, level 25 for TRAP-1e

    fig1, ax1 = plt.subplots(figsize=(5, 5))
    plota = ax1.contourf(np.roll(lons, 72), heights[:final_height],
                         total_cloud[:final_height, lat, :].data*10**4,
                         np.arange(0, 3.5, 0.1), cmap='Blues')
    ax1.set_title('Total cloud at latitude %s' % lats[lat])
    ax1.set_xlabel('Longitude [degrees]')
    ax1.set_xticks([0, 90, 180, 270, 360], ['180W', '90W', '0', '90E', '180E'])
    ax1.set_ylabel('Height [km]')
    cba = plt.colorbar(plota)
    cba.ax.set_title('$10^{-4}$ kg/kg', size=10)
    # Create height-longitude plot

    if save == True:
        plt.savefig(savedir + 'cloud_bubble_lat_%s_prox.eps' % lats[lat],
                    format='eps', bbox_inches='tight')
    else:
        pass
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    plotb = ax2.contourf(lats, heights[:final_height],
                         total_cloud[:final_height, :, lon+71].data*10**4,
                         np.arange(0, 3.5, 0.1), cmap='Blues')
    ax2.set_title('Total cloud at longitude %s' % np.round(lons[lon+71], 0))
    ax2.set_xlabel('Latitude [degrees]')
    ax2.set_xticks([-90, -60, -30, 0, 30, 60, 90],
                   ['90S', '60S', '30S', '0', '30N', '60N', '90N'])
    ax2.set_ylabel('Height [km]')
    cbb = plt.colorbar(plotb)
    cbb.ax.set_title('$10^{-4}$ kg/kg', size=10)
    # Create height-latitude plot

    if save == True:
        plt.savefig(savedir + 'cloud_bubble_lon_%s_prox.eps' % lons[lon],
                    format='eps', bbox_inches='tight')
    else:
        pass
    plt.show()
