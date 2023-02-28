#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:50:15 2023

@author: Mo Cohen
"""

import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import iris.coords
import numpy as np
import iris


def composite(cubes, time_slice=500, nlat=90, nlon=144, nlev=38, level=8, n=4,
              cloudtype='both', fractype='mass', qscale=5, sim='trap',
              meaning=False,
              savedir='/exports/csce/datastore/geos/users/s1144983/papers/cloudproject/epsfigs_v2/',
              save=False):
    """ Plot composites of the cloud cover and horizontal wind vectors
    
    Inputs: 
        cubes - Iris Datacube 
        time_slice - Time coordinate being plotted (int)
        nlat - Number of latitudes (int)
        nlon - Number of longitudes (int)
        nlev - Number of vertical levels (int)
        level - Level being plotted (int)
        n - Arrow sparsity for quiver plot (int)
        cloudtype - options 'ice', 'liq', 'both','none' (str). Default 'both'.
        fractype - options 'mass' or 'volume'  (str). Default 'mass'.
        qscale - Scale of quivers represent wind vectors in plot (int)
        planet - Name of planet for labelling saved plot (str)
        savedir - Directory to save output plot (str)
        save - Save plot or no (Boolean)

    Outputs: 
        Plot of horizontal wind vectors, with or without the cloud cover 
        superimposed                                                    """

    for cube in cubes:
        if (cube.standard_name == 'x_wind' and meaning == False):
            x_wind = cube[time_slice, :, :, :].copy()
        if (cube.standard_name == 'y_wind' and meaning == False):
            y_wind = cube[time_slice, :, :, :].copy()
        if (cube.standard_name == 'x_wind' and cloudtype == 'none' and meaning == True):
            x_wind = cube.copy()
        if (cube.standard_name == 'y_wind' and cloudtype == 'none' and meaning == True):
            y_wind = cube.copy()
        if (cube.long_name == 'ice_cloud_volume_fraction_in_atmosphere_layer'  
            and fractype == 'volume' and cloudtype != 'none'):
            ice = cube[time_slice, :, :, :].copy()
        if (cube.long_name == 'liquid_cloud_volume_fraction_in_atmosphere_layer'
        and fractype == 'volume' and cloudtype != 'none'):
            liq = cube[time_slice, :, :, :].copy()
        if (cube.standard_name == 'mass_fraction_of_cloud_ice_in_air' and 
        fractype == 'mass' and cloudtype != 'none'):
            ice = cube[time_slice, :, :, :].copy()
        if (cube.standard_name == 'mass_fraction_of_cloud_liquid_water_in_air' 
        and fractype == 'mass' and cloudtype != 'none'):
            liq = cube[time_slice, :, :, :].copy()
    # Import desired cubes based on input arguments
    
    bg = mpl_cm.get_cmap('PuBu')
    # Get colormap for cloud cover
    y_wind = y_wind.regrid(x_wind, iris.analysis.Linear())
    # Regrid y-wind to be on same coordinates as x-wind
    heights = np.round(x_wind.coord('level_height').points*1e-03, 2)
    # Extract height coords in km for labelling plot

    if cloudtype == 'ice':
        cloud = ice
        titleterm = 'Ice cloud'
        titletime = 'day %s' %time_slice
    elif cloudtype == 'liq':
        cloud = liq
        titleterm = 'Liquid cloud'
        titletime = 'day %s' %time_slice
    elif cloudtype == 'both':
        cloud = ice + liq
        titleterm = 'Total cloud'
        titletime = 'day %s' %time_slice
    elif cloudtype == 'none' and meaning == False:
        titleterm = 'Horizontal wind'
        titletime = 'day %s' %time_slice
    elif cloudtype == 'none' and meaning == True:
        x_wind = x_wind.collapsed('time', iris.analysis.MEAN)
        y_wind = y_wind.collapsed('time', iris.analysis.MEAN)
        titleterm = 'Horizontal wind'
        titletime = 'long-term mean'
    else:
        print('Argument cloudtype must be ice, liq, both, or none. Default is both.')
    # Assign values to variables for labelling plot
    
    X, Y = np.meshgrid(np.arange(0, nlon), np.arange(0, nlat))
    # Create meshgrid for quiver plot with dimensions of nlon x nlat

    if cloudtype == 'none':
        # Plot wind vectors WITHOUT cloud cover superimposed
        
        fig, ax = plt.subplots(figsize=(8.5, 5))
        q1 = ax.quiver(X[::n, ::n], Y[::n, ::n], 
                       np.roll(x_wind[level, ::n, ::n].data, 
                               int(nlon/(2*n)), axis=1),
                       np.roll(y_wind[level, ::n, ::n].data, 
                               int(nlon/(2*n)), axis=1), 
                       scale_units='xy', scale=qscale)
        ax.quiverkey(q1, X=0.9, Y=1.05, U=qscale*5, label='%s m/s' %str(qscale*5),
                     labelpos='E', coordinates='axes')
        plt.title('%s, %s, h=%s km' %
                  (titleterm, titletime, heights[level]), fontsize=14)
        plt.xticks((0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144), 
                   ('180W', '150W','120W', '90W', '60W', '30W', '0', '30E', 
                    '60E', '90E', '120E', '150E', '180E'))
        plt.yticks((0, 15, 30, 45, 60, 75, 90),
                   ('90S', '60S', '30S', '0', '30N', '60N', '90N'))
        plt.xlabel('Longitude', fontsize=14)
        plt.ylabel('Latitude', fontsize=14)

        if save == True:
            plt.savefig(savedir + 'quiver_nocloud_%s_%s_%s.eps' %
                        (time_slice, sim, meaning), format='eps', bbox_inches='tight')
        else:
            pass
        plt.show()

    else:
        # Plot wind vectors WITH cloud cover superimposed
        fig, ax = plt.subplots(figsize=(10, 5))
        
        plt.imshow(np.roll(cloud[level, :, :].data*1e4,
                   int(nlon/2), axis=1), vmin=0, vmax=2, cmap=bg, 
                   origin='lower')
        cbar = plt.colorbar()
        # Create image plot of cloud cover first

        if fractype == 'mass':
            cbar.set_label('$10^{-4}$ kg/kg', loc='center')

        q1 = ax.quiver(X[::n, ::n], Y[::n, ::n], 
                       np.roll(x_wind[level, ::n, ::n].data, 
                               int(nlon/(2*n)), axis=1),
                       np.roll(y_wind[level, ::n, ::n].data, 
                               int(nlon/(2*n)), axis=1), 
                       scale_units='xy', scale=qscale)
        ax.quiverkey(q1, X=1.05, Y=1.05, U=qscale*5, label='%s m/s' %(qscale*5),
                     labelpos='E', coordinates='axes')
        # Now superimpose quiver plot over the cloud cover image
        plt.title('%s and horizontal wind, %s, h=%s km' %
                  (titleterm, titletime, heights[level]), fontsize=14)
        plt.xticks((0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144), 
                   ('180W', '150W','120W', '90W', '60W', '30W', '0', '30E', 
                    '60E', '90E', '120E', '150E', '180E'))
        plt.yticks((90, 75, 60, 45, 30, 15, 0),
                   ('90S', '60S', '30S', '0', '30N', '60N', '90N'))
        plt.xlabel('Longitude', fontsize=14)
        plt.ylabel('Latitude', fontsize=14)
        
        if save == True:
            plt.savefig(savedir + 'quiver_withcloud_%s_%s.eps' %
                        (time_slice, sim), format='eps', bbox_inches='tight')
        else:
            pass
        plt.show()
