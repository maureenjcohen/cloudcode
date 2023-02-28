#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:42:42 2023

@author: Mo Cohen
"""

import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import iris.coords
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import iris


def hovmoeller_rwaves(cubes, start=0, end=501, level=8, lats=(55, 85), sim='trap',
                      savedir='/exports/csce/datastore/geos/users/s1144983/papers/cloudproject/epsfigs_v2/',
                      save=False):
    """ Create a Hovmoeller plot of the mean meridional wind between 
    latitudes 55 and 85 North. Directional changes over time at a given
    longitude are the signature of the Rossby gyres moving back and forth
    along a line of latitude. 
    
    Inputs:
        cubes - One Iris datacube
        start - beginning of time period to plot (int)
        end - end of time period to plot (int)
        level - model level (not height) (int)
        lats - bottom and top latitudes of area to be meaned (tuple of ints)
        title - name for labelling output plot when saving (string)
        save - True or False (Boolean)
        
    Outputs:
        Single longitude-time contourfill of mean meridional wind      """

    for cube in cubes:
        if cube.standard_name == 'y_wind':
            y_wind = cube[start:end, level, :, :].copy()
        # Exract y-wind from cube, slicing only the required period and level

    redblu = mpl_cm.get_cmap('coolwarm')
    # Get colormap for contourfill

    time_axis = np.arange(0, y_wind.shape[0])
    lons = y_wind.shape[2]/2
    # Get length of time and longitude axes

    lat_band = y_wind.intersection(latitude=(lats[0], lats[1]))
    # Select latitudes corresponding to our lat input

    band_mean = lat_band.collapsed('latitude', iris.analysis.MEAN)
    band_mean = band_mean.data
    # Calculate mean along each line of longitude from latitude 55 to 85 N

    plt.subplots(figsize=(8, 6))
    plt.contourf(np.arange(-lons, lons), time_axis, 
                 np.roll(band_mean, 72, axis=1),
                 levels=np.arange(-60, 61, 10), 
                 cmap=redblu, 
                 norm=TwoSlopeNorm(0))
    plt.title('Mean meridional wind from %s to %s N' % (lats[0], lats[1]),
              fontsize=16)
    plt.xlabel('Longitude [degrees]', fontsize=14)
    plt.ylabel('Time [days]', fontsize=14)
    plt.xticks((-72, -48, -24, 0, 24, 48, 72), 
               ('180W', '120W', '60W', '0', '60E', '120E', '180E'))
    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(-60, 61, 10))
    cbar.ax.set_title('m/s')
    # Create our contourfill
    # Limits of colorbar are set to cover the full range of wind speeds in
    # all 5 simulations

    if save == True:
        plt.savefig(savedir + 'hov_rwaves%sto%s_%s.eps' % (start, end, sim),
                    format='eps', bbox_inches='tight')
    else:
        pass
    # Save output if desired

    plt.show()

    return
