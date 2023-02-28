#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:21:29 2023

@author: Mo Cohen
"""
import iris
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import iris.coords
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def climgrid(datalist, nlat=90, nlon=144, nlev=38, start=0, end=300, ndata=5,
             savedir='/exports/csce/datastore/geos/users/s1144983/papers/cloudproject/epsfigs_v2/',
             save=False):
                 
    """ Create a full-page grid of plots displaying the vertical temperature 
    profile, zonal mean zonal wind, vertical water vapour profile, and surface 
    temp with wind flow vectors for each dataset in the input datalist.

    Inputs: 
        Datalist - List of 5 Iris CubeLists in the specified order
        nlat - Number of latitudes/rows of datacubes (int)
        nlon - Number of longitudes/columns of datacubes (int)
        nlev - Number of vertical levels (int)
        start - Start day of meaning period (int)
        end - End day of meaning period (int)
        ndata - Number of datacubes in set (int)
        savedir - Directory to save output figure (string)
        save - True or False (Boolean)
        
    Output: 
        ndata x 4 plots in a grid layout in a single image 
        Substellar and antistellar temperature profiles
        Zonal mean zonal wind
        Substellar and antistellar humidity profiles
        Surface temperature                                     """

    names = ['Control ProxB', 'Warm ProxB',
             'Control TRAP1-e', 'Warm TRAP1-e', 'Dry TRAP1-e']
    # Names of the 5 simulations being compared
    # Datacubes MUST be entered in a list in this same order

    redblu = mpl_cm.get_cmap('coolwarm')
    hot = mpl_cm.get_cmap('hot')
    # Impor colormaps for contour fill plots

    fig, ax = plt.subplots(figsize=(16, 22), nrows=5, ncols=4)
    # Create figure axes and number of subplots

    for i in range(ndata):
        data = datalist[i]
        # Extract the CubeList for the i'th simulation
        for cube in data:
            if cube.standard_name == 'air_potential_temperature':
                potential_temp = cube[start:end, :, :, :].copy()
            if cube.standard_name == 'air_pressure':
                air_pressure = cube[start:end, :, :, :].copy()
            if cube.standard_name == 'x_wind':
                x_wind = cube[start:end, :, :, :].copy()
            if cube.standard_name == 'y_wind':
                y_wind = cube[start:end, :, :, :].copy()
            if cube.standard_name == 'surface_temperature':
                surface_temp = cube[start:end, :, :].copy()
            if cube.standard_name == 'specific_humidity' and i < ndata-1:
                spec_humidity = cube[start:end, :, :, :].copy()
            elif cube.standard_name == 'specific_humidity' and i == ndata-1:
                spec_humidity = cube.copy()*0.0
            # From the CubeList, extract each datacube and assign to a variable
            # For the last cube (Dry TRAP-1e), we multiply the specific
            # humidity by 0 to avoid errors in plotting the humidity profile

        y_wind = y_wind.regrid(x_wind, iris.analysis.Linear())
        # Regrid y-wind onto x-wind grid coordinates

        vertical = [('level_height', x_wind.coord('level_height').points)]
        potential_temp = potential_temp.regrid(x_wind, iris.analysis.Linear())
        potential_temp = potential_temp.interpolate(
            vertical, iris.analysis.Linear())
        air_pressure = air_pressure.regrid(x_wind, iris.analysis.Linear())
        air_pressure = air_pressure.interpolate(
            vertical, iris.analysis.Linear())
        # Regrid horizontal and vertical coordinates for potential temperature
        # and air pressure on the grid coordinates for x-wind

        heights = np.round(x_wind.coord('level_height').points*1e-03, 0)
        lats = np.round(x_wind.coord('latitude').points, 0)
        # Extract altitude and latitude values for labelling plots

        p0 = iris.coords.AuxCoord(
            100000.0, long_name='reference_pressure', units='Pa')
        p0.convert_units(air_pressure.units)
        absolute_temp = potential_temp*((air_pressure/p0)**(287.05/1005))
        absolute_temp = np.mean(absolute_temp.data, axis=0)
        # Calculate absolute air temperature from potential temperature
        # and pressure

        zmzw = x_wind.collapsed('longitude', iris.analysis.MEAN)
        zmzw = np.mean(zmzw.data, axis=0)
        # Calculate zonal mean zonal wind and take time mean over input period

        spec_humidity = spec_humidity.regrid(x_wind, iris.analysis.Linear())
        spec_humidity = spec_humidity.interpolate(
            vertical, iris.analysis.Linear())
        spec_humidity = np.mean(spec_humidity.data, axis=0)
        # Regrid specific humidity onto x_wind coordinates and find mean

        surface_temp = np.mean(surface_temp.data, axis=0)
        # Calculate mean surface temperature

        ax[i, 0].plot(absolute_temp[:, 45, 0], heights,
                      color='r', label='Substellar')
        ax[i, 0].plot(absolute_temp[:, 45, 72], heights,
                      color='b', label='Antistellar')
        ax[i, 0].legend()
        ax[0, 0].set_title('Temperature profile [K]', fontsize=14)
        ax[ndata-1, 0].set_xlabel('Temperature [K]', fontsize=14)
        ax[i, 0].set_ylabel('%s \n Height [km]' % names[i], fontsize=18)
        # Create first plot (temperature profile) for i'th datacube
        # Label with name of simulation

        cont = ax[i, 1].contourf(
            lats, heights, zmzw, levels=np.arange(-60, 140, 20), cmap=redblu,
            norm=TwoSlopeNorm(0))
        ax[0, 1].set_title('Zonal mean zonal wind [m/s]', fontsize=14)
        ax[ndata-1, 1].set_xlabel('Latitude [degrees]', fontsize=14)
        fig.colorbar(cont, ax=ax[i, 1], orientation='vertical')
        # Create second plot (contourf of ZMZW) for i'th datacube

        ax[i, 2].plot(spec_humidity[:, 45, 0], heights,
                      color='r', label='Substellar')
        ax[i, 2].plot(spec_humidity[:, 45, 72], heights,
                      color='b', label='Antistellar')
        ax[i, 2].legend()
        ax[0, 2].set_title('Vapor profile [kg/kg]', fontsize=16)
        ax[ndata-1, 2].set_xlabel('Specific humidity [kg/kg]', fontsize=14)
        # Create third plot (humidity profile) for i'th datacube

        surf = ax[i, 3].contourf(np.arange(-72, 72)*2.5, lats, np.roll(
            surface_temp, 72, axis=1), levels=np.arange(100, 400, 20),
            cmap=hot)
        ax[0, 3].set_title('Surface temperature [K]', fontsize=14)
        ax[ndata-1, 3].set_xlabel('Longitude [degrees]', fontsize=14)
        fig.colorbar(surf, ax=ax[i, 3], orientation='vertical')
        # Create fourth plot (surface temperature) for i'th datacube

    # Iterate over all cubes/simulations and fill in the columns of the figure

    if save == True:
        plt.savefig(savedir + 'climgrid_tight.eps',
                    format='eps', bbox_inches='tight')
    else:
        pass
    # Save if happy with output

    plt.show()