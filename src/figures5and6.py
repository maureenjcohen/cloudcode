#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:06:23 2023

@author: Mo Cohen
"""
import iris
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import iris.coords
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import scipy as sp
import windspharm


def vertical_profile(cubes, start=500, end=700, level=8, top_level=30,
                     select='absolute', sim='trap',
                     savedir='/exports/csce/datastore/geos/users/s1144983/papers/cloudproject/epsfigs_v2/',
                     save=False):
    """ This function plots the vertical profile of the chosen data input over time.
     Possible inputs: air temperature ('absolute'), potential temperature ('potential'),
     air pressure ('pressure'), vertical wind ('z_wind'), cloud ('cloud') """

    for cube in cubes:
        if cube.standard_name == 'air_potential_temperature':
            theta = cube[start:end, :, :, :].copy()
        if cube.standard_name == 'air_pressure':
            pressure = cube[start:end, :, :, :].copy()
        if cube.standard_name == 'upward_air_velocity':
            z_wind = cube[start:end, :, :, :].copy()
        if (cube.standard_name == 'mass_fraction_of_cloud_ice_in_air' and 
            cube.shape[0] > end-start):
            ice = cube[start:end, :, :, :].copy()
        if (cube.standard_name == 'mass_fraction_of_cloud_liquid_water_in_air'
            and cube.shape[0] > end-start):
            liq = cube[start:end, :, :, :].copy()
        if cube.standard_name == 'x_wind':
            x_wind = cube[start:end, :, :, :].copy()

    heat = mpl_cm.get_cmap('gist_heat')
    blues = mpl_cm.get_cmap('Blues')
    redblu = mpl_cm.get_cmap('RdBu')

    p0 = iris.coords.AuxCoord(
        100000.0, long_name='reference_pressure', units='Pa')
    p0.convert_units(pressure.units)
    # R and cp in J/kgK for 300K
    temperature = theta*((pressure/p0)**(287.05/1005))

    if select == 'absolute':
        datacube = temperature.copy()
        titleterm = 'absolute temperature'
        y_axis = 'Temperature [K]'
        colors = heat
        unit = 'K'
        norm = None
        zmin, zmax, zstep = 270, 291, 2
    elif select == 'potential':
        datacube = theta.copy()
        titleterm = 'potential temperature'
        y_axis = 'Temperature [K]'
        colors = heat
        unit = 'K'
        norm = None
    elif select == 'pressure':
        datacube = pressure.copy()
        titleterm = 'air pressure'
        y_axis = 'Pressure [Pa]'
        colors = blues
        unit = 'Pa'
        norm = None
    elif select == 'z_wind':
        datacube = (z_wind.copy())*1e4
        titleterm = 'vertical wind'
        y_axis = 'Wind speed [m/s]'
        colors = redblu
        unit = '$10^{-4}$ m/s'
        norm = TwoSlopeNorm(0)
        zmin, zmax, zstep = -18, 50, 4
    elif select == 'cloud':
        datacube = (ice.copy() + liq.copy())*1e6
        titleterm = 'cloud (ice + liquid)'
        y_axis = 'Cloud mass [kg/kg]'
        colors = blues
        unit = '$10^{-6}$ kg/kg'
        norm = None
        zmin, zmax, zstep = 0, 61, 3
    elif select == 'x_wind':
        datacube = x_wind.copy()
        titleterm = 'zonal wind'
        y_axis = 'Wind speed [m/s]'
        colors = redblu
        unit = 'm/s'
        norm = TwoSlopeNorm(0)
        zmin, zmax, zstep = -20, 51, 4

    heights = np.round(datacube.coord('level_height').points*1e-03, 0)
    time_axis = np.arange(start, end)
    lats = datacube.coord('latitude')
    lons = datacube.coord('longitude')

    if lats.bounds == None:
        datacube.coord('latitude').guess_bounds()
    if lons.bounds == None:
        datacube.coord('longitude').guess_bounds()

    datacube = datacube.extract(iris.Constraint(
        longitude=lambda v: 270 < v <= 359 or 0 <= v <= 90, 
        latitude=lambda v: -90 <= v <= 90))
    grid_areas = iris.analysis.cartography.area_weights(datacube)
    dayside_mean = datacube.collapsed(
        ['latitude', 'longitude'], iris.analysis.MEAN, weights=grid_areas)
    z_axis = heights[:top_level]
    dayside_time = dayside_mean[:, :top_level].data

    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.set_xlabel('Time [days]', fontsize=14)
    ax1.set_ylabel('%s' % y_axis, fontsize=14)
    ax1.set_title('Dayside mean %s at h=%s km' % (titleterm, heights[level]),
                  fontsize=14)
    plt.plot(time_axis, dayside_mean[:, level].data)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.set_xlabel('Time [days]', fontsize=14)
    ax2.set_ylabel('Height [km]', fontsize=14)
    ax2.set_title('Dayside mean %s' % titleterm, fontsize=14)
    plt.contourf(time_axis, z_axis, dayside_time.T, 
                 levels= np.arange(zmin, zmax, zstep), cmap=colors, norm=norm)
    cbar = plt.colorbar()
    cbar.ax.set_title('%s' % unit)
    cbar.set_ticks(np.arange(zmin, zmax, zstep))

    if save == True:
        plt.savefig(savedir + '/%s_%s_%s_%s.eps' %
                    (select, start, end, sim), format='eps', bbox_inches='tight')
    else:
        pass

    plt.show()


def swresonance(cubes, start=500, end=700, level=0, sim='trap',
                savedir='/exports/csce/datastore/geos/users/s1144983/papers/cloudproject/epsfigs_v2/',
                save=False):

    for cube in cubes:
        if cube.standard_name == 'x_wind':
            x_wind = cube[start:end, :, :, :].copy()
        if cube.standard_name == 'y_wind':
            y_wind = cube[start:end, :, :, :].copy()
        if cube.standard_name == 'surface_net_downward_shortwave_flux':
            heat = cube[start:end, :, :, :].copy()

    y_wind = y_wind.regrid(x_wind, iris.analysis.Linear())
    km_heights = np.round(x_wind.coord('level_height').points*1e-03, 2)

    winds = windspharm.iris.VectorWind(
        x_wind[:, level, :, :], y_wind[:, level, :, :])
    # Create a VectorWind data object from the x and y wind cubes
    uchi, vchi, upsi, vpsi = winds.helmholtz(truncation=21)
    # Calculate the Helmholtz decomposition. Truncation is set to 21 because
    # this is what Hammond and Lewis 2021 used.

    zonal_upsi = upsi.collapsed('longitude', iris.analysis.MEAN)
    zonal_vpsi = vpsi.collapsed('longitude', iris.analysis.MEAN)
    # Calculate zonal means of the x and y components of the rotational component
    eddy_upsi = upsi - zonal_upsi
    eddy_vpsi = vpsi - zonal_vpsi
    magnitude = np.sqrt(eddy_upsi.data**2 + eddy_vpsi.data**2)

    fft2 = sp.fft.fftshift(sp.fftpack.fft2(sp.fft.ifftshift(magnitude)))
    yfreqs = sp.fft.fftshift(sp.fft.fftfreq(fft2.shape[1], d=1./90))
    xfreqs = sp.fft.fftshift(sp.fft.fftfreq(fft2.shape[2], d=1./144))
    psd = np.abs(fft2)**2

    one_zero = psd[:, 45, 73]
    print(xfreqs[73], yfreqs[45])
    two_one = psd[:, 46, 74]
    two_two = psd[:, 47, 74]
    three_two = psd[:, 47, 75]
    one_one = psd[:, 46, 73]
    wave_sum = two_one + two_two + three_two + one_one

    lats = heat.coord('latitude')
    lons = heat.coord('longitude')

    if lats.bounds == None:
        heat.coord('latitude').guess_bounds()
    if lons.bounds == None:
        heat.coord('longitude').guess_bounds()

    grid_areas = iris.analysis.cartography.area_weights(heat)
    global_mean = heat.collapsed(
        ['latitude', 'longitude'], iris.analysis.MEAN, weights=grid_areas)

    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.set_xlabel('Time [days]', fontsize=14)
    ax1.set_ylabel('Surface heating [W/m2]', fontsize=14)
    ax1.plot(global_mean.data, color='k', label='Heating')

    ax2 = ax1.twinx()
    ax2.set_ylabel('PSD', fontsize=14)
    ax2.plot(one_zero, color='r', label='1-0 wave')
    ax2.plot(wave_sum, color='b', label='Wave sum')
    ax2.ticklabel_format(axis='x', style='sci')
    plt.legend()

    plt.title('Mean SW heating and Rossby waves at h=%s km' %
              km_heights[level], y=1.05, fontsize=14)

    if save == True:
        plt.savefig(savedir +
            'swresonance_%s.eps' %sim, format='eps', bbox_inches='tight')
    else:
        pass
    plt.show()
