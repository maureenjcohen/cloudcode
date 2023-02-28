#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:44:50 2023

@author: Mo Cohen
"""
import glob, os, re
import numpy as np
import matplotlib.pyplot as plt

path1 = '/exports/csce/datastore/geos/users/s1144983/psg_files/proximab_spectra/proximab_spectra/'
path2 = '/exports/csce/datastore/geos/users/s1144983/psg_files/proximab_close_spectra/proximab_close_spectra/'
path3 = '/exports/csce/datastore/geos/users/s1144983/psg_files/trappist1e_control_spectra/trappist1e_control_spectra/'
path4 = '/exports/csce/datastore/geos/users/s1144983/psg_files/trappist1e_close_spectra/trappist1e_close_spectra/'

def keyfunc(x):
    
    numerical = re.compile('\D')
    
    return int(numerical.sub('',x))

def transit_series(path, spectral_line, sim='prox',
                   savedir='/exports/csce/datastore/geos/users/s1144983/papers/cloudproject/epsfigs_v2/',
                   save=False):
    """ 
    spectral_line arguments:
                  H20 (1.4)         CO2 (2.7)
    Control Prox  719               782
    Warm Prox     717               781
    Control Trap  690               756
    Warm Trap     690               755
    """

    if spectral_line in [719, 717, 690]:
        titleline = '1.4'
        titlespec = 'H$_2$O'
        savespec = 'H2O'
    elif spectral_line in [782, 781, 756, 755]:
        titleline = '2.7'
        titlespec = 'CO$_2$'
        savespec = 'CO2'
    else:
        print('No automatic plot title for this line')

    filelist = sorted(glob.glob(str(path) + '*.txt'), key=keyfunc)
    time_axis = np.arange(0, len(filelist))

    transit_depths = []
    for file_number in range(0, len(filelist)):
        open_file = open(filelist[file_number], 'r')
        txt_line = open_file.readlines()
        txt_line = txt_line[spectral_line]
        datapoints = txt_line.split(' ')
        transit_depth = datapoints[10].strip()
        transit_depths.append(transit_depth)
        open_file.close()

    transit_floats = []
    for string in transit_depths:
        data = float(string)
        transit_floats.append(data)

    smean = np.round(np.mean(transit_floats), 2)
    smax = np.round(np.max(transit_floats), 2)
    smin = np.round(np.min(transit_floats), 2)
    p2p = np.round(smax - smin, 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(time_axis, np.array(transit_floats))

    textstr = '\n'.join((
        r'Mean$=%.2f$' % (smean, ),
        r'Max$=%.2f$' % (smax, ),
        r'Min$=%.2f$' % (smin, ),
        r'Diff$=%.2f$' % (p2p,),))

    plt.ylabel('Transit depth [ppm]', fontsize=14)
    plt.xlabel('Time [days]', fontsize=14)
    plt.title('Time series of %s $\mu$m %s feature' % (titleline, titlespec),
              fontsize=14)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.25, textstr, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

    if save == True:
        plt.savefig(savedir + '%s_%s.eps' % (sim, savespec),
                    format='eps', bbox_inches='tight')
    else:
        pass
    plt.show()
