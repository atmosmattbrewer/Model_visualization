from netCDF4 import Dataset
import numpy as np
from wrf import(getvar,interplevel, to_np, latlon_coords, interpline, get_cartopy, cartopy_xlim, cartopy_ylim,ALL_TIMES, vertcross, smooth2d, CoordPair, GeoBounds)
import wrf
import glob
import xarray as xr
import warnings
import os
warnings.filterwarnings('ignore')

def grid_xy(lat, lon, mlat,mlon):# The Lat/lon of paradise
    wp_lat = lat
    wp_lon = lon
    #Subtracting the station lat/lon from all the lat/lon points, them from there finding the minimum difference which gives the closest grid box index
    abslat = np.abs(mlat-wp_lat)
    abslon= np.abs(mlon-wp_lon)
    c = np.maximum(abslon,abslat)
    xx, yy = np.where(c == np.min(c))
    return(xx[0],yy[0])

allFiles = glob.glob('/srv/home/mbrewer/wrf_out/acm2/wrfout_d03_2018-11*')
print ('you have %s WRF files to merge' %(len(allFiles)))
truck_lon = float(input('Enter location lon: '))
truck_lat = float(input('Enter location lat: '))
ts_name = input('Enter single column name: ')
for k,i in enumerate(allFiles,0): #pull time from import dictionaries and append to list
    ds = Dataset(i)
    time = getvar(ds, "times",timeidx=ALL_TIMES)
    z=getvar(ds,'z',timeidx=ALL_TIMES)
    lats, lons = latlon_coords(z)
    u, v =getvar(ds, 'uvmet', timeidx = ALL_TIMES)
    w = getvar(ds, 'wa', timeidx = ALL_TIMES)
    wspd,wdir = getvar(ds,'uvmet_wspd_wdir', timeidx = ALL_TIMES)
    x, y = grid_xy(truck_lat, truck_lon, lats, lons)
    u = u[:,:,x,y]
    w = w[:,:,x,y]
    v = v[:,:, x,y]
    z = z[:,:, x,y]
    wspd = wspd[:,:,x,y]
    wdir = wdir[:,:,x,y]
    sds = u.to_dataset('u')
    sds['v'] = v
    sds['w'] = w
    sds['z'] = z
    sds['wspd'] = wspd
    sds['wdir'] = wdir
    del sds['u'].attrs['projection']
    del sds['v'].attrs['projection']

    del sds['u'].attrs['coordinates']
    del sds['v'].attrs['coordinates']
    del sds['z'].attrs['projection']

    del sds['wspd'].attrs['coordinates']

    del sds['wspd'].attrs['projection']

    del sds['wdir'].attrs['coordinates']

    del sds['wdir'].attrs['projection']

    del sds['z'].attrs['coordinates']
    del sds['w'].attrs['projection']

    del sds['w'].attrs['coordinates']
    print('')
    print('processing WRF file at start time: %s' %(sds.Time[0].data))
    print('')

    if k ==0:
        sds.to_netcdf('%s_scm0.nc'%(ts_name))
    else:
        ts = xr.open_dataset('%s_scm%s.nc'%(ts_name,int(k-1)))
        tss = xr.merge([ts, sds])
        tss.to_netcdf('%s_scm%s.nc'%(ts_name,k))

    """ Due to xarray being super nice, the order that files read in doesn't matter
        the back end datetime functionality put the datat into the right order automatically """
mv = 'mv %s_scm%s.nc Final_%s_scm.nc'%(ts_name,k,ts_name)
os.system(mv)
print('')
print('Final_%s_scm.nc is the final merged single column'%(ts_name))
print('')
print('Removing extra single column files')
print('')
rm = 'rm -f %s_scm*'%(ts_name)
os.system(rm)
print('Script Complete')
