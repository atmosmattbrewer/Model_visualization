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
truck_lon = float(input('Enter station lon: '))
truck_lat = float(input('Enter station lat: '))
ts_name = input('Enter timeseries name: ')
for k,i in enumerate(allFiles,0): #pull time from import dictionaries and append to list
    ds = Dataset(i)
    d = xr.open_dataset(i)
    time = getvar(ds, "times",timeidx=ALL_TIMES)
    rh_2 = getvar(ds, 'rh2', timeidx = ALL_TIMES)
    lats, lons = latlon_coords(rh_2)
    u, v =getvar(ds, 'uvmet10', timeidx = ALL_TIMES)
    x, y = grid_xy(truck_lat, truck_lon, lats, lons)
    u_10 = u[:,x,y]
    v_10 = v[:,x,y]
    T2 = d.T2[:,x,y]
    rh2 = rh_2[:,x,y]
    sds = T2.to_dataset('T2')
    sds['u10'] = u_10
    sds['v10'] = v_10
    sds['rh2'] = rh2
    del sds['u10'].attrs['projection']
    del sds['v10'].attrs['projection']
    del sds['rh2'].attrs['projection']

    del sds['u10'].attrs['coordinates']
    del sds['v10'].attrs['coordinates']
    del sds['rh2'].attrs['coordinates']
    print('')
    print('processing WRF file at start time: %s' %(sds.Time[0].data))
    print('')
    if k ==0:
        sds.to_netcdf('%s_timeseries0.nc'%(ts_name))
    else:
        ts = xr.open_dataset('%s_timeseries%s.nc'%(ts_name,int(k-1)))
        tss = xr.merge([ts, sds])
        tss.to_netcdf('%s_timeseries%s.nc'%(ts_name,k))

    """ Due to xarray being super nice, the order that files read in doesn't matter
        the back end datetime functionality put the datat into the right order automatically """
print('')
print('timeseries%s is the final merged timeseries'%(len(allFiles)-1))
print('')
mv = 'mv %s_timeseries%s.nc Final_%s_timeseries.nc'%(ts_name,k,ts_name)
os.system(mv)
print('')
print('Final_%s_timeseries.nc is the final merged timeseries'%(ts_name))
print('')
print('Removing extra timeseries files')
print('')
rm = 'rm -f %s_timeseries*'%(ts_name)
os.system(rm)
print('Script Complete')
