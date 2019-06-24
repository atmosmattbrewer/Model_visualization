from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import scipy.ndimage as ndimage
from wrf import (getvar, interplevel, to_np, latlon_coords, interpline, get_cartopy, cartopy_xlim, cartopy_ylim,
                 ALL_TIMES, vertcross, smooth2d, CoordPair, GeoBounds)
from metpy.plots import USCOUNTIES
import metpy.calc as mpcalc
from metpy.units import units
import pandas as pd
from satpy import Scene
from satpy import find_files_and_readers
from glob import glob
from satpy.writers import cf_writer
from satpy.writers import get_enhanced_image
from metpy.plots import USCOUNTIES  # Make sure metpy is updated to latest version.
import xarray as xr
from pathlib import Path
import pyart
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import scipy.ndimage as ndimage
from wrf import (getvar, interplevel, to_np, latlon_coords, interpline, get_cartopy, cartopy_xlim, cartopy_ylim,
                 ALL_TIMES, vertcross, smooth2d, CoordPair, GeoBounds)
from metpy.plots import USCOUNTIES
import metpy.calc as mpcalc
from metpy.units import units
import pandas as pd

############################################
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']

####################################################################################
############################ WRF READ-IN #########################################
####################################################################################
ds = Dataset('/export/home/mbrewer/wrf_out/d02/wrfout_d02_2018-11-09_04:00:00')
time = getvar(ds, "times", timeidx=ALL_TIMES)
u, v = getvar(ds, 'uvmet10', timeidx=ALL_TIMES)
lats, lons = latlon_coords(u)
skip = 5

####################################################################################
############################ RADAR READ-IN #########################################
####################################################################################
rad = glob("/export/home/mbrewer/Documents/radar_files/rad_0904/*")
rad2 = glob("/export/home/mbrewer/Documents/radar_files/rad_0905/*")

# %%
### Reading in local road shapefiles to be used with the "base map", the roads help to give some spatial scale and awarenes in my opinion
reader = shpreader.Reader('/export/home/mbrewer/wrf_out/shapefiles/tl_2018_06_prisecroads.shp')
roads = list(reader.geometries())  ## Most major California roadways
roads = cfeature.ShapelyFeature(roads, crs.PlateCarree())

reader = shpreader.Reader('/export/home/mbrewer/wrf_out/shapefiles/tl_2018_06007_roads.shp')
s_roads = list(reader.geometries())  ### All roads in Butte county.... kinda messy
s_roads = cfeature.ShapelyFeature(s_roads, crs.PlateCarree())


# Function used to create the "base map for all of the plots"
def plot_background(ax):
    ax.coastlines(resolution='10m', linewidth=2, color='black', zorder=4)
    political_boundaries = NaturalEarthFeature(category='cultural',
                                               name='admin_0_boundary_lines_land',
                                               scale='10m', facecolor='none')
    states = NaturalEarthFeature(category='cultural',
                                 name='admin_1_states_provinces_lines',
                                 scale='50m', facecolor='none')

    ax.add_feature(political_boundaries, linestyle='-', edgecolor='black', zorder=4)
    ax.add_feature(states, linestyle='-', edgecolor='black', linewidth=2, zorder=4)
    ax.add_feature(USCOUNTIES.with_scale('500k'), edgecolor='black', linewidth=1,
                   zorder=1)  #### Using Metpy's county shapefiles due to hi-resolution and they also help with spartial awareness
    ax.add_feature(roads, facecolor='none', edgecolor='dimgrey', zorder=1, linewidth=1)
    return ax


##### Funtiction used to calculate the Streamwise component of a wind from a specified angle
def streamwise(Ua, Va, deg=30):
    """ Function used to calculated the streamwise component of the wind base off https://www.eol.ucar.edu/content/wind-direction-quick-reference
     deg
    """
    Ugeo = -1 * np.sin(np.deg2rad(deg))
    Vgeo = -1 * np.cos(np.deg2rad(deg))
    D = np.arctan2(Vgeo, Ugeo)
    Us = Ua * np.cos(D) + Va * np.sin(D)
    Vs = -Ua * np.sin(D) + Va * np.cos(D)
    return Us, Vs


def t_ind(Time):
    """ Time = Local time
    input time sting in Y-m-d H:M:S format """
    ds_time = pd.to_datetime(time.data).tz_localize('UTC').tz_convert('US/Pacific')
    T = pd.to_datetime(Time, format='%Y-%m-%d %H:%M:%S').tz_localize('UTC').tz_convert('US/Pacific')
    T_ind = np.where(ds_time == T)
    t = int(T_ind[0])
    ts = str(T)[:-9]
    return t, ts


def VPD(T, RH):
    """ T2 can be either kelvin or celcius, the saturation vapor pressure equation requires celcius but there is a fix in the fucntion
    Saturation Vapor Pressure ased of Bolton 1980:6.112 e^\frac{17.67T}{T + 243.5}
    Vapor Pressure is based of RH = (e/es)*100 ---> e = (RH/100) * es"""

    if T.max() > 200:
        T = T - 273.15
    svp = 6.112 * np.exp((17.67 * T) / (T + 243.5))
    vp = (RH / 100) * svp
    VPD = svp - vp
    return VPD


# %%
td2 = getvar(ds, 'td2', timeidx=ALL_TIMES)
u_10, v_10 = getvar(ds, 'uvmet10', timeidx=ALL_TIMES)

t_list = ['2018-11-09 04:00:00', '2018-11-09 04:10:00', '2018-11-09 04:20:00', '2018-11-09 04:30:00',
          '2018-11-09 04:40:00', '2018-11-09 04:50:00', '2018-11-09 05:00:00', '2018-11-09 05:10:00',
          '2018-11-09 05:20:00', '2018-11-09 05:30:00', '2018-11-09 05:40:00', '2018-11-09 05:50:00']
# t_list = ['2018-11-08 19:00:00', '2018-11-08 19:10:00','2018-11-08 19:20:00','2018-11-08 19:30:00','2018-11-08 19:40:00','2018-11-08 19:50:00']
for i, t in enumerate(t_list):
    t, ts = t_ind(t)
    skip = 10
    cf_var = td2
    u, v = u_10, v_10
    lats, lons = latlon_coords(cf_var)
    cart_proj = get_cartopy(cf_var)
    if i <= 5:
        radar = pyart.io.read_nexrad_archive(rad[-1 * (i + 1)])
        gf = pyart.filters.GateFilter(radar)
        gf.exclude_transition()
        gf.exclude_above('reflectivity', 100)  # Mask out dBZ above 100
        gf.exclude_below('reflectivity', 5)  # Mask out dBZ below 5
        despec = pyart.correct.despeckle_field(radar, 'reflectivity', gatefilter=gf,
                                               size=20)  # The despeckling mask routine that takes out small noisey reflectivity bits not near the main plume
    else:
        radar = pyart.io.read_nexrad_archive(rad2[-1 * (i - 5)])
        gf = pyart.filters.GateFilter(radar)
        gf.exclude_transition()
        gf.exclude_above('reflectivity', 100)  # Mask out dBZ above 100
        gf.exclude_below('reflectivity', 5)  # Mask out dBZ below 5
        despec = pyart.correct.despeckle_field(radar, 'reflectivity', gatefilter=gf,
                                               size=20)  # The despeckling mask routine that takes out small noisey reflectivity bits not near the main plume
    # Create the figure
    fig, ax = plt.subplots(figsize=(30, 20), subplot_kw={'projection': cart_proj}, dpi=300)
    plot_background(ax)

    # Add the color contours
    levels = np.arange(-30, 20, 1)
    cf = ax.contourf(to_np(lons), to_np(lats), to_np(cf_var[t]), levels=levels, cmap='coolwarm_r',
                     transform=crs.PlateCarree(), vmax=5, vmin=-25, zorder=1)
    cb = plt.colorbar(cf, orientation="horizontal", fraction=.045, pad=.001, label=25)
    cb.ax.tick_params(labelsize=25)
    cb.set_label('2-m Dew Point Temp ($\degree$C)', fontsize=30)

    display = pyart.graph.RadarMapDisplayCartopy(radar)
    display.plot_ppi_map('reflectivity', 0, embelish=False,
                         # The "0" is the lowest PPI scan, increasing this number increases the scanning elevation
                         vmin=-10, vmax=64, colorbar_flag=False, fig=fig, ax=ax, projection=crs, title_flag=False,
                         gatefilter=gf)

    display.plot_colorbar(label='Reflectivity (dBZ)', label_size=30, ticklabel_size=25, ax=ax)

    # adding in wind barbs, skip defined above skips that interval of barbs to make the plot more readable
    q = ax.quiver(to_np(lons[::skip, ::skip]), to_np(lats[::skip, ::skip]),
                  to_np(u[t, ::skip, ::skip]), to_np(v[t, ::skip, ::skip]), units='inches', scale=20, width=.03, \
                  transform=ccrs.PlateCarree(), color='k', alpha=.9)
    ax.quiverkey(q, 0.9, 1.01, 15, r'$10 \frac{m}{s}$', coordinates='axes',
                 fontproperties={'size': 30, 'weight': 'bold'})
    # ax.scatter(-121.6219, 39.7596, s =400,  marker = 'X', label = 'Paradise, California', transform = crs.PlateCarree(), color = '#ff0000',edgecolors = 'black')
    # legend = ax.legend(fontsize = 30, loc =  3)
    # legend.get_frame().set_facecolor('#cccfd1')
    # Set the map bounds
    ax.set_xlim(cartopy_xlim(cf_var[0]))
    ax.set_ylim(cartopy_ylim(cf_var[0]))
    # ax.gridlines()

    # Add a title
    # plt.title('WRF %0.1f m'%(ds.DX), loc='left', fontweight='bold', fontsize = 35)
    # plt.title('Color Fill: %s \n Barbs: %s'%(cf_var.description.title(), u.description) , loc='center', fontweight='bold', fontsize = 25)
    # plt.title('%s'%(cf_var.description.title()) , loc='center', fontweight='bold', fontsize = 25)
    plt.title('Valid Time: %s PST\nRadar Scan: %s PST' % (ts, pd.to_datetime(radar.time['units'][14:-1],
                                                                             format='%Y-%m-%dT%H:%M:%S').tz_localize(
        'UTC').tz_convert('US/Pacific').strftime("%Y-%m-%d %H:%M:%S")), loc='left', fontweight='bold', fontsize=25)
    plt.savefig('/export/home/mbrewer/wrf_out/%s.png' % ('radar_dpt_' + str(ds.DX) + '_' + ts), dpi=300,
                bbox_inches='tight')
    plt.show()
# %%
