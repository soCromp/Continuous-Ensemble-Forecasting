# activate svd

from pyproj import Proj 
from scipy.interpolate import RegularGridInterpolator
import pandas as pd 
import xarray as xr 
import numpy as np 
import zarr 
import os
from tqdm import tqdm 
from random import randint
import datetime 

promptpath = '/home/cyclone/train/multivar/0.25/date/natlantic/test'
predpath = '/mnt/data/sonia/cef/results/multivar/3/final/continuous-24+6h.zarr'
outpath = '/mnt/data/sonia/cef/patches/multivar/3'
os.makedirs(outpath, exist_ok=True)

# preds size (13145, 10, 8, 5, 32, 64): (t, ensemble, predictions into future, lat, lon)
preds = zarr.open(predpath, mode='r')
test_interval = pd.date_range(datetime.datetime(2016,1,1,6), datetime.datetime(2024,12,31,23), freq='6h')
START_TIME = datetime.datetime(2016,1,1,6)


regmask = xr.open_dataset('/home/cyclone/regmask_0723_anl.nc')

####### make dataframe of all tracks 
trackspath1='/mnt/data/sonia/mcms/tracker/1940-2010/era5/out_era5/era5/mcms_era5_1940_2010_tracks.txt'
trackspath2='/mnt/data/sonia/mcms/tracker/2010-2024/era5/out_era5/era5/FIXEDmcms_era5_2010_2024_tracks.txt'
joinyear = 2010 # overlap for the track data

tracks1 = pd.read_csv(trackspath1, sep=' ', header=None, 
        names=['year', 'month', 'day', 'hour', 'total_hrs', 'unk1', 'unk2', 'unk3', 'unk4', 'unk5', 'unk6', 
               'z1', 'z2', 'unk7', 'tid', 'sid'])
# storms that start before the join year (even if they continue into the join year):
sids1 = tracks1[(tracks1['sid']==tracks1['tid']) & (tracks1['year']<joinyear)]['sid'].unique()
tracks1 = tracks1[tracks1['sid'].isin(sids1)]

tracks2 = pd.read_csv(trackspath2, sep=' ', header=None, 
        names=['year', 'month', 'day', 'hour', 'total_hrs', 'unk1', 'unk2', 'unk3', 'unk4', 'unk5', 'unk6', 
               'z1', 'z2', 'unk7', 'tid', 'sid'])
# filter out storms that "start" at the beginning of the join year since they probably started before and are 
# included in tracks1
sids2 = tracks2[(tracks2['sid']==tracks2['tid']) & \
        ((tracks2['year']>=joinyear) | (tracks2['month']>1) | (tracks2['day']>1) | (tracks2['hour']>0))]['sid'].unique()
tracks2 = tracks2[tracks2['sid'].isin(sids2)]

tracks = pd.concat([tracks1, tracks2], ignore_index=True)
tracks = tracks.sort_values(by=['year', 'month', 'day', 'hour'])

# conversions from the MCMS lat/lon system, as described in Jimmy's email:
tracks['lat'] = 90-tracks['unk1'].values/100
tracks['lon'] = tracks['unk2'].values/100

tracks = tracks[['year', 'month', 'day', 'hour', 'tid', 'sid', 'lat', 'lon']]

basin = 'natlantic'
truetrain = set(os.listdir(f'/home/cyclone/train/windmag/500hpa/0.25/date/{basin}/train'))
trueval = set(os.listdir(f'/home/cyclone/train/windmag/500hpa/0.25/date/{basin}/val'))
truetest = set(os.listdir(f'/home/cyclone/train/windmag/500hpa/0.25/date/{basin}/test'))

tracks['split'] = 0
tracks.loc[tracks['sid'].isin(truetest), 'split'] = 2
tracks.loc[tracks['sid'].isin(trueval), 'split'] = 1
tracks = tracks[tracks['split']>1] # only storms in basin and in val or test split

sids = tracks['sid'].unique().tolist()
print('num storms:', len(sids))
print(tracks.iloc[-30:])

resolution = 5.625
l = 800 # (half length: l/2 km from center in each direction)
s = 32 # box will be dimensions s by s (eg 32x32)
### HERE WE REINTRODUE THE N/S FLIP: ###
x_lin = np.linspace(-l, l, s)
y_lin = np.linspace(-l, l, s)
x_grid, y_grid = np.meshgrid(x_lin, y_lin) # equal-spaced points from -l to l in both x and y dimensions


for sid in tqdm(sids):
    records = tracks[tracks['sid']==sid].to_dict('records')
    prompt = np.load(os.path.join(promptpath, str(sid), '0.npy'))
    ensemble_ind = randint(0,9) # inclusive
    
    # start with stamp 1 since 0 will come from prompt:
    target_date = pd.Timestamp(records[1]['year'], records[1]['month'], records[1]['day'], records[1]['hour'])
    index = int((target_date - START_TIME) / pd.Timedelta(hours=6))
    try:
        worlds = preds[index, ensemble_ind, :len(records)-1] 
    except:
        print(target_date, index)
        break

    boxes = [prompt]
    lat_center, lon_center = records[0]['lat'], records[0]['lon'] % 360
    track = [(lat_center, lon_center)]
        
    for t, (record, world) in enumerate(zip(records, worlds)): # iterates over time
        lats = np.linspace(90, -90, 32, endpoint=False)
        lons = np.linspace(0, 360, 64, endpoint=False)
        data_vars={'slp': (('lat', 'lon'), world[0]),
                    'u': (('lat', 'lon'), world[1]),
                    'v': (('lat', 'lon'), world[2]),
                    't': (('lat', 'lon'), world[3]),
                    'q': (('lat', 'lon'), world[4])}
        
        ds = xr.Dataset(
            data_vars=data_vars, 
            coords={"lat": lats, "lon": lons}, # Attach the coordinates
        )
        
        # Flip strictly to make coordinates ascending for RegularGridInterpolator
        ds = ds.reindex(lat=ds.lat[::-1])
        
        # LAZY TRACKER: Track SLP minimum using Geopotential (z) as a physical proxy
        key='slp'
        search_radius = 10.0
        z_search = ds[key].sel(
            lat=slice(lat_center - search_radius, lat_center + search_radius),
            lon=slice(lon_center - search_radius, lon_center + search_radius)
        )
        if z_search.size > 0: 
            min_idx = np.unravel_index(z_search.argmin().values, z_search.shape)
            lat_center = z_search.lat[min_idx[0]].values.item()
            lon_center = z_search.lon[min_idx[1]].values.item()
                
        track.append((lat_center, lon_center))

        proj_km = Proj(proj='aeqd', lat_0=lat_center, lon_0=lon_center, units='km')
        lon_grid, lat_grid = proj_km(x_grid, y_grid, inverse=True) #translate km to deg
        lon_grid=(lon_grid+360)%360 # because these datasets have lon as 0 to 360 (lat is still -90 to 90)
        lon_min = lon_grid.min() - resolution # +- reso because otherwise xarray will not include the edge points
        lon_max = lon_grid.max() + resolution
        lat_min = lat_grid.min() - resolution
        lat_max = lat_grid.max() + resolution

        selection = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        arealats = selection.lat.values 
        arealons = selection.lon.values
        data = selection.to_array().values
        data_transposed = np.transpose(data, (1, 2, 0)) # now H W V
        
        interp = RegularGridInterpolator(
            (arealats, arealons),
            data_transposed,
            bounds_error=False,
            fill_value=None
        )

        # Interpolate at new (lat, lon) pairs
        interp_points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
        interp_values = interp(interp_points).reshape(s, s, data.shape[0])
        
        slp_slice = interp_values[:, :, 0]
        u_slice = interp_values[:, :, 1]
        v_slice = interp_values[:, :, 2]
        t_slice = interp_values[:, :, 3]
        q_slice = interp_values[:, :, 4]
        
        frame = np.stack([slp_slice, u_slice, v_slice, t_slice, q_slice], axis=-1) # We want H x W x V
        boxes.append(frame)

    # print([box.shape for box in boxes])
    os.makedirs(os.path.join(outpath, sid), exist_ok=True)
    for i in range(len(boxes)):
        np.save(os.path.join(outpath, sid, f'{i}.npy'), boxes[i])
    with open(os.path.join(outpath, sid, 'track.csv'), 'w') as f:
        f.write('t,lat,lon\n')
        for t, (lat, lon) in enumerate(track):
            f.write(f'{t},{lat},{lon}\n')
            
            