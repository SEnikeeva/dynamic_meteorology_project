from datetime import datetime

import cartopy.crs as ccrs
import netCDF4 as nc
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

filename = 'p-levels_t_u_v_geopotential.nc'
dataset = nc.Dataset(filename)
print(dataset)


lats = dataset['latitude'][:]
lons = dataset['longitude'][:]
time = dataset['time'][:]
p_level = dataset['level'][:]

data_u = dataset['u'][:]
data_v = dataset['v'][:]
data_t = dataset['t'][:]
data_z = dataset['z'][:]


data_windspeed = np.sqrt(data_u ** 2 + data_v ** 2)
data_windspeed_summer_mean = np.mean(data_windspeed, axis=0)
data_u_mean = np.mean(data_u, axis=0)
data_v_mean = np.mean(data_v, axis=0)
data_t_mean = np.mean(data_t, axis=0)
data_z_mean = np.mean(data_z, axis=0)

data_u_mean_norm = data_u_mean / data_windspeed_summer_mean
data_v_mean_norm = data_v_mean / data_windspeed_summer_mean


# Теперь построим карты средней за лето скорости ветра для каждой из высот (уровней давления)

data_crs = ccrs.PlateCarree()

# Можем создать собственную цветовую палитру для карт, выбрав нужные цвета и их порядок
newcmp = LinearSegmentedColormap.from_list('', ['white', 'green', 'yellow', 'orange', 'red', 'purple'])

# Создаём цикл, в котором итератор p_i пройдёт по всем уровням давления - будет меняться от 0 
# до длины массива уровней давления (p_level) невключительно, то есть примет значения 0, 1 и 2. 
for name, el in zip(['t', 'z', 'Wind speed'], [data_t_mean, data_z_mean, data_windspeed_summer_mean]):
    for p_i in range(len(p_level)):
        f = plt.figure(figsize=(18, 9))
        f.patch.set_facecolor('white')

        if name == 'Wind speed':
            start = 0
            end = 25
            step = 1
        else:
            step = int((el[p_i, :, :].max() - el[p_i, :, :].min()) // 10)
            start = round(el[p_i, :, :].min()) - step
            end = round(el[p_i, :, :].max()) + step

        ax = plt.axes(projection=data_crs)
        # Можем указать границы для рисуемой области карты (но в данном примере нарисуем для всей Земли)
        # ax.set_extent([0, 180, 0, 90], crs = data_crs)

        plt.contourf(lons, lats, el[p_i, :, :], levels=range(start, end, step), transform=data_crs, cmap=newcmp)
        ax.coastlines()

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='black', alpha=0.2, linestyle='--')

        cbar = plt.colorbar(fraction=0.02)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label(name, rotation=270)

        delta_epoch = datetime(1970, 1, 1, 0, 0, 0) - datetime(1900, 1, 1, 0, 0, 0)
        date_start = datetime.fromtimestamp(time[0] * 3600) - delta_epoch
        date_end = datetime.fromtimestamp(time[-1] * 3600) - delta_epoch
        plt.title(f'Mean {name} for {date_start.strftime("%m.%Y")}-{date_end.strftime("%m.%Y")} at {p_level[p_i]} hPa')

        plt.show()


for p_i in range(len(p_level)):
    f = plt.figure(figsize=(18, 9))
    f.patch.set_facecolor('white')

    ax = plt.axes(projection=data_crs)

    start = 0
    end = -1
    step = 20
    plt.contourf(lons[start:end:step], lats[start:end:step],
                 data_windspeed_summer_mean[p_i, start:end:step, start:end:step], levels=range(0, 25, 1),
                 transform=data_crs, cmap=newcmp)
    ax.coastlines()

    plt.quiver(lons[start:end:step], lats[start:end:step], data_u_mean_norm[p_i, start:end:step, start:end:step],
               data_v_mean_norm[p_i, start:end:step, start:end:step], pivot='middle')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='black', alpha=0.2, linestyle='--')

    cbar = plt.colorbar(fraction=0.02)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label('Wind speed [m/s]', rotation=270)

    delta_epoch = datetime(1970, 1, 1, 0, 0, 0) - datetime(1900, 1, 1, 0, 0, 0)
    date_start = datetime.fromtimestamp(time[0] * 3600) - delta_epoch
    date_end = datetime.fromtimestamp(time[-1] * 3600) - delta_epoch
    plt.title(f'Mean wind speed for {date_start.strftime("%m.%Y")}-{date_end.strftime("%m.%Y")} at {p_level[p_i]} hPa')

    plt.show()


filename2 = 'adaptor.mars.internal-1668087999.639294-29643-4-963282dc-f53a-415d-9b63-bc914ed135de.nc'
# zust - Friction velocity
# z - Geopotential
# metss - Mean eastward turbulent surface stress
# mngwss - Mean northward gravity wave surface stress
# sp - Surface pressure
dataset2 = nc.Dataset(filename2)
print(dataset2)


lats = dataset2['latitude'][:]
lons = dataset2['longitude'][:]
time = dataset2['time'][:]

data_sp = dataset2['sp'][:]
data_metss = dataset2['metss'][:]
data_mngwss = dataset2['mngwss'][:]
data_z = dataset2['z'][:]
data_zust = dataset2['zust'][:]

g = 9.80665
data_zh = data_z / g


data_sp_mean = np.mean(data_sp, axis=0)
data_metss_mean = np.mean(data_metss, axis=0)
data_mngwss_mean = np.mean(data_mngwss, axis=0)
data_z_mean = np.mean(data_z, axis=0)
data_zh_mean = np.mean(data_zh, axis=0)
data_zust_mean = np.mean(data_zust, axis=0)

data = zip(
    ['sp', 'metss', 'mngwss', 'zh', 'zust'],
    [data_sp_mean, data_metss_mean, data_mngwss_mean, data_zh_mean, data_zust_mean])


for name, el in data:
    f = plt.figure(figsize=(18, 9))
    f.patch.set_facecolor('white')

    step = int((el.max() - el.min()) // 10)
    step = 1 if step == 0 else step
    start = round(el.min()) - step
    end = round(el.max()) + step
    if step > 1:
        start -= step
        end += step

    ax = plt.axes(projection=data_crs)
    # Можем указать границы для рисуемой области карты (но в данном примере нарисуем для всей Земли)
    # ax.set_extent([0, 180, 0, 90], crs = data_crs)

    plt.contourf(lons, lats, el, levels=range(start, end, step), transform=data_crs, cmap=newcmp)
    ax.coastlines()

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='black', alpha=0.2, linestyle='--')

    cbar = plt.colorbar(fraction=0.02)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label(name, rotation=270)

    delta_epoch = datetime(1970, 1, 1, 0, 0, 0) - datetime(1900, 1, 1, 0, 0, 0)
    date_start = datetime.fromtimestamp(time[0] * 3600) - delta_epoch
    date_end = datetime.fromtimestamp(time[-1] * 3600) - delta_epoch
    plt.title(f'Mean {name} for {date_start.strftime("%m.%Y")}-{date_end.strftime("%m.%Y")}')

    plt.show()
