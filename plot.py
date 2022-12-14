from datetime import datetime

import cartopy.crs as ccrs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def moving_average(arr, n=3):
    ret = np.cumsum(arr, dtype=float)
    ret[n:] -= ret[:-n]
    res = ret[n - 1:] / n
    res = np.concatenate((np.array([res[0]] * (n - 1)), res))
    return res


def plot_input_data(data_u, data_v, data_sp, data_metss, data_zh, p_level, lons, lats, time):
    data_windspeed = np.sqrt(data_u ** 2 + data_v ** 2)
    data_windspeed_summer_mean = np.mean(data_windspeed, axis=0)
    data_u_mean = np.mean(data_u, axis=0)
    data_v_mean = np.mean(data_v, axis=0)

    data_u_mean_norm = data_u_mean / data_windspeed_summer_mean
    data_v_mean_norm = data_v_mean / data_windspeed_summer_mean

    data_sp_mean = np.mean(data_sp, axis=0)
    data_metss_mean = np.mean(data_metss, axis=0)
    data_zh_mean = np.mean(data_zh, axis=0)

    data = zip(
        ['sp', 'metsss', 'zh'],
        [data_sp_mean, data_metss_mean, data_zh_mean])

    # Теперь построим карты средней за лето скорости ветра для каждой из высот (уровней давления)

    data_crs = ccrs.PlateCarree()

    # Можем создать собственную цветовую палитру для карт, выбрав нужные цвета и их порядок
    newcmp = LinearSegmentedColormap.from_list('', ['white', 'green', 'yellow', 'orange', 'red', 'purple'])

    # Создаём цикл, в котором итератор p_i пройдёт по всем уровням давления - будет меняться от 0
    # до длины массива уровней давления (p_level) невключительно, то есть примет значения 0, 1 и 2.
    for name, el in zip(['Wind speed'], [data_windspeed_summer_mean]):
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

            cbar = plt.colorbar(fraction=0.02)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.set_label(name, rotation=270)

            delta_epoch = datetime(1970, 1, 1, 0, 0, 0) - datetime(1900, 1, 1, 0, 0, 0)
            date_start = datetime.fromtimestamp(time[0] * 3600) - delta_epoch
            date_end = datetime.fromtimestamp(time[-1] * 3600) - delta_epoch
            plt.title(
                f'Mean {name} for {date_start.strftime("%m.%Y")}-{date_end.strftime("%m.%Y")} at {p_level[p_i]} hPa')

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

        cbar = plt.colorbar(fraction=0.02)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label('Wind speed [m/s]', rotation=270)

        delta_epoch = datetime(1970, 1, 1, 0, 0, 0) - datetime(1900, 1, 1, 0, 0, 0)
        date_start = datetime.fromtimestamp(time[0] * 3600) - delta_epoch
        date_end = datetime.fromtimestamp(time[-1] * 3600) - delta_epoch
        plt.title(
            f'Mean wind speed for {date_start.strftime("%m.%Y")}-{date_end.strftime("%m.%Y")} at {p_level[p_i]} hPa')

        plt.show()

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

        cbar = plt.colorbar(fraction=0.02)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label(name, rotation=270)

        delta_epoch = datetime(1970, 1, 1, 0, 0, 0) - datetime(1900, 1, 1, 0, 0, 0)
        date_start = datetime.fromtimestamp(time[0] * 3600) - delta_epoch
        date_end = datetime.fromtimestamp(time[-1] * 3600) - delta_epoch
        plt.title(f'Mean {name} for {date_start.strftime("%m.%Y")}-{date_end.strftime("%m.%Y")}')

        plt.show()


def plot_output(second_component, third_component, lons, lats, time, n_month=0):
    third_component_mean_in_month = np.mean(third_component[n_month::3], axis=0)
    second_component_mean_in_month = np.mean(second_component[n_month::3], axis=0)

    data = zip(
        ['second component', 'third component'],
        [second_component_mean_in_month, third_component_mean_in_month])

    # Теперь построим карты средней за лето скорости ветра для каждой из высот (уровней давления)

    data_crs = ccrs.PlateCarree()

    # Можем создать собственную цветовую палитру для карт, выбрав нужные цвета и их порядок
    newcmp = LinearSegmentedColormap.from_list('', ['white', 'green', 'yellow', 'orange', 'red', 'purple'])

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

        plt.contourf(lons, lats, el, levels=range(start, end, step), transform=data_crs, cmap=newcmp)
        ax.coastlines()
        cbar = plt.colorbar(fraction=0.02)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label(name, rotation=270)

        delta_epoch = datetime(1970, 1, 1, 0, 0, 0) - datetime(1900, 1, 1, 0, 0, 0)
        date = datetime.fromtimestamp(time[n_month] * 3600) - delta_epoch
        plt.title(f'Mean {name} in {date.strftime("%m")} for 10 years')

        plt.show()


def plot_second_component(second_component_lam_mean, lats, time, window=20):
    data_x2 = second_component_lam_mean[::3]
    for t_i in range(10):
        f = plt.figure(figsize=(18, 9))
        ax1 = f.add_subplot()
        ax1.set_xlabel('Latitude')
        ax1.set_ylabel('F1', color='tab:red')
        ax2 = ax1.twinx()
        ax2.set_ylabel('F1 moving average', color='tab:blue')

        data_lines = []

        d, = ax2.plot(lats, moving_average(data_x2[t_i], window), color='xkcd:cornflower', label='F1 moving average')
        data_lines.append(d)
        ax2.axhline(y=0, color='r', linestyle='-')

        l = ax1.legend(handles=data_lines, bbox_to_anchor=(0.95, 1.25), loc=9, title='Обозначения')
        ax1.add_artist(l)

        delta_epoch = datetime(1970, 1, 1, 0, 0, 0) - datetime(1900, 1, 1, 0, 0, 0)
        date = datetime.fromtimestamp(time[t_i] * 3600) - delta_epoch
        plt.title(f'mean F1 in {date.strftime("%m")} for 10 years')
        plt.show()


def plot_second_and_third(second_component_lam_mean, third_component_lam_mean, lats, time):
    for t_i in range(3):
        data_x_2 = np.mean(second_component_lam_mean[t_i::3], axis=0)
        data_x_3 = np.mean(third_component_lam_mean[t_i::3], axis=0)
        f = plt.figure(figsize=(18, 9))
        ax1 = f.add_subplot()
        ax1.set_xlabel('Latitude')

        data_lines = []
        d, = ax1.plot(lats, moving_average(data_x_2, 1), color='xkcd:darkblue', label='F1')
        data_lines.append(d)

        d, = ax1.plot(lats, data_x_3, color='xkcd:cornflower', label='F2')
        data_lines.append(d)

        l = ax1.legend(handles=data_lines, bbox_to_anchor=(0.95, 1.25), loc=9, title='Обозначения')
        ax1.add_artist(l)

        delta_epoch = datetime(1970, 1, 1, 0, 0, 0) - datetime(1900, 1, 1, 0, 0, 0)
        date = datetime.fromtimestamp(time[t_i] * 3600) - delta_epoch
        plt.title(f'F1 and F2 in {date.strftime("%m")}')
        plt.show()
