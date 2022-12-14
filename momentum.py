# Задача 5. Оценить поток момента импульса через подстилающую поверхность по данным реанализа.

import netCDF4 as nc
import numpy as np

from plot import plot_input_data, plot_output, plot_second_component, plot_second_and_third


def get_derivative(x, y):
    if len(y.shape) == 3:
        first_el = np.expand_dims((y[:, :, 1] - y[:, :, 0]) / (x[1] - x[0]), axis=2)
        last_el =  np.expand_dims((y[:, :, -1] - y[:, :, -2]) / (x[-1] - x[-2]), axis=2)
        dy = np.concatenate((first_el, (y[:, :, 2:] - y[:, :, :-2]) / (x[2:] - x[:-2]), last_el), axis=2)
    elif len(y.shape) == 1:
        first_el = y[0] / (x[1] - x[0])
        last_el = y[-1] / (x[-1] - x[-2])
        dy = np.append(first_el, (y[ 2:] - y[:-2]) / (x[2:] - x[:-2]))
        dy = np.append(dy, last_el)
    elif len(y.shape) == 2:
        first_el = np.expand_dims((y[:, 1] - y[:, 0]) / (x[1] - x[0]), axis=1)
        last_el =  np.expand_dims((y[:, -1] - y[:, -2]) / (x[-1] - x[-2]), axis=1)
        dy = np.concatenate((first_el, (y[:, 2:] - y[:, :-2]) / (x[2:] - x[:-2]), last_el), axis=1)
    else:
        dy = None
    return dy


def main():
    filename = 'p-levels_t_u_v_geopotential_10.nc'
    dataset = nc.Dataset(filename)
    print(dataset)

    lats = dataset['latitude'][:]
    lats_rad = (lats / 180) * np.pi

    lons = dataset['longitude'][:]
    lons_rad = (lons / 180) * np.pi

    time = dataset['time'][:]
    p_level = dataset['level'][:]
    data_u = dataset['u'][:]
    data_v = dataset['v'][:]

    filename2 = 'adaptor.mars.internal-1670423241.6391394-30342-7-a4df4ed6-7d0a-44fd-805d-5e72cf65afeb.nc'
    # z - Geopotential
    # metss - Mean eastward turbulent surface stress
    # sp - Surface pressure
    dataset2 = nc.Dataset(filename2)
    print(dataset2)

    data_sp = dataset2['sp'][:]
    data_metss = dataset2['metss'][:]

    g = 9.80665
    a = 6371 * 10 ** 3
    omega = 7.292 * (10 ** -5)

    data_zh = dataset2['z'][:] / g

    cos_lats_rad = np.cos(lats_rad)
    omega_cos_lats_rad = omega * cos_lats_rad
    u_plus_omega_cos_lats_rad = np.array(
        [(data_utp.T + omega_cos_lats_rad).T for data_ut in data_u for data_utp in data_ut]).reshape(data_u.shape)
    M = a * np.array([(utp.T * cos_lats_rad).T for ut in u_plus_omega_cos_lats_rad for utp in ut]).reshape(
        u_plus_omega_cos_lats_rad.shape)
    M_v = M * data_v
    cos_lats_rad_sp = np.array([(el.T * cos_lats_rad).T for el in data_sp])
    dy = np.array([M_v[:, p_i, :, :] * cos_lats_rad_sp for p_i in range(M.shape[1])]).reshape(M.shape)
    dy = np.mean(dy, axis=3)
    dy = np.mean(dy, axis=1)
    first_component = -1 / (a * np.cos(lats_rad)) * get_derivative(lats_rad, dy)

    second_component = -g * data_sp * get_derivative(lons_rad, data_zh)
    second_component_lam_mean = np.mean(second_component, axis=2)

    third_component = -a * g * np.array([(el.T * cos_lats_rad).T for el in data_metss])
    third_component_lam_mean = np.mean(third_component, axis=2)

    d_momentum = first_component + second_component_lam_mean + third_component_lam_mean

    to_plot = False
    if to_plot:
        plot_input_data(data_u, data_v, data_sp, data_metss, data_zh, p_level, lons, lats, time)
        plot_output(second_component, third_component, lons, lats, time, n_month=0)
        plot_second_component(second_component_lam_mean, lats, time, window=20)
        plot_second_and_third(second_component_lam_mean, third_component_lam_mean, lats, time)


if __name__ == '__main__':
    main()
