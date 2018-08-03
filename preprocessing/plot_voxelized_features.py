#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:25:18 2018

@author: anandan
"""

import numpy as np
import pandas as pd
from PC import PyntCloud
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi

path = "/media/anandan/3474068674064B56/CERN/Program/atlas_sim_gan/"
filename = "NTUP_FCS.13289379._000001.pool.root.1"

def get_hits(event_range=range(0,10), layer=0):

    xyzE = []

    with open(path+"data/layer_wise/"+filename+"_Layer_"+str(layer)+".csv", 'r') as f:
        for i,event in enumerate(f):

            if i in event_range:

                event = event.split(';')

                xyzE_temp = np.array([[], [], [], [], []]).T


                for hit in event[:-1]:
                    hit = hit.split(',')

                    xyzE_temp = np.vstack((xyzE_temp, np.array([float(hit[0]),
                                                                float(hit[1]),
                                                                float(hit[2]),
                                                                float(hit[4]),
                                                                layer])))
                xyzE.append(xyzE_temp)

            if i == event_range[-1]:
                break

    return xyzE

def get_events(event_range=range(0,10)):

    xyzE_layer = []

    for layer in [0,1,2,3,12]:
        hits = get_hits(event_range=event_range, layer=layer)
        xyzE_layer.append(hits)

    xyzE = []

    for i in range(len(event_range)):

        xyzE_temp = np.concatenate((xyzE_layer[0][i],
                                    xyzE_layer[1][i],
                                    xyzE_layer[2][i],
                                    xyzE_layer[3][i],
                                    xyzE_layer[4][i]))

        xyzE.append(xyzE_temp)

    return xyzE

def filter_hits_by_dynamic_angle(event_spherical, layer, multiplier=2):

    layer_bool_array = event_spherical.colors == layer

    if layer != 'r':
        event_spherical_layer = event_spherical.loc[layer_bool_array]
    else:
        event_spherical_layer = event_spherical.loc[event_spherical.colors == 'b']

    if event_spherical_layer.shape[0] < 3:
        return event_spherical

    eta_upper = event_spherical_layer.eta.mean() + multiplier*event_spherical_layer.eta.std()
    eta_lower = event_spherical_layer.eta.mean() - multiplier*event_spherical_layer.eta.std()
    phi_upper = event_spherical_layer.phi.mean() + multiplier*event_spherical_layer.phi.std()
    phi_lower = event_spherical_layer.phi.mean() - multiplier*event_spherical_layer.phi.std()

    eta_bool_array = np.logical_and(event_spherical.eta.values<eta_upper, event_spherical.eta>eta_lower)
    phi_bool_array = np.logical_and(event_spherical.phi.values<phi_upper, event_spherical.phi>phi_lower)
    event_bool_array = np.logical_or(np.logical_and(eta_bool_array, phi_bool_array),
                                     np.logical_not(layer_bool_array))

    event_spherical = event_spherical[event_bool_array]

    return event_spherical

def filter_hits_by_angle(event_cylindrical, layer, r_lim, alpha_lim):

    r_lim_l, r_lim_u = r_lim[0], r_lim[1]
    alpha_lim_l, alpha_lim_u = alpha_lim[0], alpha_lim[1]

    layer_bool = (event_cylindrical.colors == layer)

    r_bool_array = np.logical_and(event_cylindrical.r.values<r_lim_u, event_cylindrical.r>r_lim_l)
    alpha_bool_array = np.logical_and(event_cylindrical.alpha.values<alpha_lim_u, event_cylindrical.alpha>alpha_lim_l)

    event_bool_array = np.logical_and(r_bool_array, alpha_bool_array)
    event_bool_array = np.logical_or(event_bool_array, np.logical_not(layer_bool))

    event_cylindrical = event_cylindrical[event_bool_array]

    return event_cylindrical

def voxalize_by_layer(event_cylindrical, layer, segments):

    event_cylindrical_layer_wise = event_cylindrical.loc[event_cylindrical.colors==layer, :]

    if event_cylindrical_layer_wise.shape[0] == 0:
        return np.zeros(shape=(max(1,len(segments[0])-1)*
                               max(1,len(segments[1])-1)*
                               max(1,len(segments[2])-1)))

    ref_cloud = PyntCloud(event_cylindrical_layer_wise)
    voxelgrid_id = ref_cloud.add_structure("voxelgrid", segments=segments)
    feature_vector = ref_cloud.structures[voxelgrid_id].query_voxels(event_cylindrical_layer_wise.reindex(columns=['r','alpha','z']).values,
                                                                     event_cylindrical_layer_wise.reindex(columns=['E']).values.reshape((-1)))

    return feature_vector#.reshape((-1,))


s,e = 0,5
event_range = range(s,e)
xyzE = get_events(event_range)

event=2
event_cartisian = xyzE[event]

event_cartisian = pd.DataFrame(event_cartisian, columns=['x','y','z','E','colors'])

event_cartisian.colors[event_cartisian.colors==0] = 'r'
event_cartisian.colors[event_cartisian.colors==1] = 'b'
event_cartisian.colors[event_cartisian.colors==2] = 'g'
event_cartisian.colors[event_cartisian.colors==3] = 'c'
event_cartisian.colors[event_cartisian.colors==12] = 'm'

event_cartisian = event_cartisian[event_cartisian.E > 0]

eta = pd.read_csv(path+"data/truth_angles/"+filename+"_eta.csv", header=None, usecols=[0, 1, 2, 3])
phi = pd.read_csv(path+"data/truth_angles/"+filename+"_phi.csv", header=None, usecols=[0, 1, 2, 3])
r = pd.read_csv(path+"data/truth_angles/"+filename+"_r.csv", header=None, usecols=[0, 1, 2, 3])

event_r = np.linalg.norm(event_cartisian.loc[:,['x','y']], axis=1)
event_phi = np.arctan2(event_cartisian.loc[:,'y'],event_cartisian.loc[:,'x'])
#if np.abs(event_phi.mean()-phi) > 0.1:
#    continue
event_eta = np.arcsinh(event_cartisian.loc[:,'z']/event_r)

event_delta_phi = event_phi - phi.iloc[event, 0]
event_delta_eta = event_eta - eta.iloc[event, 0]

event_eta_jacobi = np.abs(2*np.exp(-event_eta)/(1+np.exp(-2*event_eta)))
event_phi_mm = event_delta_phi * event_r
event_eta_mm = event_delta_eta * event_eta_jacobi * np.sqrt(event_r**2 + event_cartisian.z**2)

event_r_transformed = np.sqrt(event_phi_mm**2 + event_eta_mm**2)
event_alpha_transformed = np.arctan2(event_phi_mm, event_eta_mm)
event_z_transformed = event_r

data_dict = {'r':event_r_transformed, 'alpha':event_alpha_transformed, 'z':event_z_transformed,
             'E':event_cartisian.E.values, 'colors':event_cartisian.colors.values}

event_cylindrical = pd.DataFrame(data_dict)


event_cylindrical =   filter_hits_by_angle(event_cylindrical,
                                         layer='r',
                                         r_lim=[0, 350],
                                         alpha_lim=[-3.15, 3.15])

event_cylindrical =   filter_hits_by_angle(event_cylindrical,
                                         layer='b',
                                         r_lim=[0, 350],
                                         alpha_lim=[-3.15, 3.15])

event_cylindrical =   filter_hits_by_angle(event_cylindrical,
                                         layer='g',
                                         r_lim=[0, 350],
                                         alpha_lim=[-3.15, 3.15])

event_cylindrical =   filter_hits_by_angle(event_cylindrical,
                                         layer='c',
                                         r_lim=[0, 350],
                                         alpha_lim=[-3.15, 3.15])

event_cylindrical =   filter_hits_by_angle(event_cylindrical,
                                         layer='m',
                                         r_lim=[0, 350],
                                         alpha_lim=[-3.15, 3.15])


fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(event_cylindrical.r, event_cylindrical.alpha, event_cylindrical.z, s=1,
           c=event_cylindrical.colors)
ax.set_xlabel('r')
ax.set_ylabel('phi')
ax.set_zlabel('eta')

layer_0_min = np.ceil(event_cylindrical.loc[event_cylindrical.colors=='r'].z.min())+1
layer_0_max = np.ceil(event_cylindrical.loc[event_cylindrical.colors=='r'].z.max())-1
layer_1_min = np.ceil(event_cylindrical.loc[event_cylindrical.colors=='b'].z.min())+1
layer_1_max = np.ceil(event_cylindrical.loc[event_cylindrical.colors=='b'].z.max())-1
layer_2_min = np.ceil(event_cylindrical.loc[event_cylindrical.colors=='g'].z.min())+1
layer_2_max = np.ceil(event_cylindrical.loc[event_cylindrical.colors=='g'].z.max())-1
layer_3_min = np.ceil(event_cylindrical.loc[event_cylindrical.colors=='c'].z.min())+1
layer_3_max = np.ceil(event_cylindrical.loc[event_cylindrical.colors=='c'].z.max())-1
layer_12_min = np.ceil(event_cylindrical.loc[event_cylindrical.colors=='m'].z.min())+1
layer_12_max = np.ceil(event_cylindrical.loc[event_cylindrical.colors=='m'].z.max())-1

feature_vector_r = voxalize_by_layer(event_cylindrical,
                                     layer='r',
                                     segments = [np.linspace(0, 350, 11),
                                                 np.linspace(-3.15, 3.15, 11),
                                                 np.linspace(layer_0_min, layer_0_max, 1)])

feature_vector_b = voxalize_by_layer(event_cylindrical,
                                     layer='b',
                                     segments = [np.linspace(0, 350, 11),
                                                 np.linspace(-3.15, 3.15, 11),
                                                 np.linspace(layer_1_min, layer_1_max, 1)])

feature_vector_g = voxalize_by_layer(event_cylindrical,
                                     layer='g',
                                     segments = [np.linspace(0, 350, 11),
                                                 np.linspace(-3.15, 3.15, 11),
                                                 np.linspace(layer_2_min, layer_2_max, 1)])

feature_vector_c = voxalize_by_layer(event_cylindrical,
                                     layer='c',
                                     segments = [np.linspace(0, 350, 11),
                                                 np.linspace(-3.15, 3.15, 11),
                                                 np.linspace(layer_2_min, layer_2_max, 1)])

feature_vector_m = voxalize_by_layer(event_cylindrical,
                                     layer='m',
                                     segments = [np.linspace(0, 350, 11),
                                                 np.linspace(-3.15, 3.15, 11),
                                                 np.linspace(layer_2_min, layer_2_max, 1)])

feature_vector = np.concatenate([feature_vector_r,
                                 feature_vector_b,
                                 feature_vector_g,
                                 feature_vector_c,
                                 feature_vector_m], axis=2)

ax = fig.add_subplot(122, projection='3d')
ax.voxels(feature_vector, edgecolor='k')
plt.show()
