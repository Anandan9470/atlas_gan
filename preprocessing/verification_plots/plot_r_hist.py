#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:25:18 2018

@author: anandan
"""

import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path="/media/anandan/3474068674064B56/CERN/Program/atlas_sim_gan/"
filename = "65/NTUP_FCS.13289379._000001.pool.root.1" #65
#filename = "524/NTUP_FCS.13744326._000001.pool.root.1" #524

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

def filter_hits_by_angle(event_spherical, r_angles, alpha_angles, layer):

    r_lower, r_upper = r_angles[0], r_angles[1]
    alpha_lower, alpha_upper = alpha_angles[0], alpha_angles[1]

    r_bool_array = np.logical_and(event_spherical.r.values<r_upper, event_spherical.r>r_lower)
    alpha_bool_array = np.logical_and(event_spherical.alpha.values<alpha_upper, event_spherical.alpha>alpha_lower)
    event_bool_array = np.logical_and(r_bool_array, alpha_bool_array)

    event_spherical = event_spherical[event_bool_array]

    return event_spherical

def voxalize_by_layer(event_cylindrical, layer, segments):

    event_cylindrical_layer_wise = event_cylindrical.loc[event_cylindrical.colors==layer, :]

    if event_cylindrical_layer_wise.shape[0] == 0:
        return np.zeros(shape=((len(segments[0])-1) *
                               (len(segments[1])-1) *
                               (len(segments[2])) ))

    ref_cloud = PyntCloud(event_cylindrical_layer_wise)
    voxelgrid_id = ref_cloud.add_structure("voxelgrid", segments=segments)
    feature_vector = ref_cloud.structures[voxelgrid_id].query_voxels(event_cylindrical_layer_wise.loc[:,['r','alpha','z']].values,
                                                                     event_cylindrical_layer_wise.loc[:,'E'].values)

    return feature_vector.reshape((-1,))

r_l0 = []
r_l1 = []
r_l2 = []
r_l3 = []
r_l12 = []

for n in range(0,1):

    s = n*100
    e = s+100

    event_range = range(s,e)
    xyzE = get_events(event_range)

    print("Percentage complted: %10.2f" %(n))

    for i, event in enumerate(event_range):

        event_cartisian = xyzE[i]

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
        event_phi = np.arctan2(event_cartisian.loc[:,'y'], event_cartisian.loc[:,'x'])
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

#        r_lower, r_upper = 0, 350
#        alpha_lower, alpha_upper = -3.15, 3.15
#
#        event_cylindrical = filter_hits_by_angle(event_cylindrical,
#                                                 r_angles=[r_lower, r_upper],
#                                                 alpha_angles=[alpha_lower, alpha_upper],
#                                                 layer='r')

        r_l0.extend(event_cylindrical[event_cylindrical.colors=='r'].r.values.tolist())
        r_l1.extend(event_cylindrical[event_cylindrical.colors=='b'].r.values.tolist())
        r_l2.extend(event_cylindrical[event_cylindrical.colors=='g'].r.values.tolist())
        r_l3.extend(event_cylindrical[event_cylindrical.colors=='c'].r.values.tolist())
        r_l12.extend(event_cylindrical[event_cylindrical.colors=='m'].r.values.tolist())

r_l0 = np.array(r_l0)
r_l1 = np.array(r_l1)
r_l2 = np.array(r_l2)
r_l3 = np.array(r_l3)
r_l12 = np.array(r_l12)

fig = plt.figure()
ax1 = fig.add_subplot(231)
ax1.hist(r_l0, bins=1000, histtype=u'step', density=True, label='Processed')
#bins_l0 = np.interp(np.linspace(0, len(r_l0), 3+1),
#                    np.arange(len(r_l0)),
#                    np.sort(r_l0))
#for b in bins_l0:
#    plt.axvline(x=b, c='r')
ax1.set_title('Layer 0')
ax1.legend()

ax2 = fig.add_subplot(232)
ax2.hist(r_l1, bins=1000, histtype=u'step', density=True, label='Processed')
#bins_l1 = np.interp(np.linspace(0, len(r_l1), 10+1),
#                    np.arange(len(r_l1)),
#                    np.sort(r_l1))
#for b in bins_l1:
#    plt.axvline(x=b, c='r')
ax2.set_title('Layer 1')
ax2.legend()

ax3 = fig.add_subplot(233)
ax3.hist(r_l2, bins=1000, histtype=u'step', density=True, label='Processed')
#bins_l2 = np.interp(np.linspace(0, len(r_l2), 10+1),
#                    np.arange(len(r_l2)),
#                    np.sort(r_l2))
#for b in bins_l2:
#    plt.axvline(x=b, c='r')
ax3.set_title('Layer 2')
ax3.legend()

ax4 = fig.add_subplot(234)
ax4.hist(r_l3, bins=1000, histtype=u'step', density=True, label='Processed')
#bins_l3 = np.interp(np.linspace(0, len(r_l3), 3+1),
#                    np.arange(len(r_l3)),
#                    np.sort(r_l3))
#for b in bins_l3:
#    plt.axvline(x=b, c='r')
ax4.set_title('Layer 3')
ax4.legend()

ax5 = fig.add_subplot(235)
ax5.hist(r_l12, bins=1000, histtype=u'step', density=True, label='Processed')
#bins_l12 = np.interp(np.linspace(0, len(r_l12), 3+1),
#                     np.arange(len(r_l12)),
#                     np.sort(r_l12))
#for b in bins_l12:
#    plt.axvline(x=b, c='r')
ax5.set_title('Layer 12')
ax5.legend()
plt.show()

#binnings = np.vstack([bins_l0, bins_l1, bins_l2, bins_l3, bins_l12])
#np.savetxt('r_binnings.csv', binnings, delimiter=',')
