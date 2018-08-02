#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:52:53 2018

@author: anandan
"""

import numpy as np
import matplotlib.pyplot as plt

filename = "NTUP_FCS.13289379._000001.pool.root.1"
path="/media/anandan/3474068674064B56/CERN/Program/atlas_sim_gan/"

def get_energy(layer):

    with open(path +"data/layer_wise/"+filename+"_Layer_"+str(layer)+".csv", 'r') as f:

        E_l = []
        for i,event in enumerate(f):
            event = event.split(';')[:-1]

            s = 0
            for e in event:
                e = float(e.split(',')[-1])
                if e > 0:
                    s = s + e

            E_l.append(s)

            if i%1000 == 0:
                print("Layer:%d, Event:%d" %(layer,i))

    return np.array(E_l)


E_l0 = get_energy(layer=0)
E_l1 = get_energy(layer=1)
E_l2 = get_energy(layer=2)
E_l3 = get_energy(layer=3)
E_l12 = get_energy(layer=12)
E_tot = E_l0 + E_l1 + E_l2 + E_l3+ E_l12

#l0, l1, l2, l3, l12, ltot = [], [], [], [], [], []
#
#for i in range(20):
#
#    batch = np.loadtxt(path+"data/vectorized_cylindrical_230d/batch_%d.csv" %i, delimiter=',')
#
#    layer0_sum = batch[:,:10].sum(axis=1)
#    layer1_sum = batch[:,10:110].sum(axis=1)
#    layer2_sum = batch[:,110:210].sum(axis=1)
#    layer3_sum = batch[:,210:220].sum(axis=1)
#    layer12_sum = batch[:,220:230].sum(axis=1)
#    ltot_sum = batch.sum(axis=1)
#
#    l0.extend(layer0_sum.tolist())
#    l1.extend(layer1_sum.tolist())
#    l2.extend(layer2_sum.tolist())
#    l3.extend(layer3_sum.tolist())
#    l12.extend(layer12_sum.tolist())
#    ltot.extend(ltot_sum.tolist())

#E = np.loadtxt(path+"data/data_v2/baseline/baseline.csv", delimiter=',')
#E_l0 = E[:,:10].sum(axis=1)
#E_l1 = E[:,10:110].sum(axis=1)
#E_l2 = E[:,110:210].sum(axis=1)
#E_l3 = E[:,210:220].sum(axis=1)
#E_l12 = E[:,220:230].sum(axis=1)
#E_tot = E.sum(axis=1)

batch = np.loadtxt(path+"data/data_v2/baseline/baseline.csv", delimiter=',')
l0 = batch[:,:1000].sum(axis=1)
l1 = batch[:,1000:2000].sum(axis=1)
l2 = batch[:,2000:3000].sum(axis=1)
l3 = batch[:,3000:4000].sum(axis=1)
l12 = batch[:,4000:5000].sum(axis=1)
ltot = batch.sum(axis=1)

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(231)
ax1.hist(E_l0, bins=50, histtype=u'step', density=True, label='Truth')
ax1.hist(l0, bins=50, histtype=u'step', density=True, label='Voxels')
ax1.set_title('Layer 0')
ax1.legend()

ax2 = fig.add_subplot(232)
ax2.hist(E_l1, bins=50, histtype=u'step', density=True, label='Truth')
ax2.hist(l1, bins=50, histtype=u'step', density=True, label='Voxels')
ax2.set_title('Layer 1')
ax2.legend()

ax3 = fig.add_subplot(233)
ax3.hist(E_l2, bins=50, histtype=u'step', density=True, label='Truth')
ax3.hist(l2, bins=50, histtype=u'step', density=True, label='Voxels')
ax3.set_title('Layer 2')
ax3.legend()

ax4 = fig.add_subplot(234)
ax4.hist(E_l3, bins=50, histtype=u'step', density=True, label='Truth')
ax4.hist(l3, bins=50, histtype=u'step', density=True, label='Voxels')
ax4.set_title('Layer 3')
ax4.legend()

ax5 = fig.add_subplot(235)
ax5.hist(E_l12, bins=50, histtype=u'step', density=True, label='Truth')
ax5.hist(l12, bins=50, histtype=u'step', density=True, label='Voxels')
ax5.set_title('Layer 12')
ax5.legend()

ax6 = fig.add_subplot(236)
ax6.hist(E_tot, bins=100, histtype=u'step', density=True, label='Truth')
ax6.hist(ltot, bins=100, histtype=u'step', density=True, label='Voxels')
ax6.set_title('All layers')
ax6.legend()
plt.show()

