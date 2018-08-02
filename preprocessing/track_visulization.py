import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

filename = "NTUP_FCS.13289379._000001.pool.root.1"
path="/media/anandan/3474068674064B56/CERN/Program/atlas_sim_gan/"

def get_one_event(layer=1, requireEnergy=True):

    x, y, z, E = [], [], [], []

    with open(path +"data/layer_wise/"+filename+"_Layer_"+str(layer)+".csv", 'r') as f:
        for i,event in enumerate(f):
#            if (i!= 0) and (i!=20):
#                continue
            event = event.split(';')

            for hit in event[:-1]:
                hit = hit.split(',')

                x.append(float(hit[0]))
                y.append(float(hit[1]))
                z.append(float(hit[2]))

                if requireEnergy:
                    E.append(float(hit[4]))

            if i == 0:
                break

    return np.array(x), np.array(y), np.array(z), np.array(E)

def get_all_events(layer=1, requireEnergy=True):

    x, y, z, E = [], [], [], []

    with open(path +"data/layer_wise/"+filename+"_Layer_"+str(layer)+".csv", 'r') as f:
        for i,event in enumerate(f):

            if event != '\n':

                event = event.split(';')

                hit = np.random.choice(event[:-1])
                hit = hit.split(',')

                x.append(float(hit[0]))
                y.append(float(hit[1]))
                z.append(float(hit[2]))

                if requireEnergy:
                    E.append(float(hit[4]))

#            if i == 1000:
#                break

    return x, y, z, E

#tx, ty, tz, tE = [], [], [], []
#
#x, y, z, E = get_one_event(layer=0)
#tx.extend(x)
#ty.extend(y)
#tz.extend(z)
#tE.extend(E)
#
#x, y, z, E = get_one_event(layer=1)
#tx.extend(x)
#ty.extend(y)
#tz.extend(z)
#tE.extend(E)
#
#x, y, z, E = get_one_event(layer=2)
#tx.extend(x)
#ty.extend(y)
#tz.extend(z)
#tE.extend(E)
#
#x, y, z, E = get_one_event(layer=3)
#tx.extend(x)
#ty.extend(y)
#tz.extend(z)
#tE.extend(E)
#
#x, y, z, E = get_one_event(layer=12)
#tx.extend(x)
#ty.extend(y)
#tz.extend(z)
#tE.extend(E)
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#ax.set_xlim([1350,1800])
#ax.set_ylim([-800,-400])
#ax.set_zlim([300,600])
#
#tE = np.array(tE)
#
#tE[tE<=0] = 10e-5
#tE = np.tanh(0.1*np.log(tE))
#
#ax.scatter(tx, ty, tz, s=10, c=tE, cmap='Reds', marker='.')
#
#eta = pd.read_csv(path+"data/truth_angles/"+filename+"_eta.csv", header=None, usecols=[0, 1, 2, 3])
#phi = pd.read_csv(path+"data/truth_angles/"+filename+"_phi.csv", header=None, usecols=[0, 1, 2, 3])
#r = pd.read_csv(path+"data/truth_angles/"+filename+"_r.csv", header=None, usecols=[0, 1, 2, 3])
#
#X = r*phi.apply(np.cos)
#Y = r*phi.apply(np.sin)
#Z = r*eta.apply(np.sinh)
#
#ax.plot(X.iloc[0,:], Y.iloc[0,:], Z.iloc[0,:], c='b', alpha=0.5)
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y, z, _ = get_one_event(layer=0)
ax.scatter(x, y, z, s=15, c='r', marker='.', label='Layer 0', alpha=0.5)

x, y, z, _ = get_one_event(layer=1)
ax.scatter(x, y, z, s=1, c='b', marker='.', label='Layer 1', alpha=0.5)

x, y, z, _ = get_one_event(layer=2)
ax.scatter(x, y, z, s=1, c='g', marker='.', label='Layer 2', alpha=0.5)

x, y, z, _ = get_one_event(layer=3)
ax.scatter(x, y, z, s=15, c='c', marker='.', label='Layer 3', alpha=0.5)

x, y, z, _ = get_one_event(layer=12)
ax.scatter(x, y, z, s=15, c='m', marker='.', label='Layer 12', alpha=0.5)

#eta = pd.read_csv(path+"data/truth_angles/"+filename+"_eta.csv", header=None, usecols=[0, 1, 2, 3, 12])
#phi = pd.read_csv(path+"data/truth_angles/"+filename+"_phi.csv", header=None, usecols=[0, 1, 2, 3, 12])
#r = pd.read_csv(path+"data/truth_angles/"+filename+"_r.csv", header=None, usecols=[0, 1, 2, 3, 12])
#
#X = r*phi.apply(np.cos)
#Y = r*phi.apply(np.sin)
#Z = r*eta.apply(np.sinh)
#
#ax.plot(X.iloc[0,:], Y.iloc[0,:], Z.iloc[0,:], c='k', alpha=0.5)
#ax.plot(X.iloc[20,:], Y.iloc[20,:], Z.iloc[20,:], c='k', alpha=0.5)
#ax.plot(X.iloc[2,:], Y.iloc[2,:], Z.iloc[2,:])

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')

#ax.set_xlim([1200,2000])
#ax.set_ylim([-800,-400])
#ax.set_zlim([300,600])

plt.legend()
plt.show()

