import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from matplotlib.colors import from_levels_and_colors
import time

mb_size = 128
X_dim = 230
z_dim = 20
lam = 10
n_disc = 5
lr = 1e-4

path = "/media/anandan/3474068674064B56/CERN/Program/atlas_sim_gan/"

data = np.loadtxt(path+"data/vectorized_cylindrical_230dim.csv", delimiter=',')
data[data<10] = 0
data = data/data.max()

E_l0 = data[:,:10].sum(axis=1)
E_l1 = data[:,10:110].sum(axis=1)
E_l2 = data[:,110:210].sum(axis=1)
E_l3 = data[:,210:220].sum(axis=1)
E_l12 = data[:,220:230].sum(axis=1)
E_tot = data.sum(axis=1)

def get_data():

    while(True):

        batch = data[np.random.choice(data.shape[0], mb_size, replace=False)]
        yield batch

data_gen = get_data()

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([X_dim, 230]))
D_b1 = tf.Variable(tf.zeros(shape=[230]))

D_W2 = tf.Variable(xavier_init([230, 230]))
D_b2 = tf.Variable(tf.zeros(shape=[230]))

D_W3 = tf.Variable(xavier_init([230, 230]))
D_b3 = tf.Variable(tf.zeros(shape=[230]))

D_W4 = tf.Variable(xavier_init([230, 1]))
D_b4 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4]

z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, 50]))
G_b1 = tf.Variable(tf.zeros(shape=[50]))

G_W2 = tf.Variable(xavier_init([50, 100]))
G_b2 = tf.Variable(tf.zeros(shape=[100]))

G_W3 = tf.Variable(xavier_init([100, 200]))
G_b3 = tf.Variable(tf.zeros(shape=[200]))

G_W4 = tf.Variable(xavier_init([200, 230]))
G_b4 = tf.Variable(tf.zeros(shape=[230]))

theta_G = [G_W1, G_W2, G_W3, G_W4, G_b1, G_b2, G_b3, G_b4]

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def G(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_o = tf.matmul(G_h3, G_W4) + G_b4
    return G_o


def D(X):
    D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
    D_o = tf.matmul(D_h3, D_W4) + D_b4
    return D_o

G_sample = G(z)
D_real = D(X)
D_fake = D(G_sample)

eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*X + (1. - eps)*G_sample
grad = tf.gradients(D(X_inter), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def sample_images(samples, epoch):

#    gen_imgs = samples[np.random.choice(samples.shape[0], 4, replace=False)]
#    gen_imgs = np.log(gen_imgs+10e-5)
#
#    r, c = 2, 2
#    fig, axs = plt.subplots(r, c)
#    cnt = 0
#    for i in range(r):
#        for j in range(c):
#
#            img = np.reshape(gen_imgs[cnt], newshape=(10,23), order='F')
#
#            num_levels = 20
#            vmin, vmax = img.min(), img.max()
#            midpoint = 0
#            levels = np.linspace(vmin, vmax, num_levels)
#            midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
#            vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
#            colors = plt.cm.seismic(vals)
#            cmap, norm = from_levels_and_colors(levels, colors)
#
#            im = axs[i,j].imshow(img, cmap=cmap, norm=norm, interpolation='none')
#            fig.colorbar(im, ax=axs[i,j])
#            axs[i,j].axis('off')
#            cnt += 1
#
#    fig.savefig("out/sample_%d.png" % epoch)
#    plt.close()

    l0 = samples[:,:10].sum(axis=1)
    l1 = samples[:,10:110].sum(axis=1)
    l2 = samples[:,110:210].sum(axis=1)
    l3 = samples[:,210:220].sum(axis=1)
    l12 = samples[:,220:230].sum(axis=1)
    ltot = samples.sum(axis=1)

    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax1.hist(E_l0, bins=50, histtype=u'step',label='Truth')
    ax1.hist(l0, bins=50, histtype=u'step',label='Processed')
    ax1.set_title('Layer 0')
    ax1.legend()

    ax2 = fig.add_subplot(232)
    ax2.hist(E_l1, bins=50, histtype=u'step',label='Truth')
    ax2.hist(l1, bins=50, histtype=u'step',label='Processed')
    ax2.set_title('Layer 1')
    ax2.legend()

    ax3 = fig.add_subplot(233)
    ax3.hist(E_l2, bins=50, histtype=u'step',label='Truth')
    ax3.hist(l2, bins=50, histtype=u'step',label='Processed')
    ax3.set_title('Layer 2')
    ax3.legend()

    ax4 = fig.add_subplot(234)
    ax4.hist(E_l3, bins=50, histtype=u'step',label='Truth')
    ax4.hist(l3, bins=50, histtype=u'step',label='Processed')
    ax4.set_title('Layer 3')
    ax4.legend()

    ax5 = fig.add_subplot(235)
    ax5.hist(E_l12, bins=50, histtype=u'step',label='Truth')
    ax5.hist(l12, bins=50, histtype=u'step',label='Processed')
    ax5.set_title('Layer 12')
    ax5.legend()

    ax6 = fig.add_subplot(236)
    ax6.hist(E_tot, bins=50, histtype=u'step',label='Truth')
    ax6.hist(ltot, bins=50, histtype=u'step',label='Processed')
    ax6.set_title('All layers')
    ax6.legend()
    fig.savefig("out/sample_hist_%d.png" % epoch)
    plt.close()

    if epoch == 100000-1:
        np.savetxt('out/sample.csv', samples, delimiter=',')

if not os.path.exists('out/'):
    os.makedirs('out/')

s_time = time.time()
sample_intervel = 100

for it in range(100000):
    for _ in range(n_disc):
        X_mb = data_gen.__next__()

        _, D_loss_curr = sess.run([D_solver, D_loss],
                                  feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)})

    _, G_loss_curr = sess.run([G_solver, G_loss],
                              feed_dict={z: sample_z(mb_size, z_dim)})

    if it == 2000:
        sample_intervel = 2000

    if it % sample_intervel == sample_intervel-1:
        e_time = time.time()
        time_diff = e_time - s_time
        s_time = e_time
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; Time: {}'
              .format(it, D_loss_curr, G_loss_curr, time_diff))
        samples = sess.run(G_sample, feed_dict={z: sample_z(10000, z_dim)})
        sample_images(samples, it)

