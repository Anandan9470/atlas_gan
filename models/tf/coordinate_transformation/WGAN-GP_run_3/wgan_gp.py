import tensorflow as tf
import numpy as np
import time
from plotting_and_saving import Plot_and_save

mb_size = 128
X_dim = 230
z_dim = 20
lam = 10
n_disc = 5
lr = 1e-3
sample_intervel = 1000
ITER = 100000

#path = "/media/anandan/3474068674064B56/CERN/Program/atlas_sim_gan/"
path = "/afs/inf.ed.ac.uk/user/s17/s1749290/Code/"
data = np.loadtxt(path+"data/data_v2/65/coordinate_transformation/coordinate_transformation.csv", delimiter=',')
data[data<25] = 0
data_norm = data.max()
data = data/data_norm
save_obj = Plot_and_save()

def get_data():
    while(True):
        batch = data[np.random.choice(data.shape[0], mb_size, replace=False)]
        yield batch
data_gen = get_data()

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

with tf.name_scope('image'):
    X = tf.placeholder(tf.float32, shape=[None, X_dim])
with tf.name_scope('latent_space'):
    z = tf.placeholder(tf.float32, shape=[None, z_dim])

phase = tf.placeholder(tf.bool, name='phase')

with tf.name_scope('discriminator_weights'):

    D_W1 = tf.Variable(xavier_init([X_dim, 230]), name='D_W1')
    D_b1 = tf.Variable(tf.zeros(shape=[230]), name='D_b1')

    D_W2 = tf.Variable(xavier_init([230, 230]), name='D_W2')
    D_b2 = tf.Variable(tf.zeros(shape=[230]), name='D_b2')

    D_W3 = tf.Variable(xavier_init([230, 230]), name='D_W3')
    D_b3 = tf.Variable(tf.zeros(shape=[230]), name='D_b3')

    D_W4 = tf.Variable(xavier_init([230, 1]), name='D_W4')
    D_b4 = tf.Variable(tf.zeros(shape=[1]), name='D_b4')

    tf.summary.histogram('D_W1', D_W1)
    tf.summary.histogram('D_b1', D_b1)
    tf.summary.histogram('D_W2', D_W2)
    tf.summary.histogram('D_b2', D_b2)
    tf.summary.histogram('D_W3', D_W3)
    tf.summary.histogram('D_b3', D_b3)
    tf.summary.histogram('D_W4', D_W4)
    tf.summary.histogram('D_b4', D_b4)

    theta_D = [D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4]

with tf.name_scope('generator_weigths'):

    G_W1 = tf.Variable(xavier_init([z_dim, 50]), name='G_W1')
    G_b1 = tf.Variable(tf.zeros(shape=[50]), name='G_b1')

    G_W2 = tf.Variable(xavier_init([50, 100]), name='G_w2')
    G_b2 = tf.Variable(tf.zeros(shape=[100]), name='G_b2')

    G_W3 = tf.Variable(xavier_init([100, 200]), name='G_w3')
    G_b3 = tf.Variable(tf.zeros(shape=[200]), name='G_b3')

    G_W4 = tf.Variable(xavier_init([200, 230]), name='G_w4')
    G_b4 = tf.Variable(tf.zeros(shape=[230]), name='G_b4')

    tf.summary.histogram('G_W1', G_W1)
    tf.summary.histogram('G_b1', G_b1)
    tf.summary.histogram('G_W2', G_W2)
    tf.summary.histogram('G_b2', G_b2)
    tf.summary.histogram('G_W3', G_W3)
    tf.summary.histogram('G_b3', G_b3)
    tf.summary.histogram('G_W4', G_W4)
    tf.summary.histogram('G_b4', G_b4)

    theta_G = [G_W1, G_W2, G_W3, G_W4, G_b1, G_b2, G_b3, G_b4]

def sample_z(m, n):
    return np.random.normal(0, 1., size=[m, n])

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_bh1 = tf.contrib.layers.batch_norm(G_h1, center=True, scale=True, is_training=phase)
    
    G_h2 = tf.nn.relu(tf.matmul(G_bh1, G_W2) + G_b2)
    G_bh2 = tf.contrib.layers.batch_norm(G_h2, center=True, scale=True, is_training=phase)
    
    G_h3 = tf.nn.relu(tf.matmul(G_bh2, G_W3) + G_b3)
    G_bh3 = tf.contrib.layers.batch_norm(G_h3, center=True, scale=True, is_training=phase)
    
    G_o = tf.nn.sigmoid(tf.matmul(G_bh3, G_W4) + G_b4)
    return G_o


def discriminator(X):
    D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    D_bh1 = tf.contrib.layers.batch_norm(D_h1, center=True, scale=True, is_training=phase)
    
    D_h2 = tf.nn.relu(tf.matmul(D_bh1, D_W2) + D_b2)
    D_bh2 = tf.contrib.layers.batch_norm(D_h2, center=True, scale=True, is_training=phase)
    
    D_h3 = tf.nn.relu(tf.matmul(D_bh2, D_W3) + D_b3)
    D_bh3 = tf.contrib.layers.batch_norm(D_h3, center=True, scale=True, is_training=phase)
    
    D_o = tf.matmul(D_bh3, D_W4) + D_b4
    return D_o

with tf.name_scope('generator'):
    G_sample = generator(z)
with tf.name_scope('discriminator_real'):
    D_real = discriminator(X)
with tf.name_scope('discriminator_fake'):
    D_fake = discriminator(G_sample)

eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*X + (1. - eps)*G_sample
grad = tf.gradients(discriminator(X_inter), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

with tf.name_scope('D_loss'):
    D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
    tf.summary.scalar('D_loss', D_loss)

with tf.name_scope('G_loss'):
    G_loss = -tf.reduce_mean(D_fake)
    tf.summary.scalar('G_loss', G_loss)

with tf.name_scope('D_optimizer'):
    D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9)
                .minimize(D_loss, var_list=theta_D))
with tf.name_scope('G_optimizer'):
    G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9)
                .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("out/tensorboard_files")
writer.add_graph(sess.graph)
saver = tf.train.Saver(save_relative_paths=True)

#s_time = time.time()

D_loss_list, G_loss_list = [], []

for it in range(ITER):
    for _ in range(n_disc):
        X_mb = data_gen.__next__()

        _, D_loss_curr = sess.run([D_solver, D_loss],
                                  feed_dict={X: X_mb, z: sample_z(mb_size, z_dim), phase:True})

    _, G_loss_curr = sess.run([G_solver, G_loss],
                              feed_dict={z: sample_z(mb_size, z_dim), phase:True})


    samples = sess.run(G_sample, feed_dict={z: sample_z(10000, z_dim), phase:False})
    save_obj.sample_images(samples*data_norm, it, is_last_epoch=False)

    if it % sample_intervel == sample_intervel-1:
        
        D_loss_list.append(D_loss_curr)
        G_loss_list.append(G_loss_curr)
        
#        e_time = time.time()
#        time_diff = e_time - s_time
#        s_time = e_time
#        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; Time: {:.3}'
#              .format(it+1, D_loss_curr, G_loss_curr, time_diff))

    if it == ITER-1:
        #samples = sess.run(G_sample, feed_dict={z: sample_z(10000, z_dim)})
        #save_obj.sample_images(samples*data_norm, it, is_last_epoch=True)

        np.savetxt('out/losses/D_loss.csv', D_loss_list, delimiter=',')
        np.savetxt('out/losses/G_loss.csv', G_loss_list, delimiter=',')
        np.savetxt('out/samples.csv', samples, delimiter=',')

        save_path = saver.save(sess, "out/saved_weights/model.ckpt")
        print("Model saved in path: %s" %save_path)

