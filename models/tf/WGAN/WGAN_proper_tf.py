import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

mb_size = 100
X_dim = 1000
z_dim = 700
h_dim = 500

def get_data():

    #i = np.random.randint(80)
    #batch = np.loadtxt("./data/vectorized/batch_%d.txt" %i, delimiter=',')

    batch = np.random.uniform(-1., 1., size=[100, 1000])
    return batch

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

class clip_weigts(tf.keras.constraints.Constraint):

  def __init__(self, min_value=0.0, max_value=1.0):
    self.min_value = min_value
    self.max_value = max_value

  def __call__(self, w):
    return w.assign(tf.clip_by_value(w, -0.01, 0.01))

with tf.name_scope('images'):
    X = tf.placeholder(tf.float64, shape=[None, X_dim])
with tf.name_scope('latent_space'):
    z = tf.placeholder(tf.float64, shape=[None, z_dim])

def generator(z):

    with tf.variable_scope('generator_ff', reuse=tf.AUTO_REUSE):

        G_h1 = tf.layers.dense(z, 500, activation=tf.nn.relu, use_bias=True,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               bias_initializer=tf.zeros_initializer(),
                               kernel_constraint=None,
                               bias_constraint=None,
                               trainable=True,
                               name='layer1',
                               reuse=None)

        with tf.variable_scope('layer1', reuse=True):
            w1 = tf.get_variable('kernel', dtype=tf.float64)

        G_out = tf.layers.dense(G_h1, 1000, activation=None, use_bias=True,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer(),
                                kernel_constraint=None,
                                bias_constraint=None,
                                trainable=True,
                                name='layer2',
                                reuse=None)
        with tf.variable_scope('layer2', reuse=True):
            w2 = tf.get_variable('kernel', dtype=tf.float64)

        tf.summary.histogram("w1", w1)
        tf.summary.histogram("w2", w2)

    return G_out

def discriminator(x):

    with tf.variable_scope('discriminator_ff', reuse=tf.AUTO_REUSE):

        D_h1 = tf.layers.dense(x, 500, activation=tf.nn.relu, use_bias=True,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               bias_initializer=tf.zeros_initializer(),
                               kernel_constraint=None,
                               bias_constraint=None,
                               trainable=True,
                               name='layer1',
                               reuse=None)

        with tf.variable_scope('layer1', reuse=True):
            w1 = tf.get_variable('kernel', dtype=tf.float64)

        D_out = tf.layers.dense(D_h1, 1000, activation=None, use_bias=True,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer(),
                                kernel_constraint=None,
                                bias_constraint=None,
                                trainable=True,
                                name='layer2',
                                reuse=None)

        with tf.variable_scope('layer2', reuse=True):
            w2 = tf.get_variable('kernel', dtype=tf.float64)

        tf.summary.histogram("w1", w1)
        tf.summary.histogram("w2", w2)

        return D_out

with tf.name_scope('generator'):
    G_sample = generator(z)
with tf.name_scope('discriminator_real'):
    D_real = discriminator(X)
with tf.name_scope('discriminator_fake'):
    D_fake = discriminator(G_sample)

with tf.name_scope('D_loss'):
    D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
with tf.name_scope('G_loss'):
    G_loss = -tf.reduce_mean(D_fake)

with tf.name_scope('D_optimizer'):
    D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-D_loss)
with tf.name_scope('G_optimizer'):
    G_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./tensorboard_files/not_working")
writer.add_graph(sess.graph)

s_time = time.time()
for it in range(130):

    for _ in range(5):

        X_mb = get_data()

        _, D_loss_curr = sess.run([D_solver, D_loss],
                                   feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)})

    _, G_loss_curr = sess.run([G_solver, G_loss],
                              feed_dict={z: sample_z(mb_size, z_dim)})

    if it % 10 == 0:

        s = sess.run(merged_summary, feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)})
        writer.add_summary(s, it)

        e_time = time.time()
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; Time: {:.3}'
              .format(it, D_loss_curr, G_loss_curr, e_time-s_time))
        s_time = e_time