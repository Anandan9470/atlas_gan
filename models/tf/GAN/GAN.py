import numpy as np
import tensorflow as tf
import time

mb_size = 100
X_dim = 1300
z_dim = 400
ITER = 30000

path = '/media/anandan/3474068674064B56/CERN/Program/cern_gan/'

def get_data():
    i = np.random.randint(100)
    batch1 = np.loadtxt(path+"data/vectorized_positive/batch_%d.txt" %i, delimiter=',')
    batch1 = batch1[np.random.choice(batch1.shape[0], 50, replace=False)]
    i =	np.random.randint(100)
    batch2 = np.loadtxt(path+"data/vectorized_positive/batch_%d.txt" %i, delimiter=',')
    batch2 = batch2[np.random.choice(batch2.shape[0], 50, replace=False)]

    batch = np.vstack([batch1, batch2])

    batch_sign = np.sign(batch)
    batch_log = np.log(batch+10e-20)*batch_sign
    batch_log_tanh = np.tanh(0.1*batch_log)

    return batch_log_tanh

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

with tf.name_scope('image'):
    X = tf.placeholder(tf.float32, shape=[None, X_dim])
with tf.name_scope('latent_space'):
    z = tf.placeholder(tf.float32, shape=[None, z_dim])

with tf.name_scope('discriminator_weights'):

    D_W1 = tf.Variable(xavier_init([X_dim, 128]), name='D_W1')
    D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

    D_W2 = tf.Variable(xavier_init([128, 128]), name='D_W2')
    D_b2 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

    D_W3 = tf.Variable(xavier_init([128, 1]), name='D_W3')
    D_b3 = tf.Variable(tf.zeros(shape=[1]), name='D_b3')

    tf.summary.histogram('D_W1', D_W1)
    tf.summary.histogram('D_b1', D_b1)
    tf.summary.histogram('D_W2', D_W2)
    tf.summary.histogram('D_b2', D_b2)
    tf.summary.histogram('D_W3', D_W3)
    tf.summary.histogram('D_b3', D_b3)

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

with tf.name_scope('generator_weigths'):

    G_W1 = tf.Variable(xavier_init([z_dim, 128]), name='G_W1')
    G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

    G_W2 = tf.Variable(xavier_init([128, X_dim]), name='G_w2')
    G_b2 = tf.Variable(tf.zeros(shape=[X_dim]), name='G_b2')

    tf.summary.histogram('G_W1', G_W1)
    tf.summary.histogram('G_b1', G_b1)
    tf.summary.histogram('G_W2', G_W2)
    tf.summary.histogram('G_b2', G_b2)

    theta_G = [G_W1, G_W2, G_b1, G_b2]

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(z):

    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    act = tf.matmul(G_h1, G_W2) + G_b2
    out = tf.nn.sigmoid(act)
    return out


def discriminator(x):

    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    act = tf.matmul(D_h2, D_W3) + D_b3
    out = tf.nn.sigmoid(act)
    return out

with tf.name_scope('generator'):
    G_sample = generator(z)
with tf.name_scope('discriminator_real'):
    D_real = discriminator(X)
with tf.name_scope('discriminator_fake'):
    D_fake = discriminator(G_sample)

l1_regularizer_D = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
D_penalty = tf.contrib.layers.apply_regularization(l1_regularizer_D, theta_D)
l1_regularizer_G = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
G_penalty = tf.contrib.layers.apply_regularization(l1_regularizer_G, theta_G)

with tf.name_scope('D_loss'):
    D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake)) + D_penalty
    tf.summary.scalar('D_loss', D_loss)

with tf.name_scope('G_loss'):
    G_loss = -tf.reduce_mean(tf.log(D_fake)) + G_penalty
    tf.summary.scalar('G_loss', G_loss)

with tf.name_scope('D_optimizer'):
    D_solver = (tf.train.AdamOptimizer(learning_rate=1e-4).minimize(D_loss, var_list=theta_D))
with tf.name_scope('G_optimizer'):
    G_solver = (tf.train.AdamOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./tensorboard_files/vGAN")
writer.add_graph(sess.graph)

saver = tf.train.Saver(save_relative_paths=True)

s_time = time.time()

for it in range(ITER):

    X_mb = get_data()

    _, D_loss_curr = sess.run([D_solver, D_loss],
                              feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)})

    _, G_loss_curr = sess.run([G_solver, G_loss],
                              feed_dict={z: sample_z(mb_size, z_dim)})

    if it % 10 == 0:

        #s = sess.run(merged_summary, feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)})
        #writer.add_summary(s, it)

        e_time = time.time()
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; Time: {:.3}'
              .format(it, D_loss_curr, G_loss_curr, e_time-s_time))
        s_time = e_time
        #samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

save_path = saver.save(sess, "./saved_weights/vGAN/model.ckpt")
print("Model saved in path: %s" % save_path)

