import tensorflow as tf
import numpy as np
import time
from plotting_and_saving import Plot_and_save

mb_size = 128
X_dim = 230
y_dim = 1
z_dim = 20
lam = 10
n_disc = 7
lr = 1e-4
sample_intervel = 5000
ITER = 500000

save_obj = Plot_and_save()

#path = "/media/anandan/3474068674064B56/CERN/Program/atlas_sim_gan/"
path = "/afs/inf.ed.ac.uk/user/s17/s1749290/Code/"

data65 = np.loadtxt(path+"data/data_v2/65/coordinate_transformation/coordinate_transformation.csv", delimiter=',')
data65 = data65/data65.max()
zeros_label = np.zeros((data65.shape[0], 1))
data65 = np.hstack([data65, zeros_label])

data524 = np.loadtxt(path+"data/data_v2/524/coordinate_transformation/coordinate_transformation.csv", delimiter=',')
data524 = data524/data524.max()
ones_label = np.ones((data524.shape[0], 1))
data524 = np.hstack([data524, ones_label])

def get_data():
    while(True):
        batch65 = data65[np.random.choice(data65.shape[0], int(mb_size/2), replace=False)]        
        batch524 = data524[np.random.choice(data524.shape[0], int(mb_size/2), replace=False)]
        
        batch = np.vstack([batch65, batch524])
        np.random.shuffle(batch)
        
        yield batch[:,:-1], batch[:,-1:]
data_gen = get_data()

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

with tf.name_scope('image'):
    X = tf.placeholder(tf.float32, shape=[None, X_dim])
with tf.name_scope('labels'):
    y = tf.placeholder(tf.float32, shape=[None, y_dim])
with tf.name_scope('latent_space'):
    z = tf.placeholder(tf.float32, shape=[None, z_dim])

with tf.name_scope('discriminator_weights'):

    D_W1 = tf.Variable(xavier_init([X_dim+1, 230]), name='D_W1')
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

    G_W1 = tf.Variable(xavier_init([z_dim+1, 50]), name='G_W1')
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

def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_o = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)
    return G_o


def discriminator(X, y):
    inputs = tf.concat(axis=1, values=[X, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
    D_o = tf.matmul(D_h3, D_W4) + D_b4
    return D_o

with tf.name_scope('generator'):
    G_sample = generator(z, y)
with tf.name_scope('discriminator_real'):
    D_real = discriminator(X, y)
with tf.name_scope('discriminator_fake'):
    D_fake = discriminator(G_sample, y)

eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*X + (1. - eps)*G_sample
grad = tf.gradients(discriminator(X_inter, y), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

with tf.name_scope('D_loss'):
    D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
    tf.summary.scalar('D_loss', D_loss)

with tf.name_scope('G_loss'):
    G_loss = -tf.reduce_mean(D_fake)
    tf.summary.scalar('G_loss', G_loss)

with tf.name_scope('D_optimizer'):
    D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
                .minimize(D_loss, var_list=theta_D))
with tf.name_scope('G_optimizer'):
    G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
                .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("out/tensorboard_files")
writer.add_graph(sess.graph)
saver = tf.train.Saver(save_relative_paths=True)

s_time = time.time()

D_loss_list, G_loss_list = [], []

for it in range(ITER):
    for _ in range(n_disc):
        X_mb, y_mb = data_gen.__next__()

        _, D_loss_curr = sess.run([D_solver, D_loss],
                                  feed_dict={X: X_mb, z: sample_z(mb_size, z_dim), y: y_mb})
    
    _, y_mb = data_gen.__next__()
    _, G_loss_curr = sess.run([G_solver, G_loss],
                              feed_dict={z: sample_z(mb_size, z_dim), y: y_mb})


    if it % sample_intervel == sample_intervel-1:
        e_time = time.time()
        time_diff = e_time - s_time
        s_time = e_time
#        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; Time: {}'
#              .format(it, D_loss_curr, G_loss_curr, time_diff))

        D_loss_list.append(D_loss_curr)
        G_loss_list.append(G_loss_curr)

        samples65 = sess.run(G_sample, feed_dict={z: sample_z(10000, z_dim), y:np.zeros((10000, 1))})
        samples524 = sess.run(G_sample, feed_dict={z: sample_z(10000, z_dim), y:np.ones((10000, 1))})
        save_obj.sample_images(samples65, it, '65', is_last_epoch=False)
        save_obj.sample_images(samples524, it, '524', is_last_epoch=False)

#    if it == ITER-1:
#        samples = sess.run(G_sample, feed_dict={z: sample_z(10000, z_dim), y:np.ones(10000, 1)})
#        save_obj.sample_images(samples, it, is_last_epoch=True)
#
#        np.savetxt('out/losses/D_loss.csv', D_loss_list, delimiter=',')
#        np.savetxt('out/losses/G_loss.csv', G_loss_list, delimiter=',')
#        np.savetxt('out/samples.csv', samples, delimiter=',')
#
#        save_path = saver.save(sess, "out/saved_weights/model.ckpt")
#        print("Model saved in path: %s" % save_path)

