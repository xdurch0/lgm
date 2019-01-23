import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from mnist import MNISTDataset
from utils import repeated_gibbs, gibbs_update_brbm, energy_rbm


# data
mnistd = MNISTDataset("data", 128, binarize=True)
train_steps = 1500
marginals = np.mean(mnistd.train_data, axis=0)

# model
cd = True
pcd = False
n_h = 1000
w_vh = tf.get_variable("w_vh", [784, n_h], tf.float32)
b_v = tf.get_variable("b_v", [784], tf.float32,
                      initializer=tf.zeros_initializer)
b_h = tf.get_variable("b_h", [n_h], tf.float32,
                      initializer=tf.zeros_initializer)


v_data = tf.placeholder(tf.float32, [None, 784])
start_sampler = tf.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
marginal_sampler = tf.distributions.Bernoulli(probs=marginals, dtype=tf.float32)
#v_random = start_sampler.sample(tf.shape(v_data))
v_random = marginal_sampler.sample(tf.shape(v_data)[0])
# this is just a dummy
h_random = start_sampler.sample([tf.shape(v_data)[0], n_h])

#_, h_data = gibbs_update_brbm((v_data, h_random), w_vh, b_v, b_h)
h_data = tf.nn.sigmoid(tf.matmul(v_data, w_vh) + b_h)
h_data = tf.stop_gradient(h_data)
if cd:
    v_sampled, h_sampled = repeated_gibbs(
        (v_data, h_data), 20, gibbs_update_brbm, w_vh=w_vh, b_v=b_v, b_h=b_h)
elif pcd:
    #v_store = tf.Variable(initial_value=start_sampler.sample((128, 784)), trainable=False)
    v_store = tf.Variable(initial_value=marginal_sampler.sample([128]),
                          trainable=False)
    h_store = tf.Variable(initial_value=start_sampler.sample((128, n_h)),
                          trainable=False)
    v_sampled, h_sampled = repeated_gibbs(
        (v_store, h_store), 20, gibbs_update_brbm, w_vh=w_vh, b_v=b_v, b_h=b_h)
    v_update = tf.assign(v_store, v_sampled)
    h_update = tf.assign(h_store, h_sampled)
else:
    v_sampled, h_sampled = repeated_gibbs(
        (v_random, h_random), 200, gibbs_update_brbm,
        w_vh=w_vh, b_v=b_v, b_h=b_h)
v_sampled, h_sampled = tf.stop_gradient(v_sampled), tf.stop_gradient(h_sampled)

logits_pos = tf.reduce_mean(-energy_rbm(v_data, h_data, w_vh, b_v, b_h))
logits_neg = tf.reduce_mean(-energy_rbm(v_sampled, h_sampled, w_vh, b_v, b_h))

loss = -(logits_pos - logits_neg)
opt = tf.train.GradientDescentOptimizer(0.1)
grads_vars = opt.compute_gradients(loss)
grad_update = opt.apply_gradients(grads_vars)
grads, _ = zip(*grads_vars)
grad_norm = tf.global_norm(grads)

if cd or pcd:
    from_random, _ = repeated_gibbs(
        (v_random, h_random), 200, gibbs_update_brbm,
        w_vh=w_vh, b_v=b_v, b_h=b_h)
else:
    from_random = 0  # meh


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if pcd:
        print("mean at first", sess.run(v_store).mean())
        print("doublecheck", sess.run(v_store).mean())
    for step in range(train_steps):
        img_batch, _ = mnistd.next_batch()

        to_run = [grad_update, grad_norm, loss]
        if pcd:
            to_run += [v_update, h_update]
        results = sess.run(to_run, feed_dict={v_data: img_batch})
        if not step % 50:
            print("Step", step)
            print("Gradient norm:", results[1])
            print("Loss:", results[2])
    if pcd:
        print("mean now", sess.run(v_store).mean())
        te = sess.run(v_store)
        for thing in te:
            plt.imshow(thing.reshape((28, 28)), cmap="Greys_r")
            plt.show()
    weights = sess.run(w_vh)
    print("weights...")
    #for img in weights.T:
    #    absmax = np.max(np.abs(img))
    #    plt.imshow(img.reshape((28,28)), vmin=-absmax, vmax=absmax, cmap="RdBu_r")
    #    plt.show()
    print("samples")
    while True:
        img_sample = sess.run(from_random if cd or pcd else v_sampled,
                              feed_dict={v_data: np.zeros((1, 784))})
        plt.imshow(img_sample.reshape((28, 28)), cmap="Greys_r")
        plt.show()
