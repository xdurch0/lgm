import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from mnist import MNISTDataset
from utils import repeated_gibbs, gibbs_update_brbm, energy_rbm


tf.enable_eager_execution()


# data
mnistd = MNISTDataset("data", 128, binarize=True)
train_steps = 1500
lr = 0.1
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


start_sampler = tf.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
marginal_sampler = tf.distributions.Bernoulli(probs=marginals, dtype=tf.float32)


for step in range(train_steps):
    img_batch, _ = mnistd.next_batch()
    v_data = img_batch
    v_random = marginal_sampler.sample(img_batch.shape[0])
    # this is just a dummy
    h_random = start_sampler.sample([img_batch.shape[0], n_h])
    #_, h_data = gibbs_update_brbm((v_data, h_random), w_vh, b_v, b_h)
    h_data = tf.nn.sigmoid(tf.matmul(v_data, w_vh) + b_h)

    if cd:
        v_sampled, h_sampled = repeated_gibbs(
            (v_data, h_data), 20, gibbs_update_brbm,
            w_vh=w_vh, b_v=b_v, b_h=b_h)
    else:
        v_sampled, h_sampled = repeated_gibbs(
            (v_random, h_random), 200, gibbs_update_brbm,
            w_vh=w_vh, b_v=b_v, b_h=b_h)

    with tf.GradientTape() as tape:
        logits_pos = tf.reduce_mean(-energy_rbm(v_data, h_data, w_vh, b_v, b_h))
        logits_neg = tf.reduce_mean(
            -energy_rbm(v_sampled, h_sampled, w_vh, b_v, b_h))
        loss = -(logits_pos - logits_neg)
    dW, dbv, dbh = tape.gradient(loss, [w_vh, b_v, b_h])
    w_vh.assign_sub(lr*dW)
    b_v.assign_sub(lr*dbv)
    b_h.assign_sub(lr*dbh)
    if not step % 50:
        print("Step", step)
        print("Loss:", loss)


print("weights...")
#for img in weights.T:
#    absmax = np.max(np.abs(img))
#    plt.imshow(img.reshape((28,28)), vmin=-absmax, vmax=absmax, cmap="RdBu_r")
#    plt.show()
print("samples")
while True:
    v_random = marginal_sampler.sample([1])
    h_random = start_sampler.sample([1, n_h])
    img_sample, _ = repeated_gibbs(
            (v_random, h_random), 200, gibbs_update_brbm,
            w_vh=w_vh, b_v=b_v, b_h=b_h)
    plt.imshow(img_sample.numpy().reshape((28, 28)), cmap="Greys_r")
    plt.show()
