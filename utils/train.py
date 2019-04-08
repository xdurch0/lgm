import tensorflow as tf


def step_ae(batch, encoder, decoder, loss_fn, opt):
    """Basic AE training function.

    Parameters:
        batch: Batch of target data.
        encoder: Duh.
        decoder: Duh.
        loss_fn: Loss function to use on reconstructions.
        opt: Optimizer.

    Returns:
        Reconstruction loss.

    """
    with tf.GradientTape() as tape:
        recon = decoder(encoder(batch))
        recon_loss = loss_fn(batch, recon)
    grads = tape.gradient(recon_loss, encoder.trainable_variables + decoder.trainable_variables)
    opt.apply_gradients(zip(grads, encoder.trainable_variables + decoder.trainable_variables))
    return recon_loss


def step_vae(batch, encoder, decoder, rec_loss_fn, kl_loss_fn, opt,
             kl_coeff=1.):
    """VAE training function.

    Parameters:
        batch: Batch of target data.
        encoder: Duh.
        decoder: Duh.
        rec_loss_fn: Loss function to use on reconstructions.
        kl_loss_fn: Loss function for KL divergence.
        opt: Optimizer.
        kl_coeff: Coefficient for KL loss. Likely needs to be set lower!

    Returns:
        Reconstruction loss and code samples (for checking distribution).

    """
    with tf.GradientTape() as tape:
        code = encoder(batch)
        means, logvars = tf.split(code, 2, axis=1)
        code_samples = (tf.random.normal(tf.shape(means)) *
                        tf.sqrt(tf.exp(logvars))) + means

        recon = decoder(code_samples)
        recon_loss = rec_loss_fn(batch, recon)
        total_loss = recon_loss + kl_coeff * kl_loss_fn(means, logvars)
    grads = tape.gradient(total_loss,
                          encoder.trainable_variables + decoder.trainable_variables)
    opt.apply_gradients(zip(grads,
                            encoder.trainable_variables + decoder.trainable_variables))

    return recon_loss, code_samples


def step_wae(batch, encoder, decoder, rec_loss_fn, mmd_loss_fn, opt,
             mmd_coeff=1.):
    """Basic WAE training function.

    Parameters:
        batch: Batch of target data.
        encoder: Duh.
        decoder: Duh.
        rec_loss_fn: Loss function to use on reconstructions.
        mmd_loss_fn: Loss function for KL divergence.
        opt: Optimizer.
        mmd_coeff: Coefficient for KL loss. Likely needs to be set lower!

    Returns:
        Reconstruction loss and code samples (for checking distribution).

    """
    with tf.GradientTape() as tape:
        code = encoder(batch)
        means, logvars = tf.split(code, 2, axis=1)
        code_samples = (tf.random.normal(tf.shape(means)) *
                        tf.sqrt(tf.exp(logvars))) + means

        recon = decoder(code_samples)
        recon_loss = rec_loss_fn(batch, recon)
        reg_loss = mmd_loss_fn(tf.random.normal(tf.shape(code_samples)),
                               code_samples)
        total_loss = recon_loss + mmd_coeff * reg_loss(means, logvars)
    grads = tape.gradient(total_loss,
                          encoder.trainable_variables + decoder.trainable_variables)
    opt.apply_gradients(zip(grads,
                            encoder.trainable_variables + decoder.trainable_variables))

    return recon_loss, code_samples


def step_gan(batch, generator, discriminator, noise_fn, loss_fn, gen_opt,
             disc_opt, label_smoothing=1., g_loss_thresh=None):
    """Basic GAN training function.

    Parameters:
        batch: Batch of target data.
        generator: Duh.
        discriminator: Duh.
        noise_fn: Function that takes batch size as argument and returns a noise
                  batch.
        loss_fn: Which loss function to use. E.g. BinaryCrossentropy for the
                 "inverted" loss less prone to vanishing gradients, or MSQ for
                 LSGAN.
        gen_opt: Optimizer for generator.
        disc_opt: Optimizer for discriminator.
        label_smoothing: Value for one-sided label smoothing; this replaces all
                         1 labels.
        g_loss_thresh: If given, only train discriminator if generator loss is
                       lower than this threshold.

    Returns:
        Tuple of generator loss, discriminator loss.

    """
    batch_dim = tf.shape(batch)[0]

    # fresh generated batch for generator training
    with tf.GradientTape(watch_accessed_variables=False) as g_tape:
        for vari in generator.trainable_variables:
            g_tape.watch(vari)
        gen_only_batch = generator(noise_fn(2 * batch_dim))
        g_loss = loss_fn(label_smoothing * tf.ones([2 * batch_dim, 1]),
                         discriminator(gen_only_batch))
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

    if (g_loss_thresh is not None and
            g_loss < tf.convert_to_tensor(g_loss_thresh, dtype=tf.float32)):
        # prepare mixed batch for discriminator training
        # For batchnorm to work better, we feed only real images, then only
        # generated ones and then average the gradients
        gen_batch = generator(noise_fn(batch_dim))
        real_labels = label_smoothing * tf.ones([batch_dim, 1])
        gen_labels = tf.zeros([batch_dim, 1])
        with tf.GradientTape() as d_tape:
            d_loss_real = loss_fn(real_labels, discriminator(batch))
            d_loss_fake = loss_fn(gen_labels, discriminator(gen_batch))
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        disc_opt.apply_gradients(
            zip(d_grads, discriminator.trainable_variables))
    else:
        d_loss = tf.convert_to_tensor(0., dtype=tf.float32)

    return g_loss, d_loss


def step_wgan(batch, step, generator, discriminator, noise_fn, gen_opt,
              disc_opt, n_critic=5):
    """Wasserstein GAN training function.

    Parameters:
        batch: Batch of target data.
        step: Current training step.
        generator: Duh.
        discriminator: Duh.
        noise_fn: Function that takes batch size as argument and returns a noise
                  batch.
        gen_opt: Optimizer for generator.
        disc_opt: Optimizer for discriminator.
        n_critic: How many discriminator steps to take per generator step.

    Returns:
        Loss value.

    """
    # prepare mixed batch for discriminator training
    # For batchnorm to work better, we feed only real images, then only
    # generated ones and then average the gradients
    batch_dim = tf.shape(batch)[0]
    gen_batch = generator(noise_fn(batch_dim))
    with tf.GradientTape() as d_tape:
        d_loss_real = tf.reduce_mean(discriminator(batch))
        d_loss_fake = tf.reduce_mean(discriminator(gen_batch))
        d_loss = -d_loss_real + d_loss_fake
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    if (step + 1) % n_critic != 0:
        # fresh generated batch for generator training
        with tf.GradientTape(watch_accessed_variables=False) as g_tape:
            for vari in generator.trainable_variables:
                g_tape.watch(vari)
            gen_only_batch = generator(noise_fn(2 * batch_dim))
            g_loss = -tf.reduce_mean(discriminator(gen_only_batch))
        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
        gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

    return d_loss


def step_wgangp(batch, step, generator, discriminator, noise_fn, gen_opt,
                disc_opt, penalty=10., n_critic=5):
    """'Improved' Wasserstein GAN training function.

    Parameters:
        batch: Batch of target data.
        step: Current training step.
        generator: Duh.
        discriminator: Duh.
        noise_fn: Function that takes batch size as argument and returns a noise
                  batch.
        gen_opt: Optimizer for generator.
        disc_opt: Optimizer for discriminator.
        penalty: Coefficient for gradient penalty term.
        n_critic: How many discriminator steps to take per generator step.

    Returns:
        Loss value.

    """
    # prepare mixed batch for discriminator training
    # For batchnorm to work better, we feed only real images, then only
    # generated ones and then average the gradients
    batch_dim = tf.shape(batch)[0]
    gen_batch = generator(noise_fn(batch_dim))

    interps = tf.random.uniform([tf.shape(batch)[0], 1, 1, 1], minval=0,
                                maxval=1)
    interp_batch = interps * batch + (1 - interps) * gen_batch
    with tf.GradientTape() as d_tape:
        d_loss_real = tf.reduce_mean(discriminator(batch))
        d_loss_fake = tf.reduce_mean(discriminator(gen_batch))

        with tf.GradientTape(watch_accessed_variables=False) as p_tape:
            p_tape.watch(interp_batch)
            d_loss_interp = tf.reduce_mean(discriminator(interp_batch))
        interp_grads = tf.reshape(p_tape.gradient(d_loss_interp, interp_batch),
                                  [batch_dim, -1])
        diff1 = tf.math.squared_difference(tf.norm(interp_grads, axis=1), 1)

        d_loss = -d_loss_real + d_loss_fake + penalty * tf.reduce_mean(diff1)
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    if (step + 1) % n_critic != 0:
        # fresh generated batch for generator training
        with tf.GradientTape(watch_accessed_variables=False) as g_tape:
            for vari in generator.trainable_variables:
                g_tape.watch(vari)
            gen_only_batch = generator(noise_fn(2 * batch_dim))
            g_loss = -tf.reduce_mean(discriminator(gen_only_batch))
        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
        gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

    return d_loss


@tf.function
def graph_wrapper(train_fn, *args, **kwargs):
    return train_fn(*args, **kwargs)
