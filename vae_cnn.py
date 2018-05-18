import tensorflow as tf
import numpy as np

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')

def get_bias(shape, name):
    return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer)

def get_weights(shape, name, mask_mode='noblind', mask=None):
    weights_initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable(name, shape, tf.float32, weights_initializer)
    return W

class ConvolutionalEncoder(object):
    def __init__(self, X, conf):
        '''
            This is the 6-layer architecture for Convolutional Autoencoder
            mentioned in the original paper:
            Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction
            Note that only the encoder part is implemented as PixelCNN is taken
            as the decoder.
        '''
        W_conv1 = get_weights([5, 5, conf["n_input"][2], 500], "W_conv1")
        b_conv1 = get_bias([500], "b_conv1")
        conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
        pool1 = max_pool_2x2(conv1)

        W_conv2 = get_weights([5, 5, 500, 550], "W_conv2")
        b_conv2 = get_bias([550], "b_conv2")
        conv2 = tf.nn.relu(conv2d(pool1, W_conv2) + b_conv2)
        pool2 = max_pool_2x2(conv2)

        W_conv3 = get_weights([3, 3, 550, 700], "W_conv3")
        b_conv3 = get_bias([700], "b_conv3")
        conv3 = tf.nn.relu(conv2d(pool2, W_conv3) + b_conv3)

        # at this point we have done 2 layes of 2x2 max pooling operations so the image size will be
        # w_t = w_init / 2 / 2
        # h_t = h_init / 2 / 2 
        final_w = conf["n_input"][0] / 2 / 2
        final_h = conf["n_input"][1] / 2 / 2

        conv3_reshape = tf.reshape(conv3, (-1, final_w*final_h*700))
        W_fc_sigma = get_weights([final_w*final_h*700, conf["n_z"]], "W_fc_sigma")
        b_fc_sigma = get_bias([conf["n_z"]], "b_fc_sigma")

        W_fc_mean = get_weights([final_w*final_h*700, conf["n_z"]], "W_fc_mean")
        b_fc_mean = get_bias([conf["n_z"]], "b_fc_mean")
        #self.pred = tf.nn.softmax(tf.add(tf.matmul(conv3_reshape, W_fc), b_fc))

        # output parametrization of latent gaussian 
        self.z_mean = tf.add(tf.matmul(conv3_reshape, W_fc_mean),b_fc_mean)
        self.z_log_sigma_sq = tf.minimum(100.0,tf.add(tf.matmul(conv3_reshape, W_fc_sigma), b_fc_sigma))

class DeconvolutionDecoder(object):
    def __init__(self, z, conf):
        '''
            This decoder takes as input the latent gaussian code and produces Bernoulli
            distributions of same size as the input image shape
        '''
        # this decoder will perform 2 layes of deconv where the image size doubles:
        # w_t = w_init / 2 / 2
        # h_t = h_init / 2 / 2
        """
        h = ((len(value[1]) - 1) * stride_h) + kernel_h - 2 * pad_h
        w = ((len(value[2]) - 1) * stride_w) + kernel_w - 2 * pad_w
        """ 
        final_w = conf["n_input"][0] / 2 / 2
        final_h = conf["n_input"][1] / 2 / 2
        channels = conf["n_input"][2]
        im_w_after1_deconv = ((final_w - 1) * 2 + 3)
        im_h_after1_deconv = ((final_h - 1) * 2 + 3)

        # calculation to ensure output dim matched input shape
        # THIS CALCULATION IS NOT RELIABLE
        final_kernel_w =  conf["n_input"][0] - (im_w_after1_deconv-1)*2 + 2
        final_kernel_h =  conf["n_input"][1] - (im_h_after1_deconv-1)*2 + 2
        #print batch_size

        # z had dimensions 1 x conf.n_z
        b_fc_DC = get_bias([conf["n_z"]], "b_fc_DC")
        W_fc_DC = get_weights([conf["n_z"], final_w*final_h*700], "W_fc_DC")
        conv3_reshape = tf.matmul(tf.add(z, b_fc_DC), W_fc_DC)
        conv3 = tf.reshape(conv3_reshape, (-1, final_w, final_h, 700))

        b_conv3 = get_bias([700], "b_conv3_DC")
        conv2 = tf.layers.conv2d_transpose(tf.nn.relu(conv3 + b_conv3), 550, [3, 3], 
                                           strides=(2, 2), padding='same',name="W_deconv3_DC", 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())

        b_conv2 = get_bias([550], "b_conv2_DC")
        conv1 = tf.layers.conv2d_transpose(tf.nn.relu(conv2 + b_conv2), channels, [final_kernel_w, final_kernel_h], 
                                   strides=(2, 2), name="W_deconv2_DC", padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.x_reconstr_mean =  tf.nn.sigmoid(conv1)


class VariationalCNNAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001, batch_size=100):
        self.conf = conf
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.channels = conf["n_input"][2]
        
        # tf Graph input
        input_shape = [None].extend(conf["n_input"])
        self.x = tf.placeholder(tf.float32, input_shape)
        
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)


    def _create_network(self):

        # creat the cnn encoder and decoder networks
        self.cnn_enc = ConvolutionalEncoder(self.x, conf)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = self.cnn_enc.z_mean, self.cnn_enc.z_log_sigma_sq

        # Draw one sample z from Gaussian distribution
        n_z = self.conf["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.cnn_dec = DeconvolutionDecoder(self.z, conf)
        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = self.cnn_dec.x_reconstr_mean
            
          
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-8 to avoid evaluation of log(0.0)
        flatten_shape = self.conf["n_input"][2] * self.conf["n_input"][1] * self.conf["n_input"][0]
        x_vectorized  = tf.reshape(self.x, [-1, flatten_shape], name='x-vectorized')
        x_reconstr_mean_vectorized = tf.reshape(self.x_reconstr_mean, [-1, flatten_shape], name='x_reconstr_mean_vectorized')

        #reconstr_loss = \
        #    -tf.reduce_sum(self.x * tf.log(1e-8 + self.x_reconstr_mean)
        #                   + (1-self.x) * tf.log(1e-8 + 1 - self.x_reconstr_mean),
        #                   None)

        reconstr_loss = \
            -tf.reduce_sum(x_vectorized * tf.log(1e-8 + x_reconstr_mean_vectorized)
                           + (1-x_vectorized) * tf.log(1e-8 + 1 - x_reconstr_mean_vectorized),
                           1)

        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X})
        return cost
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})


def vae_train(X_train, network_architecture, learning_rate=0.0005,
          batch_size=1000, training_epochs=10, display_step=1):

    n_samples = X_train.shape[0]
    print "n_samples ", n_samples

    vae = VariationalCNNAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
        	# assuming input in range from 0-1
            batch_xs = X_train[(i*batch_size):((i+1)*batch_size),:]

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), 
                  "cost=", "{:.9f}".format(avg_cost))
    return vae


# tf Graph input
import tensorflow as tf
import numpy as np

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


test = unpickle("/home/ec2-user/cifar-100-python/test")
train = unpickle("/home/ec2-user/cifar-100-python/train")

X_test = test["data"]
y_test = test["coarse_labels"]
names_test = test["filenames"]

X_train = train["data"]
y_train = train["coarse_labels"]
names_train = train["filenames"]

#tf.reset_default_graph()
#x = tf.placeholder(tf.float32, [None,32,32, 3])

conf = \
    dict(n_input=(32,32,3), # Input shape of image data 
         n_z=30)  # dimensionality of latent space

X_train_rshp = np.reshape(X_train, (X_train.shape[0],32,32,3),order='F') / 255.0


tf.reset_default_graph()
vae = vae_train(X_train_rshp, conf, 
                training_epochs=100,batch_size=1000,learning_rate=0.0001)
#cnn_enc = ConvolutionalEncoder(x, conf)
#cnn_dec = DeconvolutionDecoder(cnn_enc.z_mean, conf)
# test encoder network
# Initializing the tensor flow variables
#init = tf.global_variables_initializer()
# Launch the session
#sess = tf.InteractiveSession()
#sess.run(init)
#z = sess.run([cnn_enc.z_mean, cnn_dec.x_reconstr_mean], feed_dict={x: np.reshape(X_test[0:2],(2,32,32,3))})


import matplotlib.pyplot as plt

def show_image_and_reconstruct(vae, X, names, idx):
    fig=plt.figure(figsize=(8, 4))
    img = np.reshape(X[idx],(32,32,3),order='F')

    # somehow generateing a few samples here?
    x_recon = vae.reconstruct(np.reshape(X[idx]/255.0,(1,32,32,3),order='F'))
    print x_recon.shape
    x_recon_mean = np.sum(x_recon,axis=0) / x_recon.shape[0]
    x_recon_mean = x_recon_mean * 255
    x_recon = x_recon_mean.astype('uint8')

    fig.add_subplot(1, 2, 1)
    plt.imshow(x_recon)
    plt.title("VAE Reconstruction")
    fig.add_subplot(1, 2, 2)
    plt.imshow(img)
    plt.title(names[idx])

    plt.savefig("./test_imgs/"+names[idx]+".png")