import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize

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
        W_conv1 = get_weights([8, 8, conf["n_input"][2], 500], "W_conv1")
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
        final_w = int(conf["n_input"][0] / 2 / 2)
        final_h = int(conf["n_input"][1] / 2 / 2)

        conv3_reshape = tf.reshape(conv3, (-1, final_w*final_h*700))
        W_fc_sigma = get_weights([final_w*final_h*700, conf["n_z"]], "W_fc_sigma")
        b_fc_sigma = get_bias([conf["n_z"]], "b_fc_sigma")

        W_fc_mean = get_weights([final_w*final_h*700, conf["n_z"]], "W_fc_mean")
        b_fc_mean = get_bias([conf["n_z"]], "b_fc_mean")
        #self.pred = tf.nn.softmax(tf.add(tf.matmul(conv3_reshape, W_fc), b_fc))

        # output parametrization of latent gaussian 
        self.z_mean = tf.add(tf.matmul(conv3_reshape, W_fc_mean),b_fc_mean, name='z_mean')
        self.z_log_sigma_sq = tf.minimum(100.0,tf.add(tf.matmul(conv3_reshape, W_fc_sigma), b_fc_sigma), name='z_log_sigma_sq')

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
        final_w = int(conf["n_input"][0] / 2 / 2)
        final_h = int(conf["n_input"][1] / 2 / 2)
        channels = int(conf["n_input"][2])
        im_w_after1_deconv = ((final_w - 1) * 2)
        im_h_after1_deconv = ((final_h - 1) * 2)

        # calculation to ensure output dim matched input shape
        # THIS CALCULATION IS NOT RELIABLE
        final_kernel_w =  conf["n_input"][0] - (im_w_after1_deconv-1)*2 + 2
        final_kernel_h =  conf["n_input"][1] - (im_h_after1_deconv-1)*2 + 2
        #print batch_size

        """
        TEST FOR more deconvs
        """
        im_w_after2_deconv = ((im_w_after1_deconv - 1) * 2)
        im_h_after2_deconv = ((im_h_after1_deconv - 1) * 2)
        
        final_kernel_w =  conf["n_input"][0] - (im_w_after2_deconv-1)*2 + 2
        final_kernel_h =  conf["n_input"][1] - (im_h_after2_deconv-1)*2 + 2


        # z had dimensions 1 x conf.n_z
        b_fc_DC = get_bias([conf["n_z"]], "b_fc_DC")
        W_fc_DC = get_weights([conf["n_z"], final_w*final_h*900], "W_fc_DC")
        conv3_reshape = tf.matmul(tf.add(z, b_fc_DC), W_fc_DC)
        conv3 = tf.reshape(conv3_reshape, (-1, final_w, final_h, 900))

        b_conv3 = get_bias([900], "b_conv3_DC")
        conv2 = tf.layers.conv2d_transpose(tf.nn.relu(conv3 + b_conv3), 500, [2, 2], 
                                           strides=(2, 2), padding='same',name="W_deconv3_DC", 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())

        """
        b_conv2 = get_bias([1050], "b_conv2_DC")
        conv1 = tf.layers.conv2d_transpose(tf.nn.relu(conv2 + b_conv2), channels, [final_kernel_w, final_kernel_h], 
                                   strides=(2, 2), name="W_deconv2_DC", padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.x_reconstr_mean =  tf.nn.sigmoid(conv1)
        """

        b_conv2 = get_bias([500], "b_conv2_DC")
        conv1 = tf.layers.conv2d_transpose(tf.nn.relu(conv2 + b_conv2), 500, [2, 2], 
                                   strides=(2, 2), name="W_deconv2_DC", padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        b_conv1 = get_bias([500], "b_conv1_DC")
        conv0 = tf.layers.conv2d_transpose(tf.nn.relu(conv1 + b_conv1), channels, [5, 5], 
                                   strides=(1, 1), name="W_deconv1_DC", padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.x_reconstr_mean =  tf.nn.sigmoid(conv0, name='x_reconstr_mean')


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
        self.x = tf.placeholder(tf.float32, input_shape, name='x')
        
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
            -tf.reduce_sum(x_vectorized * tf.log(1e-6 + x_reconstr_mean_vectorized)
                           + (1-x_vectorized) * tf.log(1e-6 + 1 - x_reconstr_mean_vectorized),
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
        # tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
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
    print("n_samples ", n_samples)

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
import glob

#def unpickle(file):
#    import cPickle
#    with open(file, 'rb') as fo:
#        dict = cPickle.load(fo)
#    return dict


#test = unpickle("/home/ubuntu/cifar-100-python/test")
#train = unpickle("/home/ubuntu/cifar-100-python/train")

#X_test = test["data"]
#y_test = test["coarse_labels"]
#names_test = test["filenames"]

#X_train = train["data"]
#y_train = train["coarse_labels"]
#names_train = train["filenames"]

from PIL import Image
import glob, os
import numpy as np

box_grasp_ids = [
    '1ba5b0f5-2200-4f09-8256-d4b505149a29',
    '1e36bfe9-7b87-48bf-996c-0cf0318d464a',
    '2a958c47-de4b-4964-91cb-6262a9b7bb5a',
    '2aab4df5-9dfd-4deb-a31a-5858a2e0bad6',
    '2b747d3c-d7ae-46c6-9747-013e0f755819',
    '2c2466d2-d41e-4a4a-aa6b-d5e7221a02b2',
    '3b4e9607-846b-416a-b074-83da59b3901a',
    '3d38ed28-0173-4cd2-82ce-169871dfcc33',
    '4afb5316-2d99-46f5-8967-1fb2523eacd5',
    '5ff543b3-4b78-407b-b13c-c31e6d9dc3cc',
    '6c245912-5335-4dac-8de4-9df22c434531',
    '6e93ad54-2e8f-4e7b-ab44-0f366900ee77',
    '6eb81efb-23fc-4aec-a3fd-2dbe8177e470',
    '7c8ef8a9-1386-4f52-9888-3b5403e6b7e4',
    '7df1d020-8ac7-4d6b-aecb-0a7a70818cc8',
    '7f015a0c-7228-407a-8194-b5d8e170ffb8',
    '8e522f8a-89ab-4111-a7a2-f36f39ccf9e5',
    '9cfbe2e5-c9d6-41f3-a905-9414474cd456',
    '11c66a51-62d9-493a-8ce5-7b0c2b50bcd6',
    '14b9160b-7171-41a4-96f1-3d7a5d44c833',
    '22c0552a-1faa-439f-bd88-1dc081f382e8',
    '30baa683-fe8d-422f-a9b6-722ecfb226d8',
    '51f7ea0a-5725-4a6c-b6c6-4432e4faa290',
    '68c80c97-43ed-482c-8931-c3ae4c25caa7',
    '85f4d5db-949c-42b2-bbce-e6ab1da1f57d',
    '92fcddd6-61d3-4c90-b8ca-21aa705e4c67',
    '151d7649-7e28-44cd-a021-d0b83ac2bdf9',
    '261e5ecb-2619-417c-b848-36ad74084662',
    '291a78d1-3af9-4cda-83bd-91872d5ebe70',
    '312abc9b-547e-4952-8bfe-8391d063762f',
    '350d5e41-6fef-4777-b8b8-419ab39e8f20',
    '9655bea6-d943-4f25-964d-7a592d684a93',
    '9706c5de-d5e6-4e74-8629-49fa8b6e5f0f',
    '17243ad4-7086-4a54-be16-0ae78125c753',
    '545702f2-2ce9-4de3-be2c-e56e3215cfe5',
    '1410947d-d46d-42c1-9661-eee46905dce1',
    '10031797-24c7-444b-a329-e967a633249b',
    'a2ad5bc1-fd15-48ff-a90b-025773d5a17b',
    'a8d654b5-34cd-4e52-95b7-68f037927535',
    'a91805d1-129c-4efa-ae57-5756c78b5149',
    'acdce5e6-5b03-4152-8130-983a28ca6828',
    'b1ff98ad-38cd-4fbe-813c-7dd05afd35c0',
    'c72b12d9-6986-44f4-ad1a-6944e2d01b64',
    'c901a61e-fa65-4f80-8fcc-4d9cd56b3715',
    'd3d0c882-1015-4d9b-beed-46756da8cd6d',
    'd5e5db25-0eda-464c-a9ec-00c5aaf99ba1',
    'ddb8609f-fea3-46a4-8ebc-3bcafdecf1fd',
    'deef5d25-4b77-44a1-a7c3-0cfebc4cc7ce',
    'dfc4a9cd-a1f0-4b2b-a03e-0ecc16e810b2',
    'e203a3cc-216c-42ea-aa30-a4cef0779b45',
    'e5933cd8-d897-4b2d-8956-b142c2b64a15',
    'e7553cd8-c9ef-4821-91df-218117c6f8b8',
    'f460b11d-4223-4fb6-9e5a-828cb613f7c0',
    'fecfa937-36e5-4471-a5f6-223d1686a4b6',
]

def load_box_rgb_images(grasp_dir, regex):
    png_paths = glob.glob(grasp_dir+"/**/"+regex)
    print("{} images found!".format(len(png_paths)))
    X = []
    for path in png_paths:
        if any(box_id in path for box_id in box_grasp_ids):
            img = Image.open(path)

            # test for colormapping depth map
            #depth_array = np.array(img, dtype=np.uint16) / 1000.0
            #cmap = cm.hot
            #norm = Normalize(vmin=1.0, vmax=1.7)
            #color_map_image = cmap(norm(depth_array))[:,:,0:3]
            img = img.crop((0,0,100,64))

            X.append(np.array(img))
    X = np.stack(X,axis=0)
    return X

def load_rgb_images(grasp_dir, regex):
    png_paths = glob.glob(grasp_dir+"/**/"+regex)
    print("{} images found!".format(len(png_paths)))
    X = []
    for path in png_paths:
        img = Image.open(path)

        # test for colormapping depth map
        #depth_array = np.array(img, dtype=np.uint16) / 1000.0
        #cmap = cm.hot
        #norm = Normalize(vmin=1.0, vmax=1.7)
        #color_map_image = cmap(norm(depth_array))[:,:,0:3]
        img = img.crop((0,0,100,64))

        X.append(np.array(img))
    X = np.stack(X,axis=0)
    return X

X_train_unpick = load_box_rgb_images("/data/help_grasps/", "*128_64_rgb.png")
X_train_unpick.shape

X_train_empty = load_rgb_images("/data/empty_bin_grasps/", "*128_64_rgb.png")
X_train = load_rgb_images("/data/normal_grasps/", "*128_64_rgb.png")

#X_train_unpick = load_box_rgb_images("/home/ubuntu/data/help_grasps/", "*128_64_depth.png")
#X_train_unpick.shape

#X_train_empty = load_rgb_images("/home/ubuntu/data/empty_bin_grasps/", "*128_64_depth.png")
#X_train = load_rgb_images("/home/ubuntu/data/normal_grasps/", "*128_64_depth.png")

#tf.reset_default_graph()
#x = tf.placeholder(tf.float32, [None,32,32, 3])

conf = \
    dict(n_input=(64,100,3), # Input shape of image data 
         n_z=20)  # dimensionality of latent space

#X_train_rshp = np.reshape(X_train, (X_train.shape[0],32,32,3),order='F') / 255.0


tf.reset_default_graph()
vae = vae_train(np.concatenate((X_train[::4],X_train_empty, X_train_unpick)) / 255.0, conf, 
                training_epochs=100, batch_size=100, learning_rate=0.000805)
#cnn_enc = ConvolutionalEncoder(x, conf)
#cnn_dec = DeconvolutionDecoder(cnn_enc.z_mean, conf)
# test encoder network
# Initializing the tensor flow variables
#init = tf.global_variables_initializer()
# Launch the session
#sess = tf.InteractiveSession()
#sess.run(init)
#z = sess.run([cnn_enc.z_mean, cnn_dec.x_reconstr_mean], feed_dict={x: np.reshape(X_test[0:2],(2,32,32,3))})


#import matplotlib.pyplot as plt

"""
def show_image_and_reconstruct(vae, X, names, idx):
    fig=plt.figure(figsize=(8, 4))
    img = np.reshape(X[idx],(32,32,3),order='F')

    # somehow generateing a few samples here?
    x_recon = vae.reconstruct(np.reshape(X[idx]/255.0,(1,32,32,3),order='F'))
    #print x_recon.shape
    x_recon_mean = np.sum(x_recon,axis=0) / x_recon.shape[0]
    x_recon_mean = x_recon_mean * 255
    #x_recon_mean = x_recon[0] * 255
    x_recon = x_recon_mean.astype('uint8')

    fig.add_subplot(1, 2, 1)
    plt.imshow(x_recon)
    plt.title("VAE Reconstruction")
    fig.add_subplot(1, 2, 2)
    plt.imshow(img)
    plt.title(names[idx])

    plt.savefig("./test_imgs/"+names[idx]+".png")
"""

import matplotlib.pyplot as plt
def show_image_and_reconstruct(vae, X, idx, name_prefix):
    fig=plt.figure(figsize=(8, 4))
    # somehow generateing a few samples here?
    x_recon = vae.reconstruct(np.reshape(X[idx]/255.0,(1,64,100,3)))
    z = vae.transform(np.reshape(X[idx]/255.0,(1,64,100,3)))
    #print x_recon.shape
    #x_recon_mean = np.sum(x_recon,axis=0) / x_recon.shape[0]
    #x_recon_mean = x_recon_mean * 255
    x_recon_mean = x_recon[0] * 255
    x_recon = x_recon_mean.astype('uint8')

    fig.add_subplot(1, 2, 1)
    plt.imshow(x_recon)
    plt.title("VAE Reconstruction")
    fig.add_subplot(1, 2, 2)
    plt.imshow(X[idx])
    img_name = name_prefix+"_"+str(idx)
    plt.title(img_name)
    plt.savefig("./test_imgs/"+img_name+".png",bbox_inches='tight', pad_inches=0)
    img_url = "http://ec2-18-219-138-68.us-east-2.compute.amazonaws.com:8000/"+img_name+".png"
    return z, img_url

Z = []
URLs = []
colors = []
for i in range(X_train_unpick.shape[0]):
    z, img_url = show_image_and_reconstruct(vae, X_train_unpick, i, "unpickable")
    Z.append(z)
    URLs.append(img_url)
    colors.append('red')

for i in range(X_train_empty.shape[0]):
    z, img_url = show_image_and_reconstruct(vae, X_train_empty, i, "empty_bin")
    Z.append(z)
    URLs.append(img_url)
    colors.append('blue')

for i in range(0,X_train.shape[0],50):
    z, img_url = show_image_and_reconstruct(vae, X_train, i, "normal")
    Z.append(z)
    URLs.append(img_url)
    colors.append('green')

Z = np.array(Z)

Z_unpick = Z[0:X_train_unpick.shape[0]]
Z_empty = Z[X_train_unpick.shape[0]:(X_train_unpick.shape[0]+X_train_empty.shape[0])]
Z_normal = Z[(X_train_unpick.shape[0]+X_train_empty.shape[0]):]

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool

source = ColumnDataSource(data=dict(
    x=Z[:,0,0],
    y=Z[:,0,1],
    imgs=URLs,
    color=colors
))

hover = HoverTool( tooltips="""
    <div>
        <div>
            <img
                src="@imgs" height="256" alt="@imgs" width="512"
                style="float: right; margin: 0px 0px 0px 0px; opacity:1.0;"
            ></img>
        </div>
    </div>
    """
)

output_file("./test_imgs/embedding.html")

p = figure(plot_width=800, plot_height=800, tools=[hover, 'pan', 'wheel_zoom'],
           title="CNN VAE Embedding")
# add a square renderer with a size, color, and alpha
p.circle('x', 'y', size=20, source=source, color='color', alpha=0.5)

# show the results
show(p)

# attempt to fit model to discern between unpick and normal + empty
import sklearn.neighbors
import sklearn.metrics
import sklearn.ensemble
import sklearn.linear_model
from sklearn.model_selection import train_test_split

clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
clf = sklearn.linear_model.LogisticRegression(class_weight={0.0: 1.5, 1.0:0.1},penalty='l2')
#clf = sklearn.ensemble.RandomForestClassifier(class_weight={0.0: 6.0, 1.0:0.01})
Z_train = np.vstack([Z_normal[:,0,:], Z_empty[:,0,:], Z_unpick[:,0,:]])
y_train = np.hstack([np.ones(Z_normal.shape[0]+Z_empty.shape[0]), np.zeros(Z_unpick.shape[0])])

Z_train = np.vstack([Z_normal[:,0,:], Z_unpick[:,0,:]])
y_train = np.hstack([np.ones(Z_normal.shape[0]), np.zeros(Z_unpick.shape[0])])

Z_train, Z_test, y_train, y_test = train_test_split(Z_train, y_train, test_size=0.4, random_state=42)
clf.fit(Z_train, y_train)
y_pred = clf.predict(Z_test)

fpr = np.where((y_pred == 1) & (y_pred != y_test))[0].shape[0]
fnr = np.where((y_pred == 0) & (y_pred != y_test))[0].shape[0]
print "FPR : ",fpr," / ",np.where((y_test == 0))[0].shape[0]," FNR : ",fnr," / ",np.where((y_test == 1))[0].shape[0]

Z_train = np.vstack([Z_normal[:,0,:], Z_empty[:,0,:], Z_unpick[:,0,:]])
y_train = np.hstack([np.ones(Z_normal.shape[0]+Z_empty.shape[0]), np.zeros(Z_unpick.shape[0])])

clf = sklearn.linear_model.LogisticRegression(class_weight={0.0: 1.5, 1.0:0.1},penalty='l1')
clf.fit(Z_train, y_train)

def adhoc_clf(img_path, clf):
    img = Image.open(img_path)
    img_resize = img.resize((128,64))
    img_resize = img_resize.crop((0,0,100,64))
    x = np.array(img_resize)
    #x_recon = vae.reconstruct(np.reshape(x/255.0,(1,64,100,3)))
    z = vae.transform(np.reshape(x/255.0,(1,64,100,3)))
    print z
    print img_path
    return clf.predict(z)

def do_eval():
    import time
    start = time.time()
    print ("\n Unpickable Grasps \n")
    print adhoc_clf("test_unpick_1.png",clf)
    print adhoc_clf("test_unpick_2.png",clf)
    print adhoc_clf("test_unpick_3.png",clf)
    print adhoc_clf("test_unpick_4.png",clf)
    print adhoc_clf("test_unpick_5.png",clf)
    print adhoc_clf("test_unpick_6.png",clf)
    print adhoc_clf("test_unpick_7.png",clf)
    print adhoc_clf("test_unpick_8.png",clf)
    print adhoc_clf("test_unpick_9.png",clf)
    print adhoc_clf("test_unpick_10.png",clf)
    print adhoc_clf("test_unpick_11.png",clf)
    print adhoc_clf("test_unpick_12.png",clf)
    print adhoc_clf("test_unpick_13.png",clf)
    print adhoc_clf("test_unpick_14.png",clf)
    print adhoc_clf("test_unpick_15.png",clf)
    print adhoc_clf("test_unpick_16.png",clf)
    print adhoc_clf("test_unpick_17.png",clf)
    print adhoc_clf("test_unpick_18.png",clf)
    print adhoc_clf("test_unpick_19.png",clf)
    print adhoc_clf("test_unpick_20.png",clf)
    print adhoc_clf("test_unpick_21.png",clf)
    print adhoc_clf("test_unpick_22.png",clf)
    print adhoc_clf("test_unpick_23.png",clf)
    print adhoc_clf("test_unpick_24.png",clf)

    print ("\n Normal Grasps \n")
    print adhoc_clf("test_normal_1.png",clf)
    print adhoc_clf("test_normal_2.png",clf)
    print adhoc_clf("test_normal_3.png",clf)
    print adhoc_clf("test_normal_4.png",clf)
    print adhoc_clf("test_normal_5.png",clf)
    print adhoc_clf("test_normal_6.png",clf)
    print adhoc_clf("test_normal_7.png",clf)
    print adhoc_clf("test_normal_8.png",clf)
    print adhoc_clf("test_normal_9.png",clf)
    print adhoc_clf("test_normal_11.png",clf)
    print adhoc_clf("test_normal_12.png",clf)
    print adhoc_clf("test_normal_13.png",clf)
    print adhoc_clf("test_normal_14.png",clf)
    print adhoc_clf("test_normal_15.png",clf)
    print adhoc_clf("test_normal_16.png",clf)
    print adhoc_clf("test_normal_17.png",clf)
    print adhoc_clf("test_normal_18.png",clf)
    print adhoc_clf("test_normal_19.png",clf)
    print adhoc_clf("test_normal_20.png",clf)
    print adhoc_clf("test_normal_21.png",clf)
    print adhoc_clf("test_normal_22.png",clf)
    print adhoc_clf("test_normal_23.png",clf)
    print adhoc_clf("test_normal_24.png",clf)
    print adhoc_clf("test_normal_25.png",clf)
    print adhoc_clf("test_normal_26.png",clf)
    print adhoc_clf("test_normal_27.png",clf)
    print adhoc_clf("test_normal_28.png",clf)
    print adhoc_clf("test_normal_29.png",clf)
    print adhoc_clf("test_normal_30.png",clf)
    print adhoc_clf("test_normal_31.png",clf)
    print adhoc_clf("test_normal_32.png",clf)
    print adhoc_clf("test_normal_33.png",clf)
    print adhoc_clf("test_normal_34.png",clf)
    print adhoc_clf("test_normal_35.png",clf)
    print adhoc_clf("test_normal_36.png",clf)
    print adhoc_clf("test_normal_37.png",clf)
    print adhoc_clf("test_normal_38.png",clf)
    print adhoc_clf("test_normal_39.png",clf)
    print adhoc_clf("test_normal_40.png",clf)
    print adhoc_clf("test_normal_41.png",clf)

end = time.time()
print(end - start)


import cloudpickle
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        cloudpickle.dump(obj, output)

tf.train.write_graph(vae.sess.graph_def,
                     "./",
                     'VCNNAE.pb',
                     as_text=False)


# Testing saving graph from tensorflow
from tensorflow.python.framework.graph_util import convert_variables_to_constants

# access the default graph
graph = tf.get_default_graph()

# retrieve the protobuf graph definition
input_graph_def = graph.as_graph_def()

output_node_names = "x,x_reconstr_mean,z_mean"

# TensorFlow built-in helper to export variables to constants
output_graph_def = convert_variables_to_constants(
    sess=vae.sess,
    input_graph_def=input_graph_def, # GraphDef object holding the network
    output_node_names=output_node_names.split(",") # List of name strings for the result nodes of the graph
) 

tf.train.write_graph(output_graph_def,
                     "./",
                     'VCNNAE.pb',
                     as_text=False)



from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=7, n_iter=27, random_state=42)
X_svd = np.concatenate((X_train[::4],X_train_empty, X_train_unpick)) / 255.0
X_svd = np.reshape(X_svd,(X_svd.shape[0],X_svd.shape[1]*X_svd.shape[2]*X_svd.shape[3]))
svd.fit(X_svd)

Z_svd_unpick = svd.transform(np.reshape(X_train_unpick,(X_train_unpick.shape[0],X_train_unpick.shape[1]*X_train_unpick.shape[2]*X_train_unpick.shape[3])))
Z_svd_empty = svd.transform(np.reshape(X_train_empty,(X_train_empty.shape[0],X_train_empty.shape[1]*X_train_empty.shape[2]*X_train_empty.shape[3])))
Z_svd_normal = svd.transform(np.reshape(X_train[::50],(X_train[::50].shape[0],X_train[::50].shape[1]*X_train[::50].shape[2]*X_train[::50].shape[3])))

Z_svd_train = np.vstack([Z_svd_normal, Z_svd_unpick])
y_svd_train = np.hstack([np.ones(Z_svd_normal.shape[0]), np.zeros(Z_svd_unpick.shape[0])])

clf = sklearn.linear_model.LogisticRegression(class_weight={0.0: 4.95, 1.0:0.1},penalty='l1')
Z_train, Z_test, y_train, y_test = train_test_split(Z_svd_train, y_svd_train, test_size=0.4, random_state=42)
clf.fit(Z_train, y_train)
y_pred = clf.predict(Z_test)

fpr = np.where((y_pred == 1) & (y_pred != y_test))[0].shape[0]
fnr = np.where((y_pred == 0) & (y_pred != y_test))[0].shape[0]
print "FPR : ",fpr," / ",np.where((y_test == 0))[0].shape[0]," FNR : ",fnr," / ",np.where((y_test == 1))[0].shape[0]



clf = sklearn.linear_model.LogisticRegression(class_weight={0.0: 1.2, 1.0:0.1},penalty='l1')
Z_train = np.vstack([Z_normal[:,0,:], Z_unpick[:,0,:]])
y_train = np.hstack([np.ones(Z_normal.shape[0]), np.zeros(Z_unpick.shape[0])])

Z_train, Z_test, y_train, y_test = train_test_split(Z_train, y_train, test_size=0.4, random_state=42)
clf.fit(Z_train, y_train)
y_pred = clf.predict(Z_test)

fpr = np.where((y_pred == 1) & (y_pred != y_test))[0].shape[0]
fnr = np.where((y_pred == 0) & (y_pred != y_test))[0].shape[0]
print "FPR : ",fpr," / ",np.where((y_test == 0))[0].shape[0]," FNR : ",fnr," / ",np.where((y_test == 1))[0].shape[0]


Z_train = np.vstack([Z_normal[:,0,:], Z_empty[:,0,:], Z_unpick[:,0,:]])
y_train = np.hstack([np.ones(Z_normal.shape[0]+Z_empty.shape[0]), np.zeros(Z_unpick.shape[0])])
clf.fit(Z_train, y_train)


import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize

graph_prefix = ''
graph_path = './VCNNAE.pb'

with tf.gfile.GFile(graph_path, "rb") as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name=graph_prefix)

if self.cpu_only:
config = tf.ConfigProto(
    device_count={"GPU": 0, "CPU": 1})
else:
    config = tf.ConfigProto()

self._graph = graph
session = tf.Session(
    graph=graph,
    config=config)

for op in graph.get_operations():
    print str(op.name) 

class TensorflowVariationalAutoencoder(object):
    def __init__(self,
                 graph_path=None,
                 graph_prefix='',
                 input_tensor_name='x',
                 output_reconstr_tensor_name='x_reconstr_mean',
                 output_latent_tensor_name='z_mean',
                 additional_tensors=None,
                 cpu_only=True,
                 batch_size=100):
        self.graph_path = graph_path
        self.graph_prefix = graph_prefix
        self.input_tensor_name = input_tensor_name
        self.output_reconstr_tensor_name = output_reconstr_tensor_name
        self.output_latent_tensor_name = output_latent_tensor_name
        self.additional_tensors = additional_tensors or {}
        self.cpu_only = cpu_only
        self.batch_size = batch_size

        if self.graph_path is not None:
            self.load_graph(graph_path=self.graph_path,
                            graph_prefix=self.graph_prefix,
                            input_tensor_name=self.input_tensor_name,
                            output_reconstr_tensor_name=self.output_reconstr_tensor_name,
                            output_latent_tensor_name=self.output_latent_tensor_name,
                            additional_tensors=self.additional_tensors,
                            cpu_only=self.cpu_only,
                            batch_size=self.batch_size)


    def load_graph(self, graph_path,
               graph_prefix='',
               input_tensor_name='x',
               output_reconstr_tensor_name='x_reconstr_mean',
               output_latent_tensor_name='z_mean',
               additional_tensors=None,
               cpu_only=True,
               batch_size=100):
        
        self.graph_path = graph_path
        self.graph_prefix = graph_prefix
        self.input_tensor_name = input_tensor_name
        self.output_reconstr_tensor_name = output_reconstr_tensor_name
        self.output_latent_tensor_name = output_latent_tensor_name
        self.additional_tensors = additional_tensors or {}
        self.cpu_only = cpu_only
        self.batch_size = batch_size

        with tf.gfile.GFile(graph_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name=self.graph_prefix)
        self._graph = graph

        if self.cpu_only:
            config = tf.ConfigProto(
                device_count={"GPU": 0, "CPU": 1})
        else:
            config = tf.ConfigProto()

        self._session = tf.Session(
            graph=graph,
            config=config)

        if self.graph_prefix:
            gp = self.graph_prefix + '/'
        else:
            gp = ''

        self._session = tf.Session(
            graph=graph,
            config=config)

        # establish input / output tensors of graph
        self.output_reconstr_tensor = self._graph.get_tensor_by_name(
            "{}:0".format(self.output_reconstr_tensor_name))
        self.output_latent_tensor = self._graph.get_tensor_by_name(
            "{}:0".format(self.output_latent_tensor_name))
        self.input_tensor = self._graph.get_tensor_by_name(
            "{}:0".format(self.input_tensor_name))


    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self._session.run(self.output_latent_tensor, feed_dict={self.input_tensor: X})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self._session.run(self.output_reconstr_tensor, 
                             feed_dict={self.input_tensor: X})



def adhoc_clf2(img_path, vae):
    img = Image.open(img_path)
    img_resize = img.resize((128,64))
    img_resize = img_resize.crop((0,0,100,64))
    x = np.array(img_resize)
    #x_recon = vae.reconstruct(np.reshape(x/255.0,(1,64,100,3)))
    z = vae.transform(np.reshape(x/255.0,(1,64,100,3)))
    print(z)