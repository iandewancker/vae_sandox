import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image
import glob, os


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

def load_box_rgb_images_full_size(grasp_dir, regex):
    png_paths = glob.glob(grasp_dir+"/**/"+regex)
    print("{} images found!".format(len(png_paths)))
    X = []
    for path in png_paths:
        if any(box_id in path for box_id in box_grasp_ids):
            img = Image.open(path)
            img_resize = img.resize((512,254))
            X.append(np.array(img_resize))
    X = np.stack(X,axis=0)
    return X

def load_rgb_images_full_size(grasp_dir, regex):
    png_paths = glob.glob(grasp_dir+"/**/"+regex)
    print("{} images found!".format(len(png_paths)))
    X = []
    for path in png_paths:
        img = Image.open(path)
        img_resize = img.resize((512,254))
        X.append(np.array(img_resize))
    X = np.stack(X,axis=0)
    return X

X_train_unpick = load_box_rgb_images("/home/ubuntu/data/help_grasps/", "*128_64_rgb.png")
X_train_unpick.shape

#X_train_empty = load_rgb_images("/home/ubuntu/data/empty_bin_grasps/", "*128_64_rgb.png")
#X_train = load_rgb_images("/home/ubuntu/data/normal_grasps/", "*128_64_rgb.png")
X_train_normal = load_rgb_images("/home/ubuntu/data/canonical_train_grasps/", "*128_64_rgb.png")

# testing with full size images
#X_train_unpick = load_box_rgb_images_full_size("/home/ubuntu/data/help_grasps/", "rgb.png")
#X_train_unpick.shape

#X_train_empty = load_rgb_images("/home/ubuntu/data/empty_bin_grasps/", "rgb.png")
#X_train = load_rgb_images("/home/ubuntu/data/normal_grasps/", "*128_64_rgb.png")
#X_train_normal = load_rgb_images_full_size("/home/ubuntu/data/canonical_train_grasps/", "rgb.png")

def adhoc_raw_features(img_path):
    img = Image.open(img_path)
    img_resize = img.resize((128,64))
    img_resize = img_resize.crop((0,0,100,64))
    x = np.array(img_resize)
    x_input = np.reshape(x,(1,64,100,3))
    return x_input

X_unpick_adhoc = [
    adhoc_raw_features("test_unpick_1.png"),
    adhoc_raw_features("test_unpick_2.png"),
    adhoc_raw_features("test_unpick_3.png"),
    adhoc_raw_features("test_unpick_4.png"),
    adhoc_raw_features("test_unpick_5.png"),
    adhoc_raw_features("test_unpick_6.png"),
    adhoc_raw_features("test_unpick_7.png"),
    adhoc_raw_features("test_unpick_8.png"),
    adhoc_raw_features("test_unpick_9.png"),
    adhoc_raw_features("test_unpick_10.png"),
    adhoc_raw_features("test_unpick_11.png"),
    adhoc_raw_features("test_unpick_12.png"),
    adhoc_raw_features("test_unpick_13.png"),
    adhoc_raw_features("test_unpick_14.png"),
    adhoc_raw_features("test_unpick_15.png"),
    adhoc_raw_features("test_unpick_16.png"),
    adhoc_raw_features("test_unpick_17.png"),
    adhoc_raw_features("test_unpick_18.png"),
    adhoc_raw_features("test_unpick_19.png"),
    adhoc_raw_features("test_unpick_20.png"),
    adhoc_raw_features("test_unpick_21.png"),
    adhoc_raw_features("test_unpick_22.png"),
    adhoc_raw_features("test_unpick_23.png"),
    adhoc_raw_features("test_unpick_24.png"),
    adhoc_raw_features("test_unpick_25.png"),
    adhoc_raw_features("test_unpick_26.png"),
    adhoc_raw_features("test_unpick_27.png"),
    adhoc_raw_features("test_unpick_28.png"),
    adhoc_raw_features("test_unpick_29.png"),
    adhoc_raw_features("test_unpick_30.png"),
    adhoc_raw_features("test_unpick_31.png"),
    adhoc_raw_features("test_unpick_32.png"),
    adhoc_raw_features("test_unpick_33.png"),
    adhoc_raw_features("test_unpick_34.png"),
    adhoc_raw_features("test_unpick_35.png"),
    adhoc_raw_features("test_unpick_36.png"),
    adhoc_raw_features("test_unpick_37.png"),
    adhoc_raw_features("test_unpick_38.png"),
    adhoc_raw_features("test_unpick_39.png"),
    adhoc_raw_features("test_unpick_40.png"),
    adhoc_raw_features("test_unpick_41.png"),
    adhoc_raw_features("test_unpick_42.png"),
    adhoc_raw_features("test_unpick_43.png"),
    adhoc_raw_features("test_unpick_44.png"),
    adhoc_raw_features("test_unpick_45.png"),
    adhoc_raw_features("test_unpick_46.png"),
    adhoc_raw_features("test_unpick_47.png"),
    adhoc_raw_features("test_unpick_48.png"),
    adhoc_raw_features("test_unpick_49.png"),
    adhoc_raw_features("test_unpick_50.png"),
    adhoc_raw_features("test_unpick_51.png"),
    adhoc_raw_features("test_unpick_52.png"),
    adhoc_raw_features("test_unpick_53.png"),
    adhoc_raw_features("test_unpick_54.png"),
    adhoc_raw_features("test_unpick_55.png"),
    adhoc_raw_features("test_unpick_56.png"),
    adhoc_raw_features("test_unpick_57.png"),
    adhoc_raw_features("test_unpick_58.png"),
    adhoc_raw_features("test_unpick_59.png"),
    adhoc_raw_features("test_unpick_60.png"),
    adhoc_raw_features("test_unpick_61.png"),
    adhoc_raw_features("test_unpick_62.png"),
    adhoc_raw_features("test_unpick_63.png"),
    adhoc_raw_features("test_unpick_64.png"),
    adhoc_raw_features("test_unpick_65.png"),
    adhoc_raw_features("test_unpick_66.png"),
    adhoc_raw_features("test_unpick_67.png"),
    adhoc_raw_features("test_unpick_68.png"),
    adhoc_raw_features("test_unpick_69.png"),
    adhoc_raw_features("test_unpick_70.png"),
    adhoc_raw_features("test_unpick_71.png"),
    adhoc_raw_features("test_unpick_72.png"),
    adhoc_raw_features("test_unpick_73.png"),
    adhoc_raw_features("test_unpick_74.png"),
    adhoc_raw_features("test_unpick_75.png"),
    adhoc_raw_features("test_unpick_76.png"),
    adhoc_raw_features("test_unpick_77.png"),
    adhoc_raw_features("test_unpick_78.png"),
    adhoc_raw_features("test_unpick_79.png"),
    adhoc_raw_features("test_unpick_80.png"),
    adhoc_raw_features("test_unpick_81.png"),
    adhoc_raw_features("test_unpick_82.png"),
    adhoc_raw_features("test_unpick_83.png"),
    adhoc_raw_features("test_unpick_84.png"),
    adhoc_raw_features("test_unpick_85.png"),
    adhoc_raw_features("test_unpick_86.png"),
    adhoc_raw_features("test_unpick_87.png"),
    adhoc_raw_features("test_unpick_88.png"),
    adhoc_raw_features("test_unpick_89.png"),
    adhoc_raw_features("test_unpick_90.png"),
    adhoc_raw_features("test_unpick_91.png"),
    adhoc_raw_features("test_unpick_92.png"),
    adhoc_raw_features("test_unpick_93.png"),
    adhoc_raw_features("test_unpick_94.png"),
    adhoc_raw_features("test_unpick_95.png"),
]
X_unpick_adhoc = np.vstack(X_unpick_adhoc)

X_train_unpick = np.vstack([X_train_unpick,X_unpick_adhoc])


# split up normal and unpickable into train and validation set
test_perc = 0.25
shuff_idxs = np.arange(X_train_unpick.shape[0])
np.random.shuffle(shuff_idxs)
test_idx = int(X_train_unpick.shape[0]*test_perc)
X_test_unpick = X_train_unpick[shuff_idxs[:test_idx]]
X_train_unpick = X_train_unpick[shuff_idxs[test_idx:]]

shuff_idxs = np.arange(X_train_normal.shape[0])
np.random.shuffle(shuff_idxs)
test_idx = int(X_train_normal.shape[0]*test_perc)
X_test_normal  = X_train_normal[shuff_idxs[:test_idx]]
X_train_normal = X_train_normal[shuff_idxs[test_idx:]]

tf.reset_default_graph()
def mynet(input, reuse=False):
    with tf.name_scope("model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 64, [30, 30], stride=7, activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                biases_initializer=tf.contrib.layers.xavier_initializer(), scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        
        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [10, 10],stride=4, activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                biases_initializer=tf.contrib.layers.xavier_initializer(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 32, [5, 5],stride=2, activation_fn=tf.tanh, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                biases_initializer=tf.contrib.layers.xavier_initializer(),scope=scope,reuse=reuse)

        with tf.variable_scope("fc1") as scope:
            net = tf.contrib.layers.fully_connected(net,5,scope=scope,reuse=reuse,activation_fn=None)
            net = tf.sigmoid(net)

        net = tf.layers.flatten(net,name='z')
        """
        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 10, [10, 10],stride=4, activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [2, 2], stride=2, activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        net = tf.layers.flatten(net,name='z')
        """

        """
        old mode
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv5") as scope:
            net = tf.contrib.layers.conv2d(net, 10, [1, 1], activation_fn=None, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        net = tf.layers.flatten(net,name='z')
        """
    return net


def contrastive_loss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keepdims=True))
        tmp= y * tf.square(d)    
        tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
        return tf.reduce_mean(tmp + tmp2) / 2


left = tf.placeholder(tf.float32, [None, 64, 100, 3], name='x')
right = tf.placeholder(tf.float32, [None, 64, 100, 3], name='right')
with tf.name_scope("similarity"):
    label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
    label = tf.to_float(label)
margin = 1.2

left_output = mynet(left, reuse=False)
right_output = mynet(right, reuse=True)

loss = contrastive_loss(left_output, right_output, label, margin)

# construct training data pairs :
# (left,right,sim), where sim = 0 if same class, 1 if different
train_left = []
train_right = []
train_sim = []

# similar normal images : take subset of all pairs
num_pairs = 400
index_list = np.random.choice(range(X_train_normal.shape[0]), num_pairs*2, replace=False).tolist()
similar_normals_count = 0
for i in range(num_pairs):
    idx1 = index_list.pop()
    idx2 = index_list.pop()
    train_left.append(X_train_normal[idx1])
    train_right.append(X_train_normal[idx2])
    train_sim.append(1)
    similar_normals_count += 1
print ("Generated ",similar_normals_count," similar normal grasp images")

# similar box images
similar_unpickable_count = 0
for i in range(X_train_unpick.shape[0]):
    for k in range(i+1,X_train_unpick.shape[0]):
        train_left.append(X_train_unpick[i])
        train_right.append(X_train_unpick[k])
        train_sim.append(1)
        similar_unpickable_count += 1
print ("Generated ",similar_unpickable_count," similar unpickable grasp images")

# dissimilar images
dissimilar_unpickable_count = 0
for i in range(0,X_train_normal.shape[0],150):
    for k in range(X_train_unpick.shape[0]):
        train_left.append(X_train_normal[i])
        train_right.append(X_train_unpick[k])
        train_sim.append(0)
        dissimilar_unpickable_count += 1
print ("Generated ",dissimilar_unpickable_count," dissimilar unpickable grasp images")

train_left = np.stack(train_left,axis=0)
train_right = np.stack(train_right,axis=0)
train_sim = np.stack(train_sim,axis=0)
train_sim = np.reshape(train_sim,(train_sim.shape[0],1))

# shuffle order of training examples
shuff_idxs = np.arange(train_sim.shape[0])
np.random.shuffle(shuff_idxs)
train_left = train_left[shuff_idxs]
train_right = train_right[shuff_idxs]
train_sim = train_sim[shuff_idxs]

training_epochs = 45
learning_rate = 0.00001
margin = 0.5
global_step = tf.Variable(0, trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
#optimizer = tf.train.MomentumOptimizer(learning_rate , 0.99, use_nesterov=True).minimize(loss, global_step=global_step)
batch_size = 500
n_samples = len(train_sim)
display_step=1
#tf.reset_default_graph()
# Initializing the tensor flow variables
init = tf.global_variables_initializer()

# Launch the session
sess = tf.InteractiveSession()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0.0
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        # assuming input in range from 0-1
        b_l = train_left[(i*batch_size):((i+1)*batch_size)]
        b_r = train_right[(i*batch_size):((i+1)*batch_size)]
        b_sim = train_sim[(i*batch_size):((i+1)*batch_size)]

        # Fit training using batch data
        _, cost = sess.run([optimizer, loss], 
            feed_dict={left:b_l, right:b_r, label: b_sim})
        
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), 
              "cost=", "{:.9f}".format(avg_cost))



import sklearn.neighbors
import sklearn.metrics
import sklearn.ensemble
import sklearn.linear_model
from sklearn.model_selection import train_test_split

def adhoc_transform(img_path):
    img = Image.open(img_path)
    img_resize = img.resize((128,64))
    img_resize = img_resize.crop((0,0,100,64))
    x = np.array(img_resize)
    #x_recon = vae.reconstruct(np.reshape(x/255.0,(1,64,100,3)))
    x_input = np.reshape(x,(1,64,100,3))
    z = sess.run([left_output], 
            feed_dict={left:x_input})
    return z[0]


Z_train_normal = []
for i in range(0,X_train_normal.shape[0],50):
    x_input = np.reshape(X_train_normal[i:i+50],(X_train_normal[i:i+50].shape[0],64,100,3))
    z = sess.run([left_output], 
            feed_dict={left:x_input})
    Z_train_normal.append(z[0])
Z_train_normal = np.vstack(Z_train_normal)

Z_test_normal = []
for i in range(0,X_test_normal.shape[0],50):
    x_input = np.reshape(X_test_normal[i:i+50],(X_test_normal[i:i+50].shape[0],64,100,3))
    z = sess.run([left_output], 
            feed_dict={left:x_input})
    Z_test_normal.append(z[0])
Z_test_normal = np.vstack(Z_test_normal)

Z_train_unpick = []
for i in range(0,X_train_unpick.shape[0],50):
    x_input = np.reshape(X_train_unpick[i:i+50],(X_train_unpick[i:i+50].shape[0],64,100,3))
    z = sess.run([left_output], 
            feed_dict={left:x_input})
    Z_train_unpick.append(z[0])
Z_train_unpick = np.vstack(Z_train_unpick)

Z_test_unpick = []
for i in range(0,X_test_unpick.shape[0],50):
    x_input = np.reshape(X_test_unpick[i:i+50],(X_test_unpick[i:i+50].shape[0],64,100,3))
    z = sess.run([left_output], 
            feed_dict={left:x_input})
    Z_test_unpick.append(z[0])
Z_test_unpick = np.vstack(Z_test_unpick)

Z_train_all = np.vstack([Z_train_normal, Z_train_unpick])
y_train_all = np.hstack([np.ones(Z_train_normal.shape[0]), np.zeros(Z_train_unpick.shape[0])])

Z_test_all = np.vstack([Z_test_normal, Z_test_unpick])
y_test_all = np.hstack([np.ones(Z_test_normal.shape[0]), np.zeros(Z_test_unpick.shape[0])])

def do_clf(Z_train, y_train, Z_test, y_test,clf):
    clf.fit(Z_train, y_train)
    y_train_pred = clf.predict(Z_train)
    y_test_pred = clf.predict(Z_test)
    fpr = np.where((y_train_pred == 1) & (y_train_pred != y_train))[0].shape[0]
    fnr = np.where((y_train_pred == 0) & (y_train_pred != y_train))[0].shape[0]
    print("TRAIN FPR : ",fpr," / ",np.where((y_train == 0))[0].shape[0]," FNR : ",fnr," / ",np.where((y_train == 1))[0].shape[0])
    fpr = np.where((y_test_pred == 1) & (y_test_pred != y_test))[0].shape[0]
    fnr = np.where((y_test_pred == 0) & (y_test_pred != y_test))[0].shape[0]
    print("TEST FPR : ",fpr," / ",np.where((y_test == 0))[0].shape[0]," FNR : ",fnr," / ",np.where((y_test == 1))[0].shape[0])
    test_fpr = fpr / float(np.where((y_test == 0))[0].shape[0])
    test_fnr = fnr / float(np.where((y_test == 1))[0].shape[0])

    Z_all = np.vstack([Z_train,Z_test])
    y_all = np.hstack([y_train,y_test])
    #clf.fit(Z_all, y_all)
    y_pred_all = clf.predict(Z_all)
    fpr = np.where((y_pred_all == 1) & (y_pred_all != y_all))[0].shape[0]
    fnr = np.where((y_pred_all == 0) & (y_pred_all != y_all))[0].shape[0]
    print("ALL FPR : ",fpr," / ",np.where((y_all == 0))[0].shape[0]," FNR : ",fnr," / ",np.where((y_all == 1))[0].shape[0])
    return (test_fpr, test_fnr)

#clf = sklearn.linear_model.LogisticRegression(class_weight={0.0: 0.5, 1.0:0.1},penalty='l2',C=0.000455)
clf = sklearn.linear_model.LogisticRegression(class_weight={0.0: 65.1, 1.0:0.1},C=0.0017)
clf = sklearn.linear_model.LogisticRegression(class_weight={0.0: 65.1, 1.0:0.1},C=0.409)
#clf = sklearn.linear_model.LogisticRegression()
#clf = sklearn.linear_model.LogisticRegression(class_weight={0.0: 45.1, 1.0:0.1},C=0.0001)
#clf = sklearn.linear_model.LogisticRegression()
clf = sklearn.linear_model.LogisticRegression(class_weight={0.0: 35.1, 1.0:0.1},C=0.001)
do_clf(Z_train_all, y_train_all, Z_test_all, y_test_all,clf)


def adhoc_clf(img_path,clf):
    z = adhoc_transform(img_path)
    return clf.predict(z)

from tensorflow.python.framework.graph_util import convert_variables_to_constants
from sklearn.externals import joblib

# access the default graph
graph = tf.get_default_graph()

# retrieve the protobuf graph definition
input_graph_def = graph.as_graph_def()
#test_graph_def = tf.graph_util.remove_training_nodes(
#    input_graph_def,
#    protected_nodes=None
#)

output_node_names = "x,z"

# TensorFlow built-in helper to export variables to constants
output_graph_def = convert_variables_to_constants(
    sess=sess,
    input_graph_def=input_graph_def, # GraphDef object holding the network
    output_node_names=output_node_names.split(",") # List of name strings for the result nodes of the graph
) 

model_name = "siamese_cnn_123k_latent_16_date_06_07"

tf.train.write_graph(output_graph_def,
                     "./",
                     model_name+"_graph.pb",
                     as_text=False)

joblib.dump(clf,model_name+"_clf.pkl")