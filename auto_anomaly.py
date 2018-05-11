import numpy as np
import matplotlib.pyplot as plt
from vae import *

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def show_image(X, names, idx):
	img = np.reshape(X[idx],(32,32,3),order='F')
	print img.dtype
	img_T = np.zeros(img.shape,dtype='uint8')
	img_T[:,:,0] = img[:,:,0].T
	img_T[:,:,1] = img[:,:,1].T
	img_T[:,:,2] = img[:,:,2].T
	print img[:,:,2].T
	print img_T[:,:,2]
	plt.imshow(img_T)
	plt.title(names[idx])
	plt.show()

def show_image_and_reconstruct(vae, X, names, idx):
	fig=plt.figure(figsize=(8, 4))
	img = np.reshape(X[idx],(32,32,3),order='F')
	img_T = np.zeros(img.shape,dtype='uint8')
	img_T[:,:,0] = img[:,:,0].T
	img_T[:,:,1] = img[:,:,1].T
	img_T[:,:,2] = img[:,:,2].T

	# somehow generateing a few samples here?
	x_recon = vae.reconstruct(np.reshape(X[idx]/255.0,(1,3072)))[0] * 255
	x_recon = np.reshape(x_recon,(32,32,3),order='F')
	x_recon = x_recon.astype('uint8')

	fig.add_subplot(1, 2, 1)
	plt.imshow(x_recon)
	plt.title("VAE Reconstruction")
	fig.add_subplot(1, 2, 2)
	plt.imshow(img)
	plt.title(names[idx])
	#fig.add_subplot(1, 3, 3)
	#plt.imshow(abs(img-x_recon)/255.0,cmap=plt.cm.gray)
	#plt.title("Diff Image ({}) ".format(2.3))
	plt.show()

def compute_image_delta(x):
	img = np.reshape(x,(32,32,3),order='F')
	x_recon = vae.reconstruct(np.reshape(x/255.0,(1,3072)))[0] * 255
	x_recon = np.reshape(x_recon,(32,32,3),order='F')
	#x_recon = x_recon.astype('uint8')
	return np.sum(np.abs(img-x_recon))


test = unpickle("/home/idewanck/Downloads/cifar-100-python/test")
train = unpickle("/home/idewanck/Downloads/cifar-100-python/train")

X_test = test["data"]
y_test = test["coarse_labels"]
names_test = test["filenames"]

X_train = train["data"]
y_train = train["coarse_labels"]
names_train = train["filenames"]


# Test some autoencoder losses
network_architecture = \
    dict(n_hidden_recog_1=750, # 1st layer encoder neurons
         n_hidden_recog_2=600, # 2nd layer encoder neurons
         n_hidden_gener_1=750, # 1st layer decoder neurons
         n_hidden_gener_2=600, # 2nd layer decoder neurons
         n_input=3072, # CIFAR-100 data input (img shape: 32*32)
         n_z=20)  # dimensionality of latent space

# only show the VAE non-motorcycles (everything but class 18)
non_bikes_train = np.where(np.array(y_train) != 18)
non_bikes_test = np.where(np.array(y_test) != 18)
bikes_test = np.where(np.array(y_test) == 18)

vae = vae_train(X_train[non_bikes_train], network_architecture, 
	            training_epochs=35,learning_rate=0.0002)

#show_image_and_reconstruct(vae,X_test,names_test,677)
#show_image_and_reconstruct(vae,X_test,names_test,887)
#show_image_and_reconstruct(vae,X_test,names_test,4130)
#show_image_and_reconstruct(vae,X_test,names_test,4137)
#show_image_and_reconstruct(vae,X_test,names_test,4139)
#show_image_and_reconstruct(vae,X_test,names_test,4143)

# Hopefully train to recognize that class 18 is novel

#show_image_and_reconstruct(vae,X_test,names_test,non_bikes_test[0][0])
#show_image_and_reconstruct(vae,X_test,names_test,bikes_test[0][0])

bike_test_deltas = [compute_image_delta(X_test[idx]) for idx in bikes_test[0]]
non_bike_test_deltas = [compute_image_delta(X_test[idx]) for idx in non_bikes_test[0]]

bins = np.linspace(10000, 250000, 1000)
plt.hist(bike_test_deltas, bins, alpha=0.5, label='vehicles', density=True, color='r')
plt.hist(non_bike_test_deltas, bins, alpha=0.5, label='non-vehicles', density=True, color='b')
plt.axvline(np.mean(bike_test_deltas), color='r')
plt.axvline(np.mean(non_bike_test_deltas), color='b')
plt.legend(loc='upper right')
plt.show()

# TP (vehicles that classified as novel)
# FP (non-vehicle that is classified as novel)
# TN (non-vehicle that is classified as not novel)
# FN (vehicle that is classified as not novel)

idx_delta_bikes = [(idx,compute_image_delta(X_test[idx])) for idx in bikes_test[0]]
idx_delta_non_bikes = [(idx,compute_image_delta(X_test[idx])) for idx in non_bikes_test[0]]

anom_thresh = 110000
tp = [idx for idx,delta in idx_delta_bikes if delta > anom_thresh]
fp = [idx for idx,delta in idx_delta_non_bikes if delta > anom_thresh]

fn = [idx for idx,delta in idx_delta_bikes if delta < anom_thresh]
tn = [idx for idx,delta in idx_delta_non_bikes if delta < anom_thresh]

def perf_stats(tp,fp,tn,fn):
	print "FPR : ",float(len(fp)) / float( len(fp) + len(tn))
	print "FNR : ",float(len(fn)) / float( len(fn) + len(tp))

perf_stats(tp,fp,tn,fn)