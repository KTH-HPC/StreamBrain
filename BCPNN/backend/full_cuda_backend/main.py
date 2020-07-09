import numpy as np
import backend_full_cuda_internals as backend_cuda
import faulthandler

def load_mnist(image_file, label_file, dtype=np.float64):
    imgs = np.fromfile(image_file, dtype=np.uint8)
    imgs = imgs[16:]
#    imgs = np.reshape(imgs, [-1, 28, 28])
    imgs = np.reshape(imgs, [-1, 28*28])
    imgs = imgs.astype(dtype)
    imgs /= 255.0

    l = np.fromfile(label_file, dtype=np.uint8)
    l = l[8:]

    labels = np.zeros([l.shape[0], 10]).astype(dtype)
    labels[np.arange(l.shape[0]), l] = 1.0
        
    return imgs, labels

backend_cuda.initialize()
#faulthandler.enable()

training_images, training_labels = load_mnist("/tmp/train-images-idx3-ubyte","/tmp/train-labels-idx1-ubyte", np.float32)
testing_images, testing_labels = load_mnist("/tmp/t10k-images-idx3-ubyte", "/tmp/t10k-labels-idx1-ubyte", np.float32)

net = backend_cuda.PyNetwork_float32()

net.add_plastic_layer(28*28, 30, 100)
net.add_dense_layer(3000, 1, 10)

net.initiate_training(training_images, training_labels)
net.train_layer(128, 0, 15)
net.train_layer(128, 1, 25)
print(net.evaluate(testing_images, testing_labels))
