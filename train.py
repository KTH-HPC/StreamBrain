import numpy as np
import BCPNN
import math
import time

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python", sys.argv[0], "<batch size>")
        sys.exit(1)

    batch_size = int(sys.argv[1])

    training_images, training_labels = load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    testing_images, testing_labels = load_mnist("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")

    n_inputs = 28*28

    l1_epochs = 30
    l2_epochs = 60

    dataset = "mnist"
    if dataset == "mnist":
      n_hypercolumns = 15
      l1_taupdt = 0.006908854922437549
      l1_pmin = 0.06195283356566607
      l1_khalf = -139.02586632856327
      l1_taubdt = 0.02218100432360414
      l2_taupdt = 0.0007449780036161324
    elif dataset == "fashion_mnist":
      n_hypercolumns = 30
      l1_taupdt = 0.0014712266428529468
      l1_pmin = 0.10328182984859212
      l1_khalf = -160.06386313883948
      l1_taubdt = 0.0008418555122620076
      l2_taupdt = 0.00202951854188782
    else:
      print("Unknown dataset")
      sys.exit(1)

    l1_taupdt *= math.sqrt(batch_size / 128)
    l2_taupdt *= math.sqrt(batch_size / 128)

    n_minicolumns = 3000 // n_hypercolumns
    n_hidden = n_hypercolumns * n_minicolumns
    n_outputs = 10

    #net = BCPNN.Network(np.float32)
    net = BCPNN.Network(np.float64)
    net.add_layer(BCPNN.StructuralPlasticityLayer(n_inputs, n_hypercolumns, n_minicolumns, l1_taupdt, l1_khalf, l1_pmin, l1_taubdt, (1, 1/n_minicolumns, 1 * 1/n_minicolumns)))
    net.add_layer(BCPNN.DenseLayer(n_hidden, 1, n_outputs, l2_taupdt, (1/n_minicolumns, 1/10, 1/n_minicolumns * 1/10)))

    train_start = time.time()
    net.fit(training_images, training_labels, batch_size, [(0, l1_epochs), (1, l2_epochs)])
    train_stop = time.time()

    test_start = time.time()
    accuracy = net.evaluate(testing_images, testing_labels, batch_size)
    test_stop = time.time()

    print('Training duration: '+str(train_stop-train_start))
    print('Testing duration: '+str(test_stop-test_start))
    print('Accuracy: '+str(accuracy))
