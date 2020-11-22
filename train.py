import numpy as np
import BCPNN
import time

np.random.seed(0)

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
    precision = np.float32

    training_images, training_labels = load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", dtype=precision)
    testing_images, testing_labels = load_mnist("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", dtype=precision)

    n_inputs = 28*28
    n_hypercolumns = 30
    n_minicolumns = 100
    n_hidden = n_hypercolumns * n_minicolumns
    n_outputs = 10

    taupdt = 0.002996755526968425
    l1_epochs = 15 #23
    l2_epochs = 25 #298

    l1_pmin = 0.3496214817513042
    l1_khalf = -435.08426155834593
    l1_taubdt = 0.27826430798917945

    net = BCPNN.Network(precision)
    net.add_layer(BCPNN.StructuralPlasticityLayer(n_inputs, n_hypercolumns, n_minicolumns, taupdt, l1_khalf, l1_pmin, l1_taubdt, (1, 1/n_minicolumns, 1 * 1/n_minicolumns)))
    net.add_layer(BCPNN.DenseLayer(n_hidden, 1, n_outputs, taupdt, (1/n_minicolumns, 1/10, 1/n_minicolumns * 1/10)))

    train_start = time.time()
    net.fit(training_images, training_labels, batch_size, [(0, l1_epochs), (1, l2_epochs)])
    train_stop = time.time()

    test_start = time.time()
    accuracy = net.evaluate(testing_images, testing_labels, batch_size)
    test_stop = time.time()

    print('Training duration: '+str(train_stop-train_start))
    print('Testing duration: '+str(test_stop-test_start))
    print('Accuracy: '+str(accuracy))
