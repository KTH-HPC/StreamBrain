import numpy as np
import BCPNN
import math
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

def load_stl_10_images(file):
    data = np.fromfile(file, dtype=np.uint8)
    data = np.reshape(data, [-1, 3 * 96 * 96])
    data = data / 255.0

    return data

def load_stl_10_labels(file):
    l = np.fromfile(file, dtype=np.uint8) - 1
    count = l.shape[0]

    lbls = np.zeros([count, 10])
    lbls[np.arange(count), l] = 1.0

    return lbls

if __name__ == "__main__":
    import sys
    from mpi4py import MPI

    if len(sys.argv) != 4:
        print("Usage: python", sys.argv[0], "<mnist/fashion_mnist/stl-10> <single/double> <batch size>")
        sys.exit(1)

    world_rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

    dataset = sys.argv[1]
    if sys.argv[2] == "single":
        precision = np.float32
    else:
        precision = np.float64
    batch_size = int(sys.argv[3])

    print('Dataset:', dataset, 'Batch size:', batch_size, 'precision:', precision)

    if dataset == "mnist":
      n_hypercolumns = 15

      l1_taupdt = 0.006908854922437549
      l1_pmin = 0.06195283356566607
      l1_khalf = -139.02586632856327
      l1_taubdt = 0.02218100432360414
      l1_density = 0.1
      l1_mask_iterations = 16
      l2_taupdt = 0.0007449780036161324
      l1_epochs = 30
      l2_epochs = 60

      n_inputs = 28*28
      l1_training_images, l1_training_labels = load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", dtype=precision)
      l2_training_images = l1_training_images
      l2_training_labels = l1_training_labels
#      l1_training_labels = np.zeros([l1_training_images.shape[0], 10])
#      l2_training_images = l1_training_images

      testing_images, testing_labels = load_mnist("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", dtype=precision)
    elif dataset == "fashion_mnist":
      n_hypercolumns = 30

      l1_taupdt = 0.0014712266428529468
      l1_pmin = 0.10328182984859212
      l1_khalf = -160.06386313883948
      l1_taubdt = 0.0008418555122620076
      l1_density = 0.1
      l1_mask_iterations = 16
      l2_taupdt = 0.00202951854188782
      l1_epochs = 30
      l2_epochs = 60

      n_inputs = 28*28
      l1_training_images, l1_training_labels = load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", dtype=precision)
      l2_training_images = l1_training_images
      l2_training_labels = l1_training_labels

      testing_images, testing_labels = load_mnist("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", dtype=precision)
    elif dataset == "stl-10":
      n_hypercolumns = 15

      l1_taupdt = 0.0006510421564830368
      l1_pmin = 0.0070169782620369365
      l1_khalf = 1666.644815059196
      l1_taubdt = 0.0024793529936481956
      l1_density = 0.568746541818903
      l1_mask_iterations = 584
      l2_taupdt = 4.983000352087509e-07
      l1_epochs = 10
      l2_epochs = 20
      n_inputs = 3*96*96

      l1_training_images = load_stl_10_images("unlabeled_X.bin")[:15000, :]
      l1_training_labels = np.zeros([l1_training_images.shape[0], 10])

      l2_training_images = load_stl_10_images("train_X.bin")
      l2_training_labels = load_stl_10_labels("train_y.bin")

      testing_images = load_stl_10_images("test_X.bin")
      testing_labels = load_stl_10_labels("test_y.bin")
    else:
      print("Unknown dataset")
      sys.exit(1)

    l1_taupdt *= math.sqrt(batch_size / 128)
    l2_taupdt *= math.sqrt(batch_size / 128)

    n_minicolumns = (3000 // n_hypercolumns)
    n_hidden = n_hypercolumns * n_minicolumns
    n_outputs = 10

    net = BCPNN.Network(precision)
    net.add_layer(BCPNN.StructuralPlasticityLayer(n_inputs, n_hypercolumns, n_minicolumns, l1_taupdt, l1_khalf, l1_pmin, l1_taubdt, l1_density, l1_mask_iterations, (1, 1/n_minicolumns, 1 * 1/n_minicolumns)))
    net.add_layer(BCPNN.DenseLayer(n_hidden, 1, n_outputs, l2_taupdt, (1/n_minicolumns, 1/10, 1/n_minicolumns * 1/10)))

    train_start = time.time()
    net.fit(l1_training_images, l1_training_labels, batch_size, [(0, l1_epochs)])
    net.fit(l1_training_images, l1_training_labels, batch_size, [(1, l2_epochs)])
    train_stop = time.time()

    test_start = time.time()
    correct, total = net.evaluate(testing_images, testing_labels, batch_size)
    test_stop = time.time()

    train_duration = np.array(train_stop-train_start).astype(np.double)
    test_duration = np.array(test_stop-test_start).astype(np.double)
    print(world_rank, train_duration, test_duration, correct, total)

    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, train_duration, op=MPI.MAX)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, test_duration,  op=MPI.MAX)

    if world_rank == 0:
        print('Training duration: '+str(train_duration))
        print('Testing duration:  '+str(test_duration))
        print('Accuracy:          '+str(correct / total))
