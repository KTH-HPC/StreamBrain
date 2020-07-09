


#if 0

size_t batch_size = 0;
size_t constexpr n_inputs = 28*28;
size_t constexpr n_hypercolumns = 30;
size_t constexpr n_minicolumns = 100;
size_t constexpr n_hidden = n_hypercolumns * n_minicolumns;
size_t constexpr n_outputs = 10;

int
main(int argc, char * argv[])
{
  if (argc != 2) { std::cerr << "Usage: " << argv[0] << " <batch size>" << std::endl; exit(1); }

  batch_size = atoi(argv[1]);

  seed_generator(generator);

  dataset_t<uint8_t, uint8_t> _train_dataset;
  dataset_t<uint8_t, uint8_t> _test_dataset;

  _train_dataset = read_mnist_dataset("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
  _test_dataset = read_mnist_dataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

  dataset_t<float, float> train_dataset = convert_dataset<float>(_train_dataset);
  dataset_t<float, float> test_dataset  = convert_dataset<float>(_test_dataset);

  std::cout << "Training set:" << std::endl;
  std::cout << "=============" << std::endl;
  print_dataset(train_dataset);

  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Test set:" << std::endl;
  std::cout << "=========" << std::endl;
  print_dataset(test_dataset);

  dataloader loader(train_dataset, 8, 2);

  CUBLAS_CALL(cublasCreate(&handle));
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

#ifndef BCPNN_FETCH_PARAMETERS
#if 0
  // 94.8% or so, has reached > 95%
  float taupdt = 0.006000765502330436;
  size_t l1_epochs = 325;
  size_t l2_epochs = 347;

  float l1_pmin = 0.07830636855214834;
  float l1_khalf = -81.55156765133142;
  float l1_taubdt = 0.054491579066962004;
#endif
#if 1
  // 94% - 94.5%
  float taupdt = 0.002996755526968425;
  size_t l1_epochs = 23;
  size_t l2_epochs = 298;

  float l1_pmin = 0.3496214817513042;
  float l1_khalf = -435.08426155834593;
  float l1_taubdt = 0.27826430798917945;
#endif
#if 0
  // 93% - 94%
  float taupdt = 0.000861296756773373;
  size_t l1_epochs = 50;
  size_t l2_epochs = 250;

  float l1_pmin = 0.1911360664476474;
  float l1_khalf = -378.52897008489674;
  float l1_taubdt = 0.1395597316674668;
#endif
#else
  float taupdt;
  size_t l1_epochs;
  size_t l2_epochs;

  float l1_pmin;
  float l1_khalf;
  float l1_taubdt;

  char const* BASE_URL = "http://172.17.0.3:5000";
  std::stringstream url;

  url.str(""); url.clear();
  url << BASE_URL << "/new";

  std::string trial_id = make_request(url.str().c_str());

  std::string parameters;
  while (parameters.size() == 0) {
    url.str(""); url.clear();
    url << BASE_URL << "/get/" << trial_id;
    parameters = make_request(url.str().c_str());
    usleep(1000000);
  }

  get_parameters(parameters.c_str(), parameters.size(), &taupdt, &l1_epochs, &l2_epochs, &l1_pmin, &l1_khalf, &l1_taubdt);
#endif

  // Create layers
  MaskedDenseLayer layer_1(n_inputs, n_hypercolumns, n_minicolumns);
  DenseLayer       layer_2(n_hypercolumns * n_minicolumns, 1, 10);

  layer_1.taupdt = taupdt;
  layer_1.pmin = l1_pmin;
  layer_1.khalf = l1_khalf;
  layer_1.taubdt = l1_taubdt;

  layer_2.taupdt = taupdt;

  l1_epochs = 20;
  l2_epochs = 25;

  Network network;
  network.layers_.push_back(&layer_1);
  network.layers_.push_back(&layer_2);

  network.train_layer(loader, batch_size, 0, 15);
  network.train_layer(loader, batch_size, 1, 25);

  double accuracy = network.evaluate(test_dataset, batch_size);

  std::cout << accuracy << std::endl;

#ifdef BCPNN_FETCH_PARAMETERS
  url.str(""); url.clear();
  url << BASE_URL << "/complete/" << trial_id << "/" << accuracy;
  make_request(url.str().c_str());
#endif

  CURAND_CALL(curandDestroyGenerator(gen));
  loader.stop();

  return 0;
}

#endif
