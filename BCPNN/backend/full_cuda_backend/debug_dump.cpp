/*
  float * h_Ci = (float *)malloc(n_hidden * sizeof(float));
  float * h_Cj = (float *)malloc(n_outputs * sizeof(float));
  float * h_Cij = (float *)malloc(n_hidden * n_outputs * sizeof(float));

  CUDA_CALL(cudaMemcpy(h_Ci, Ci, n_hidden * sizeof(float),
  cudaMemcpyDeviceToHost)); CUDA_CALL(cudaMemcpy(h_Cj, Cj, n_outputs *
  sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CALL(cudaMemcpy(h_Cij, Cij,
  n_hidden * n_outputs * sizeof(float), cudaMemcpyDeviceToHost));

  std::ofstream ci("/tmp/ci", std::ios::binary);
  std::ofstream cj("/tmp/cj", std::ios::binary);
  std::ofstream cij("/tmp/cij", std::ios::binary);

  ci.write((char *)h_Ci, n_hidden * sizeof(float));
  cj.write((char *)h_Cj, n_outputs * sizeof(float));
  cij.write((char *)h_Cij, n_hidden * n_outputs * sizeof(float));

  ci.flush();
  cj.flush();
  cij.flush();
*/
