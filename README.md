# ModelNet40 Canonicalization Experiments

This repository contains the code and experimental setup for studying the effects of point cloud canonicalization on the ModelNet40 dataset. The project evaluates canonicalization techniques using a global MLP and conducts a comprehensive rotation study across various frame-alignment models.

## 📂 Data Preparation

Before running the experiments, ensure you have downloaded the **ModelNet40** dataset in HDF5 format. The data should be extracted and placed in the following directory: `~/data/modelnet40_ply_hdf5_2048/`.

Your directory structure must look exactly like this:

```text
~/data/modelnet40_ply_hdf5_2048$ ls
fps_cache                     ply_data_test1.h5             ply_data_train_0_id2file.json  ply_data_train2.h5             ply_data_train_3_id2file.json  shape_names.txt
ply_data_test0.h5             ply_data_test_1_id2file.json  ply_data_train1.h5             ply_data_train_2_id2file.json  ply_data_train4.h5             test_files.txt
ply_data_test_0_id2file.json  ply_data_train0.h5            ply_data_train_1_id2file.json  ply_data_train3.h5             ply_data_train_4_id2file.json  train_files.txt