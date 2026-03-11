#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def load_ordered_data(partition, ordering_type="lex", data_root="data"):
    """
    Loads pre-ordered data from either the 'lex' or 'hilbert' directories.
    """
    if ordering_type not in ["lex", "hilbert"]:
        raise ValueError(f"Unknown ordering type: {ordering_type}. Must be 'lex' or 'hilbert'.")

    dataset_name = f"modelnet40_{ordering_type}_hdf5_2048"
    dataset_dir = os.path.join(data_root, dataset_name)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Could not find dataset directory at {dataset_dir}")

    all_data = []
    all_label = []

    file_pattern = os.path.join(dataset_dir, f'ply_data_{partition}*.h5')
    files = glob.glob(file_pattern)

    if not files:
        raise FileNotFoundError(f"No h5 files found matching: {file_pattern}")

    for h5_name in files:
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            all_data.append(data)
            all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label


class OrderedModelNet40(Dataset):
    def __init__(self, num_points, partition='train', ordering='lex', data_root='data'):
        """
        Dataloader for pre-ordered Point Clouds. 
        NO shuffling or rotations are allowed to preserve the strict 1D sequence.
        """
        self.data, self.label = load_ordered_data(partition, ordering_type=ordering, data_root=data_root)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        # Slice to the requested number of points. 
        # (Assuming the original canonicalization script preserved the ordering in the first N points)
        # Calculate the stride to evenly sample the points while preserving order
        stride = self.data[item].shape[0] // self.num_points
        pointcloud = self.data[item][::stride][:self.num_points]
        label = self.label[item]

        # NOTE: We intentionally DO NOT shuffle, translate, or rotate here.
        # Doing so would destroy the Lexicographical or Hilbert sequence order!

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    # Quick sanity check
    train = OrderedModelNet40(1024, partition='train', ordering='hilbert')
    for data, label in train:
        print("Point Cloud Shape:", data.shape)
        print("Label Shape:", label.shape)
        break