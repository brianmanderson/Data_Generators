__author__ = 'Brian M Anderson'
# Created on 4/7/2020
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image, plt
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np


def load_obj(path):
    if path.find('.pkl') == -1:
        path += '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        out = {}
        return out


class PyTorchDataset(Dataset):
    def __init__(self, record_paths, shuffle=False, debug=False,
                 transform=None, target_transform=None):
        """
        :param record_paths: List of paths to a folder full of .pkl files
        :param in_parallel: Boolean, perform the actions in parallel?
        :param delete_old_cache: Boolean, delete the previous cache?
        :param shuffle: Boolean, shuffle the record names?
        :param debug: Boolean, debug process
        """
        self.transform = transform
        self.target_transform = target_transform
        assert record_paths is not None, 'Need to pass a list of record names!'
        if not isinstance(record_paths, list):
            raise ValueError("Provide a list of record paths.")
        self.record_names = []
        for record_path in record_paths:
            assert os.path.isdir(record_path), 'Pass a directory, not a tfrecord\n{}'.format(record_path)
            self.record_names += [os.path.join(record_path, i) for i in os.listdir(record_path) if i.endswith('.pkl')]

    def __getitem__(self, index):
        record = load_obj(self.record_names[index])
        return record['ct_array'], record['mask_array']

    def __len__(self):
        return len(self.record_names)


if __name__ == '__main__':
    pass
