__author__ = 'Brian M Anderson'
# Created on 4/7/2020
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image, plt
from Image_Processors_Module.src.Processors.TFDataSetProcessors import DecodeImagesAnnotations
import glob
import pickle
import tensorflow as tf
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


def return_parse_function(image_feature_description):

    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, image_feature_description)
    return _parse_image_function


class DataGeneratorClass(object):
    def __init__(self, record_paths=None, in_parallel=True, delete_old_cache=False, shuffle=False, debug=False):
        """
        :param record_paths: List of paths to a folder full of records files
        :param in_parallel: Boolean, perform the actions in parallel?
        :param delete_old_cache: Boolean, delete the previous cache?
        :param shuffle: Boolean, shuffle the record names?
        :param debug: Boolean, debug process
        """
        self.delete_old_cache = delete_old_cache
        if in_parallel:
            self.in_parallel = tf.data.experimental.AUTOTUNE
        else:
            self.in_parallel = None
        assert record_paths is not None, 'Need to pass a list of record names!'
        if not isinstance(record_paths, list):
            raise ValueError("Provide a list of record paths.")
        self.total_examples = 0
        data_set = None
        record_names = []
        for record_path in record_paths:
            assert os.path.isdir(record_path), 'Pass a directory, not a tfrecord\n{}'.format(record_path)
            record_names += [os.path.join(record_path, i) for i in os.listdir(record_path) if i.endswith('.tfrecord')]
        raw_dataset = tf.data.Dataset.list_files(record_names, shuffle=shuffle)
        raw_dataset = raw_dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename, num_parallel_reads=self.in_parallel),
            cycle_length=4,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        raw_dataset = raw_dataset.repeat()
        # raw_dataset = tf.data.TFRecordDataset(record_names, num_parallel_reads=self.in_parallel)
        features = None
        d_types = None
        for record_name in record_names:
            if features is None:
                features = load_obj(record_name.replace('.tfrecord', '_features.pkl'))
            if d_types is None:
                d_types = load_obj(record_name.replace('.tfrecord', '_dtype.pkl'))
            if os.path.exists(record_name.replace('.tfrecord', '_Num_Examples.txt')):
                fid = open(record_name.replace('.tfrecord', '_Num_Examples.txt'))
                examples = fid.readline()
                fid.close()
                self.total_examples += int(examples)
            else:
                self.total_examples += 1
        parsed_image_dataset = raw_dataset.map(return_parse_function(features))
        Decode = DecodeImagesAnnotations(d_type_dict=d_types)
        if debug:
            data = next(iter(parsed_image_dataset))
            data = Decode.parse(image_features=data)
        self.data_set = parsed_image_dataset.map(tf.function(Decode.parse), num_parallel_calls=self.in_parallel)

    def compile_data_set(self, image_processors=None, debug=False):
        data = None
        if debug and data is None:
            data = next(iter(self.data_set))
        if image_processors is not None:
            for image_processor in image_processors:
                print(image_processor)
                if type(image_processor) not in [dict, set]:
                    processor = image_processor.__call__
                    if debug:
                        if data is None:
                            data = next(iter(self.data_set))
                        if type(data) is tuple:
                            data = image_processor.__call__(*data)
                        elif data is not None:
                            data = image_processor.__call__(data)
                    else:
                        processor = tf.function(image_processor.__call__)
                    self.data_set = self.data_set.map(processor, num_parallel_calls=self.in_parallel)
                elif type(image_processor) in [dict, set]:
                    data = None
                    value = None
                    if type(image_processor) is dict:
                        value = [image_processor[i] for i in image_processor][0]
                    if 'batch' in image_processor:
                        assert value is not None, "You need to provide a batch size with {'batch':batch_size}"
                        self.total_examples = self.total_examples//value
                        self.data_set = self.data_set.batch(value, drop_remainder=False)
                    elif 'cache' in image_processor:
                        if value is None:
                            self.data_set = self.data_set.cache()
                        else:
                            assert not os.path.isfile(value), 'Pass a path to {cache:path}, not a file!'
                            if not os.path.exists(value):
                                os.makedirs(value)
                            if self.delete_old_cache:
                                existing_files = glob.glob(os.path.join(value,'*cache.tfrecord*')) # Delete previous ones
                                for file in existing_files:
                                    os.remove(file)
                            path = os.path.join(value,'cache.tfrecord')
                            self.data_set = self.data_set.cache(path)
                    elif 'unbatch' in image_processor:
                        self.data_set = self.data_set.unbatch()
                    elif 'prefetch' in image_processor:
                        self.data_set = self.data_set.prefetch(tf.data.experimental.AUTOTUNE)
                else:
                    raise ModuleNotFoundError('Need to provide either a image processor, dict, or set!')

    def __len__(self):
        return self.total_examples


class Data_Generator_Class(DataGeneratorClass):
    def __init__(self, **kwargs):
        print('Please move from using Data_Generator_Class to DataGeneratorClass, same arguments are passed')
        super().__init__(**kwargs)


if __name__ == '__main__':
    pass
