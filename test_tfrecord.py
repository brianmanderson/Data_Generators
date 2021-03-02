__author__ = 'Brian M Anderson'
# Created on 3/2/2021

import unittest
import os
from Image_Processors_Module.Image_Processors_TFRecord import dictionary_to_tf_record, tf, load_obj
from Image_Processors_Module.Image_Processors_DataSet import DecodeImagesAnnotations
import numpy as np


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
        assert type(record_paths) is list, "Provide a list of record paths"
        self.total_examples = 0
        data_set = None
        for record_path in record_paths:
            assert os.path.isdir(record_path), 'Pass a directory, not a tfrecord\n{}'.format(record_path)
            record_names = [os.path.join(record_path,i) for i in os.listdir(record_path) if i.endswith('.tfrecord')]
            if shuffle:
                perm = np.arange(len(record_names))
                np.random.shuffle(perm)
                record_names = list(np.asarray(record_names)[perm])
            raw_dataset = tf.data.TFRecordDataset(record_names, num_parallel_reads=self.in_parallel)
            features = None
            d_types = None
            for record_name in record_names:
                if features is None:
                    features = load_obj(record_name.replace('.tfrecord', '_features.pkl'))
                if d_types is None:
                    d_types = load_obj(record_name.replace('.tfrecord', '_dtype.pkl'))
                if os.path.exists(record_name.replace('.tfrecord','_Num_Examples.txt')):
                    fid = open(record_name.replace('.tfrecord','_Num_Examples.txt'))
                    examples = fid.readline()
                    fid.close()
                    self.total_examples += int(examples)
            parsed_image_dataset = raw_dataset.map(return_parse_function(features))
            Decode = DecodeImagesAnnotations(d_type_dict=d_types)
            if debug:
                data = next(iter(parsed_image_dataset))
                data = Decode.parse(image_features=data)
            decoded_dataset = parsed_image_dataset.map(Decode.parse, num_parallel_calls=self.in_parallel)
            if data_set is None:
                data_set = decoded_dataset
            else:
                data_set = data_set.concatenate(decoded_dataset)
        self.data_set = data_set


class TFRecordTest(unittest.TestCase):
    def setUp(self):
        self.example = {'int': 5, 'float': 5.0, 'mask': np.zeros([2, 2]).astype('int8'),
                        'image': np.ones([2, 2]).astype('float'), 'string': 'test'}
        dictionary_to_tf_record(filename=os.path.join('.', 'test_record.tfrecord'), input_features={'0': self.example})

    def test_keys_present(self):
        data_generator = DataGeneratorClass(record_paths=['.'])
        x = next(iter(data_generator.data_set))
        for key in self.example.keys():
            self.assertIn(key, x)
        return None

    def tearDown(self):
        for file in os.listdir('.'):
            if file.startswith('test_record'):
                os.remove(os.path.join('.', file))


def return_suite():
    suite = unittest.TestSuite()
    suite.addTest(TFRecordTest('test_keys_present'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(return_suite())
