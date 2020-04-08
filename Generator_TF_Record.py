__author__ = 'Brian M Anderson'
# Created on 4/7/2020

from .Image_Processors_TF import *
import pickle, os


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


class Data_Generator_Class(object):
    def __init__(self, record_names=None, shuffle=False, image_processors=None):
        self.shuffle = shuffle
        assert record_names is not None, print('Need to pass a list of record names!')
        data_sets = []
        for record_name in record_names:
            raw_dataset = tf.data.TFRecordDataset([record_name])
            features = load_obj(record_name.replace('.tfrecord', '_features.pkl'))
            parsed_image_dataset = raw_dataset.map(return_parse_function(features))
            data_sets.append(parsed_image_dataset)
        if len(data_sets) > 1:
            data_sets = tuple(data_sets)
            data_set = tf.data.Dataset.zip(tuple(data_sets))
        else:
            data_set = data_sets[0]
        self.data_set = data_set
        if image_processors is not None:
            for image_processor in image_processors:
                data_set = data_set.map(image_processor.parsing_function)
        data = next(iter(data_set))
        xxx = 1

    def __on_epoch_end__(self):
        if self.shuffle:
            self.data_set.shuffle()


if __name__ == '__main__':
    pass
