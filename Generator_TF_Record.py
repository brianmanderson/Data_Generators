__author__ = 'Brian M Anderson'
# Created on 4/7/2020

from .Image_Processors.Image_Processors_DataSet import *
from .Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt
import pickle, os, glob


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
    def __init__(self, record_names=None):
        assert record_names is not None, 'Need to pass a list of record names!'
        data_sets = []
        self.total_examples = 0
        for record_name in record_names:
            raw_dataset = tf.data.TFRecordDataset([record_name], num_parallel_reads=tf.data.experimental.AUTOTUNE)
            features = load_obj(record_name.replace('.tfrecord', '_features.pkl'))
            if os.path.exists(record_name.replace('.tfrecord','_Num_Examples.txt')):
                fid = open(record_name.replace('.tfrecord','_Num_Examples.txt'))
                examples = fid.readline()
                fid.close()
                self.total_examples += int(examples)
            parsed_image_dataset = raw_dataset.map(return_parse_function(features))
            data_sets.append(parsed_image_dataset)
        if len(data_sets) > 1:
            data_sets = tuple(data_sets)
            data_set = tf.data.Dataset.zip(tuple(data_sets))
        else:
            data_set = data_sets[0]
        self.data_set = data_set

    def compile_data_set(self, image_processors=None, debug=False):
        data_set = self.data_set
        data = None
        if debug and data is None:
            data = next(iter(data_set))
        if image_processors is not None:
            for image_processor in image_processors:
                print(image_processor)
                if type(image_processor) not in [dict, set]:
                    if debug:
                        if data is None:
                            data = next(iter(data_set))
                        if type(data) is tuple:
                            data = image_processor.parse(*data)
                        elif data is not None:
                            data = image_processor.parse(data)
                    data_set = data_set.map(image_processor.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                elif type(image_processor) in [dict, set]:
                    data = None
                    value = None
                    if type(image_processor) is dict:
                        value = [image_processor[i] for i in image_processor][0]
                    if 'batch' in image_processor:
                        assert value is not None, "You need to provide a batch size with {'batch':batch_size}"
                        self.total_examples = self.total_examples//value
                        data_set = data_set.batch(value, drop_remainder=False)
                    elif 'shuffle' in image_processor:
                        assert value is not None, "You need to provide a shuffle_buffer with {'shuffle':buffer}"
                        data_set = data_set.shuffle(value, reshuffle_each_iteration=True)
                    elif 'cache' in image_processor:
                        if value is None:
                            data_set = data_set.cache()
                        else:
                            assert os.path.isdir(value), 'Pass a path to {cache:path}, not a file!'
                            existing_files = glob.glob(os.path.join(value,'*cache.tfrecord*')) # Delete previous ones
                            for file in existing_files:
                                os.remove(file)
                            path = os.path.join(value,'cache.tfrecord')
                            data_set = data_set.cache(path)
                    elif 'repeat' in image_processor:
                        if value is not None:
                            data_set = data_set.repeat(value)
                        else:
                            data_set = data_set.repeat()
                    elif 'unbatch' in image_processor:
                        data_set = data_set.unbatch()
                else:
                    raise ModuleNotFoundError('Need to provide either a image processor, dict, or set!')
        self.data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)

    def __len__(self):
        return self.total_examples



if __name__ == '__main__':
    pass
