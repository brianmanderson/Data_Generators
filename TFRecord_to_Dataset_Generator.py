__author__ = 'Brian M Anderson'
# Created on 4/7/2020

from .Image_Processors_Module.Image_Processors_DataSet import *
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
    def __init__(self, record_paths=None, in_parallel=True, delete_old_cache=False):
        '''
        :param record_paths: List of paths to a folder full of records files
        :param in_parallel:
        :param delete_old_cache:
        '''
        self.delete_old_cache = delete_old_cache
        if in_parallel:
            self.in_parallel = tf.data.experimental.AUTOTUNE
        else:
            self.in_parallel = None
        assert record_paths is not None, 'Need to pass a list of record names!'
        self.total_examples = 0
        data_set = None
        for record_path in record_paths:
            assert os.path.isdir(record_path), 'Pass a directory, not a tfrecord'
            record_names = [os.path.join(record_path,i) for i in os.listdir(record_path) if i.endswith('.tfrecord')]
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
            Decode = Decode_Images_Annotations(d_type_dict=d_types)
            decoded_dataset = parsed_image_dataset.map(Decode.parse, num_parallel_calls=self.in_parallel)
            if data_set is None:
                data_set = decoded_dataset
            else:
                data_set = data_set.concatenate(decoded_dataset)
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
                    data_set = data_set.map(image_processor.parse, num_parallel_calls=self.in_parallel)
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
                            assert not os.path.isfile(value), 'Pass a path to {cache:path}, not a file!'
                            if not os.path.exists(value):
                                os.makedirs(value)
                            if self.delete_old_cache:
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
