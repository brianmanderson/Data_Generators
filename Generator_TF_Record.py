__author__ = 'Brian M Anderson'
# Created on 4/7/2020

from .Image_Processors_TF import *
from .Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt
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

    image_processors = {'pre_cache_map':[Decode_Images_Annotations(), Decode_Bounding_Boxes_Volumes_Spacing(),
                                         Return_Images_Annotations()],
                        'pre_cache_batch':[Expand_Dimensions(axis=-1, on_images=True, on_annotations=True),
                                           Repeat_Channel(axis=-1, repeats=3, on_images=True),
                                           Normalize_Images(mean_val=80, std_val=42),
                                           Threshold_Images(lower_bound=-5, upper_bound=5)],
                        'post_cache_map':[],
                        'post_cache_batch':[]}
class Data_Generator_Class(object):
    def __init__(self, record_names=None, shuffle=False, batch_size=1, is_2D_whole_patient=False, debug=False,
                 image_processors=None):
        self.shuffle = shuffle
        assert record_names is not None, print('Need to pass a list of record names!')
        data_sets = []
        for record_name in record_names:
            raw_dataset = tf.data.TFRecordDataset([record_name], num_parallel_reads=tf.data.experimental.AUTOTUNE)
            features = load_obj(record_name.replace('.tfrecord', '_features.pkl'))
            parsed_image_dataset = raw_dataset.map(return_parse_function(features))
            data_sets.append(parsed_image_dataset)
        if len(data_sets) > 1:
            data_sets = tuple(data_sets)
            data_set = tf.data.Dataset.zip(tuple(data_sets))
        else:
            data_set = data_sets[0]
        self.data_set = data_set
        data = None
        if debug:
            data = next(iter(data_set))
        if image_processors is not None:
            for image_processor in image_processors:
                if image_processor == 'batch':
                    data_set = data_set.batch(batch_size, drop_remainder=False)
                elif image_processor == 'cache':
                    data_set = data_set.cache()
                elif image_processor == 'unbatch':
                    data_set = data_set.unbatch()
                else:
                    data_set = data_set.map(image_processor.parse,
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
                    if debug:
                        data = image_processor.parse(data)
        self.data_set = data_set


if __name__ == '__main__':
    pass
