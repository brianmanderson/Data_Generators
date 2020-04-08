__author__ = 'Brian M Anderson'
# Created on 4/8/2020
import tensorflow as tf
import numpy as np
from .Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt


class Image_Processor(object):
    def parsing_function(self, image_features):
        '''
        Giving in a dictionary of values
        '''
        return image_features


class Decode_Images_Annotations(Image_Processor):
    def parsing_function(self, image_features):
        tensor_image = tf.reshape(tf.io.decode_raw(image_features['image'], out_type='float'),
                                  (image_features['z_images'], image_features['rows'],
                                   image_features['cols']))
        annotation_image = tf.reshape(tf.io.decode_raw(image_features['annotation'], out_type='int8'),
                                      (image_features['z_images'], image_features['rows'],
                                       image_features['cols']))
        image_features['image'] = tensor_image
        image_features['annotation'] = annotation_image
        return image_features


class Decode_Bounding_Boxes_and_Volumes(Image_Processor):
    def __init__(self, annotation_indexes=None):
        '''
        annotation_indexes: list of indexes [1,2,3...]
        '''
        self.bbox_names = []
        self.volume_names = []
        if annotation_indexes is not None:
            for index in annotation_indexes:
                self.volume_names.append('volumes_{}'.format(index))
                self.bbox_names.append('bounding_boxes_{}'.format(index))

    def parsing_function(self, image_features):
        for name in self.bbox_names:
            if name in image_features:
                bboxes = tf.io.decode_raw(image_features[name], out_type='int32')
                bboxes = tf.reshape(bboxes,(len(bboxes)//6,6))
                image_features[name] = bboxes
        for name in self.volume_names:
            if name in image_features:
                volumes = tf.io.decode_raw(image_features[name], out_type='float')
                volumes = tf.reshape(volumes, (1, len(volumes)))
                image_features[name] = volumes
        return image_features


class Return_Images_Annotations(Image_Processor):
    def parsing_function(self, image_features):
        return image_features['image'], image_features['annotation']


class Expand_Dimensions(Image_Processor):
    def __init__(self, axis=-1, on_images=True, on_annotations=False):
        self.axis = axis
        self.on_images = on_images
        self.on_annotations = on_annotations

    def parsing_function(self, image_features):
        if self.on_images:
            image = image_features['image']
            image = tf.expand_dims(image, axis=self.axis)
            image_features['image'] = image
        if self.on_annotations:
            annotation = image_features['annotation']
            annotation = tf.expand_dims(annotation, axis=self.axis)
            image_features['annotation'] = annotation
        return image_features


class Repeat_Channel(Image_Processor):
    def __init__(self, axis=-1, repeats=3, on_images=True, on_annotations=False):
        '''
        :param axis: axis to expand
        :param repeats: number of repeats
        :param on_images: expand the axis on the images
        :param on_annotations: expand the axis on the annotations
        '''
        self.axis = axis
        self.repeats = repeats
        self.on_images = on_images
        self.on_annotations = on_annotations

    def parsing_function(self, image_features):
        if self.on_images:
            image = image_features['image']
            image = tf.repeat(image, axis=self.axis, repeats=self.repeats)
            image_features['image'] = image
        if self.on_annotations:
            annotation = image_features['annotation']
            annotation = tf.repeat(annotation, axis=self.axis, repeats=self.repeats)
            image_features['annotation'] = annotation
        return image_features


class Normalize_Images(Image_Processor):
    def __init__(self, mean_val=0, std_val=1):
        '''
        :param mean_val: Mean value to normalize to
        :param std_val: Standard deviation value to normalize to
        '''
        self.mean_val, self.std_val = tf.constant(mean_val, dtype='float32'), tf.constant(std_val, dtype='float32')

    def parsing_function(self, image_features):
        image = image_features['image']
        image = (image - self.mean_val)/self.std_val
        image_features['image'] = image
        return image_features


class Threshold_Images(Image_Processor):
    def __init__(self, lower_bound=-np.inf, upper_bound=np.inf):
        '''
        :param lower_bound: Lower bound to threshold images, normally -3.55 if Normalize_Images is used previously
        :param upper_bound: Upper bound to threshold images, normally 3.55 if Normalize_Images is used previously
        :param inverse_image: Should the image be inversed after threshold?
        :param post_load: should this be done each iteration? If False, gets slotted under pre_load_process
        :param final_scale_value: Value to scale the entire image to (255 scales to 0-255), (1 scales to 0-1)
        '''
        self.lower = tf.constant(lower_bound, dtype='float32')
        self.upper = tf.constant(upper_bound, dtype='float32')

    def parsing_function(self, image_features):
        image = image_features['image']
        image = tf.where(image > self.upper, self.upper, image)
        image = tf.where(image < self.lower, self.lower, image)
        image_features['image'] = image
        return image_features


if __name__ == '__main__':
    pass
