__author__ = 'Brian M Anderson'
# Created on 4/8/2020
import tensorflow as tf


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


if __name__ == '__main__':
    pass
