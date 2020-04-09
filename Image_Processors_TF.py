__author__ = 'Brian M Anderson'
# Created on 4/8/2020
import tensorflow as tf
import numpy as np
from .Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt


class Image_Processor(object):
    def parse(self, *args):
        return args


class Decode_Images_Annotations(Image_Processor):
    def pre_cache_processes(self, image_features):
        if 'z_images' in image_features:
            tensor_image = tf.reshape(tf.io.decode_raw(image_features['image'], out_type='float'),
                                      (image_features['z_images'], image_features['rows'],
                                       image_features['cols']))
            annotation_image = tf.reshape(tf.io.decode_raw(image_features['annotation'], out_type='int8'),
                                          (image_features['z_images'], image_features['rows'],
                                           image_features['cols']))
        else:
            tensor_image = tf.reshape(tf.io.decode_raw(image_features['image'], out_type='float'),
                                      (image_features['rows'], image_features['cols']))
            annotation_image = tf.reshape(tf.io.decode_raw(image_features['annotation'], out_type='int8'),
                                          (image_features['rows'], image_features['cols']))
        image_features['image'] = tensor_image
        image_features['annotation'] = annotation_image
        return image_features


class Decode_Bounding_Boxes_Volumes_Spacing(Image_Processor):
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

    def parse(self, image_features):
        if 'spacing' in image_features:
            spacing = tf.io.decode_raw(image_features['spacing'], out_type='float')
            image_features['spacing'] = spacing
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
    def parse(self, image_features):
        return image_features['image'], image_features['annotation']


class Expand_Dimensions(Image_Processor):
    def __init__(self, axis=-1, on_images=True, on_annotations=False):
        self.axis = axis
        self.on_images = on_images
        self.on_annotations = on_annotations

    def parse(self, image_features):
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

    def parse(self, image_features):
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

    def parse(self, image_features):
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

    def parse(self, image_features):
        image = image_features['image']
        image = tf.where(image > self.upper, self.upper, image)
        image = tf.where(image < self.lower, self.lower, image)
        image_features['image'] = image
        return image_features


class Clip_Images(Image_Processor):
    def __init__(self, annotations_index=None, bounding_box_expansion=(10, 10, 10), power_val_z=1, power_val_r=1,
                 power_val_c=1, min_images=0, min_rows=0, min_cols=0):
        self.annotations_index = annotations_index
        self.bounding_box_expansion = tf.convert_to_tensor(bounding_box_expansion)
        self.power_val_z, self.power_val_r, self.power_val_c = tf.constant(power_val_z), tf.constant(power_val_r), tf.constant(power_val_c)
        self.min_images, self.min_rows, self.min_cols = tf.constant(min_images), tf.constant(min_rows), tf.constant(min_cols)

    def parse(self, image_features):
        zero = tf.constant(0)
        image = image_features['image']
        annotation = image_features['annotation']
        img_shape = tf.shape(image)
        if self.annotations_index:
            bounding_box = image_features['bounding_boxes_{}'.format(self.annotations_index)][0] # Assuming one bounding box
            c_start, r_start, z_start = bounding_box[0], bounding_box[1], bounding_box[2]
            c_stop, r_stop, z_stop = c_start + bounding_box[3], r_start + bounding_box[4], z_start + bounding_box[5]
            z_start = tf.maximum(zero, z_start - self.bounding_box_expansion[0])
            z_stop = tf.minimum(z_stop + self.bounding_box_expansion[0], img_shape[0])
            r_start = tf.maximum(zero, r_start - self.bounding_box_expansion[1])
            r_stop = tf.minimum(img_shape[1], r_stop + self.bounding_box_expansion[1])
            c_start = tf.maximum(zero, c_start - self.bounding_box_expansion[2])
            c_stop = tf.minimum(img_shape[2], c_stop + self.bounding_box_expansion[2])
        else:
            z_stop, r_stop, c_stop = img_shape[:3]
            z_start, r_start, c_start = 0, 0, 0
        z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start

        remainder_z = tf.math.floormod(self.power_val_z - z_total,self.power_val_z) if tf.math.floormod(z_total,
                                                                                                        self.power_val_z) != zero else zero
        remainder_r = tf.math.floormod(self.power_val_r - r_total, self.power_val_r) if tf.math.floormod(r_total,
                                                                                                         self.power_val_r) != zero else zero
        remainder_c = tf.math.floormod(self.power_val_c - r_total, self.power_val_c) if tf.math.floormod(r_total,
                                                                                                         self.power_val_c) != zero else zero
        min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r, c_total + remainder_c
        min_images = tf.maximum(self.min_images,min_images)
        min_rows = tf.maximum(self.min_rows, min_rows)
        min_cols = tf.maximum(self.min_cols, min_cols)
        output_dims = tf.convert_to_tensor([min_images, min_rows, min_cols])
        image_cube = image[z_start:z_start + min_images, r_start:r_start + min_rows, c_start:c_start + min_cols,...]
        annotation_cube = annotation[z_start:z_start + min_images, r_start:r_start + min_rows, c_start:c_start + min_cols]
        img_shape = tf.shape(image_cube)
        size_dif = output_dims - img_shape[:3]
        if tf.reduce_max(size_dif) > 0:
            paddings = tf.convert_to_tensor([[size_dif[0], zero], [size_dif[1], zero], [size_dif[2], zero], [zero, zero]])
            image_cube = tf.pad(image_cube,paddings=paddings,constant_values=tf.reduce_min(image))
            annotation_cube = tf.pad(annotation_cube, paddings=paddings)
        image_features['image'] = image_cube
        image_features['annotation'] = annotation_cube
        return image_features


class Pull_Cube(Image_Processor):
    def __init__(self, annotation_index=None, max_cubes=10, z_images=16, rows=100, cols=100, min_volume=0, min_voxels=0,
                 max_volume=np.inf, max_voxels=np.inf):
        '''
        annotation_index = scalar value referring to annotation value for bbox
        '''
        self.annotation_index = annotation_index
        self.max_cubes = tf.constant(max_cubes)
        self.z_images, self.rows, self.cols = tf.constant(z_images), tf.constant(rows), tf.constant(cols)
        if min_volume != 0 and min_voxels != 0:
            raise AssertionError('Cannot have both min_volume and min_voxels specified')
        if max_volume != np.inf and min_voxels != np.inf:
            raise AssertionError('Cannot have both max_volume and max_voxels specified')
        self.min_volume = tf.constant(min_volume * 1000, dtype='float')
        self.min_voxels = tf.constant(min_voxels, dtype='float')
        self.max_volume, self.max_voxels = tf.constant(max_volume * 1000, dtype='float'), tf.constant(max_voxels, dtype='float')

    def parse(self, image_features):
        if self.annotation_index is not None:
            bounding_boxes = image_features['bounding_boxes_{}'.format(self.annotation_index)]
            volumes = image_features['volumes_{}'.format(self.annotation_index)]


if __name__ == '__main__':
    pass
