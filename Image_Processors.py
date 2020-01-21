import numpy as np
from scipy.ndimage import interpolation, filters
from keras.utils import np_utils
import cv2, math, copy, os, sys
from skimage.measure import block_reduce
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  #Add path to module
from Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt
from Fill_Missing_Segments.Fill_In_Segments_sitk import Fill_Missing_Segments

'''
Description of code
Ensure_Image_Proportions(image_size-row, image_size_col): ensures images are proper proportions
Normalize_Images(mean_val, std_val): Normalize images
Threshold_Images(lower_bound, upper_bound): threshold, normally after normalizing, recommended -3.55 to +3.55
Add_Noise_To_Images(by_patient, variation): add noise to patient images
Random_Horizontal_Vertical_Flips(by_patient, h_flip, v_flip): randomly flip 2D images vertically/horizontally
Random_Scale_Processor(by_patient, variation): randomly scale images by certain amounts
Rotate_Images_2D_Processor(by_patient, image_size, variation): Randomly rotate images in 2D plane
Shift_Images_Processor(by_patient, variation, positive_to_negative): Shift images by a certain variation in 2D plane
Random_2D_Deformation_Processor(by_patient, image_size, variation): Randomly deform images in 2D plane
'''
class Image_Processor(object):
    def preload_single_image_process(self, image, annotation):
        '''
        This is for image processes done loading on a slice by slice basis that only need to be done once, like
        normalizing a CT slice with a mean and std
        :param image: Some image of shape [n_row, m_col]
        :param annotation: Some image of shape [n_row, m_col]
        :return:
        '''
        return image, annotation

    def post_load_process(self, images, annotations):
        '''
        :param images: Images set to values of 0 to max - min. This is done
        :param annotations:
        :return:
        '''
        return images, annotations


class Fuzzy_Segment_Liver_Lobes(Image_Processor):
    def __init__(self, variation=None, spacing=(1,1,5)):
        '''
        :param variation: margin to expand region, mm. np.arange(start=0, stop=1, step=1)
        :param spacing: Spacing of images, assumes this is constance
        '''
        self.variation = variation
        self.spacing = spacing
        self.Fill_Missing_Segments_Class = Fill_Missing_Segments()

    def make_fuzzy_label(self, annotation, variation):
        distance_map = np.zeros(annotation.shape)
        for i in range(1, annotation.shape[-1]):
            temp_annotation = annotation[..., i].astype('int')
            distance_map[..., i] = self.Fill_Missing_Segments_Class.run_distance_map(temp_annotation,
                                                                                     spacing=self.spacing)
        distance_map[distance_map > 0] = 0
        distance_map = np.abs(distance_map)
        distance_map[distance_map > variation] = variation  # Anything greater than 10 mm away set to 0
        distance_map = 1 - distance_map / variation
        distance_map[annotation[..., 0] == 1] = 0
        distance_map[..., 0] = annotation[..., 0]
        total = np.sum(distance_map, axis=-1)
        distance_map /= total[..., None]
        return distance_map

    def post_load_process(self, images, annotations):
        '''
        :param images: Images set to values of 0 to max - min. This is done
        :param annotations:
        :return:
        '''
        if self.variation is not None:
            variation = self.variation[np.random.randint(len(self.variation))]
            annotations = self.make_fuzzy_label(annotations, variation)
        return images, annotations


class Annotations_To_Categorical(Image_Processor):
    def __init__(self, num_of_classes=2):
        '''
        :param num_of_classes: number of classes
        '''
        self.num_of_classes = num_of_classes

    def preload_single_image_process(self, images, annotations):
        '''
        :param images: Images set to values of 0 to max - min. This is done
        :param annotations:
        :return:
        '''
        annotations = np_utils.to_categorical(annotations, self.num_of_classes)
        return images, annotations


class Ensure_Image_Proportions(Image_Processor):
    def __init__(self, image_size_row=512, image_size_col=512):
        self.image_size_row, self.image_size_col = image_size_row, image_size_col

    def convert_image_size(self, images, annotations):
        dif_1 = (self.image_size_row - images.shape[1])
        dif_2 = (self.image_size_col - images.shape[2])
        if dif_1 > 0 and dif_2 > 0:
            out_image = np.ones([1,self.image_size_row, self.image_size_col],dtype=images.dtype) * np.min(images)
            out_annotations = np.zeros([1, self.image_size_row, self.image_size_col],dtype=annotations.dtype)
            out_image[:, dif_1//2:dif_1//2 + images.shape[1], dif_2//2:dif_2//2 + images.shape[2],...] = images
            out_annotations[:, dif_1//2:dif_1//2 + images.shape[1], dif_2//2:dif_2//2 + images.shape[2],...] = annotations
            return out_image, out_annotations
        if dif_1 != 0:
            if dif_1 > 0:
                images = np.concatenate((images, images[:, :dif_1//2, ...]),axis=1)
                images = np.concatenate((images[:, -dif_1//2:, ...], images),axis=1)
                annotations = np.concatenate((annotations, annotations[:, :dif_1//2, ...]),axis=1)
                annotations = np.concatenate((annotations[:, -dif_1//2:, ...], annotations),axis=1)
            elif dif_1 < 0:
                images = images[:, abs(dif_1)//2:-abs(dif_1//2), ...]
                annotations = annotations[:, abs(dif_1)//2:-abs(dif_1//2), ...]
        if dif_2 != 0:
            if dif_2 > 0:
                images = np.concatenate((images, images[:, :, :dif_2//2, ...]),axis=2)
                images = np.concatenate((images[:, :, -dif_2//2:, ...], images),axis=2)
                annotations = np.concatenate((annotations, annotations[:, :, :dif_2//2, ...]),axis=2)
                annotations = np.concatenate((annotations[:, :, -dif_2//2:, ...], annotations),axis=2)
            elif dif_2 < 0:
                images = images[:, :, abs(dif_2)//2:-abs(dif_2//2), ...]
                annotations = annotations[:, :, abs(dif_2)//2:-abs(dif_2//2), ...]
        return images, annotations

    def preload_single_image_process(self, image, annotation):
        if image.shape[0] != 1:
            image = image[None, ...]
            annotation = annotation[None, ...]
        if image.shape[1] != self.image_size_row or image.shape[2] != self.image_size_col:
            if image.shape[1] >= self.image_size_row * 2 and image.shape[2] >= self.image_size_col * 2:
                if len(annotation.shape) == 3:
                    block = (2, 2)
                else:
                    block = (2, 2, 1)
                image = block_reduce(image[0, ...], block, np.average).astype('float32')[None, ...]
                annotation = block_reduce(annotation[0, ...].astype('int'), block, np.max).astype('int')[
                    None, ...]
            image, annotation = self.convert_image_size(image, annotation)
        return image, annotation


class Normalize_Images(Image_Processor):
    def __init__(self, mean_val=0, std_val=1):
        '''
        :param mean_val: Mean value to normalize to
        :param std_val: Standard deviation value to normalize to
        '''
        self.mean_val, self.std_val = mean_val, std_val

    def preload_single_image_process(self, images, annotations):
        images = (images - self.mean_val)/self.std_val
        return images, annotations


class Threshold_Images(Image_Processor):
    def __init__(self, lower_bound=-np.inf, upper_bound=np.inf, inverse_image=False):
        '''
        :param lower_bound: Lower bound to threshold images, normally -3.55 if Normalize_Images is used previously
        :param upper_bound: Upper bound to threshold images, normally 3.55 if Normalize_Images is used previously
        :param inverse_image: Should the image be inversed after threshold?
        '''
        self.lower = lower_bound
        self.upper = upper_bound
        self.inverse_image = inverse_image

    def post_load_process(self, images, annotations):
        images[images<self.lower] = self.lower
        images[images>self.upper] = self.upper
        if self.inverse_image:
            if self.upper != np.inf and self.lower != -np.inf:
                images = (self.upper + self.lower) - images
            else:
                images = -1*images
        return images, annotations


class Add_Noise_To_Images(Image_Processor):
    def __init__(self, by_patient=False, variation=None):
        '''
        :param by_patient:
        :param variation: range of values np.round(np.arange(start=0, stop=1.0, step=0.1),2)
        '''
        self.by_patient = by_patient
        self.variation = variation

    def post_load_process(self, images, annotations):
        if self.variation is not None:
            if self.by_patient:
                variation = self.variation[np.random.randint(len(self.variation))]
                noisy_image = variation * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
                images += noisy_image
            else:
                for i in range(images.shape[0]):
                    variation = self.variation[np.random.randint(len(self.variation))]
                    noisy_image = variation * np.random.normal(loc=0.0, scale=1.0, size=images[i].shape)
                    images[i] += noisy_image
        return images, annotations


class Random_Horizontal_Vertical_Flips(Image_Processor):
    def __init__(self, by_patient=False, h_flip=False, v_flip=False):
        '''
        :param by_patient: Process all images as single patient (flip together)
        :param h_flip: Perform horizontal flips?
        :param v_flip: Perform vertical flips?
        '''
        self.by_patient, self.h_flip, self.v_flip = by_patient, h_flip, v_flip

    def post_load_process(self, images, annotations):
        if self.h_flip or self.v_flip:
            if self.by_patient:
                if self.h_flip and np.random.randint(2) == 1:
                    images = images[..., :, ::-1]
                    annotations = annotations[..., :, ::-1]
                if self.v_flip and np.random.randint(2) == 1:
                    images = images[..., ::-1, :]
                    annotations = annotations[..., ::-1, :]
            else:
                for i in range(images.shape[0]):
                    if self.h_flip and np.random.randint(2) == 1:
                        images[i] = images[i, :, ::-1]
                        annotations[i] = annotations[i, :, ::-1]
                    if self.v_flip and np.random.randint(2) == 1:
                        images[i] = images[i, ::-1, :]
                        annotations[i] = annotations[i, ::-1, :]
        return images, annotations


class Random_Scale_Processor(Image_Processor):
    def __init__(self, by_patient=False, variation=None):
        '''
        :param by_patient: perform randomly on each image in stack, or on the entire collective
        :param variation: range of values by which to scale the image, np.round(np.arange(start=0, stop=2.6, step=0.5),2)
        '''
        self.by_patient = by_patient
        self.variation = variation

    def scale_image(self, im, variation=0, interpolator='linear'):
        if interpolator is 'linear':
            temp_scale = cv2.resize(im, None, fx=1 + variation, fy=1 + variation,
                                    interpolation=cv2.INTER_LINEAR)
        elif interpolator is 'nearest':
            temp_scale = cv2.resize(im, None, fx=1 + variation, fy=1 + variation,
                                    interpolation=cv2.INTER_NEAREST)
        else:
            return im

        center = (temp_scale.shape[0] // 2, temp_scale.shape[1] // 2)
        if variation > 0:
            im = temp_scale[int(center[0] - 512 / 2):int(center[0] + 512 / 2),
                 int(center[1] - 512 / 2):int(center[1] + 512 / 2)]
        elif variation < 0:
            padx = (512 - temp_scale.shape[0]) / 2
            pady = (512 - temp_scale.shape[1]) / 2
            im = np.pad(temp_scale, [
                (math.floor(padx), math.ceil(padx)),
                (math.floor(pady), math.ceil(pady))], mode='constant',
                        constant_values=np.min(temp_scale))
        return im

    def post_load_process(self, images, annotations):
        if self.variation is not None:
            min_val = np.min(images)
            images -= min_val
            if self.by_patient:
                variation = self.variation[np.random.randint(len(self.variation))]
                for i in range(images.shape[0]):
                    images[i], annotations[i] = self.run_perturbation(images[i],annotations[i],variation)
            else:
                for i in range(images.shape[0]):
                    variation = self.variation[np.random.randint(len(self.variation))]
                    images[i], annotations[i] = self.run_perturbation(images[i],annotations[i],variation)
            images += min_val
        return images, annotations

    def run_perturbation(self, images, annotations, variation):
        images = self.scale_image(images, variation, 'linear')
        output_annotation = np.zeros(annotations.shape, dtype=annotations.dtype)
        for val in range(1, int(annotations.max()) + 1):
            temp = copy.deepcopy(annotations).astype('int')
            temp[temp != val] = 0
            temp[temp > 0] = 1
            im = temp
            if np.max(im) != 0:
                im = self.scale_image(im, variation, 'nearest')
                im[im > 0.1] = val
                im[im < val] = 0
                output_annotation[im == val] = val
        annotations = output_annotation
        return images, annotations


class Rotate_Images_2D_Processor(Image_Processor):
    def __init__(self, by_patient=False, image_size=512, variation=None):
        '''
        :param image_size: size of image row/col
        :param by_patient: perform on all images in stack, or vary for each one
        :param variation: range of values np.round(np.arange(start=-5, stop=5, step=1))
        '''
        self.image_size = image_size
        self.variation = variation
        self.by_patient = by_patient
        self.M_image = {}

    def post_load_process(self, images, annotations):
        if self.variation is not None:
            min_val = np.min(images)
            images -= min_val
            if self.by_patient:
                variation = self.variation[np.random.randint(len(self.variation))]
                for i in range(images.shape[0]):
                    images[i], annotations[i] = self.run_perturbation(images[i],annotations[i],variation)
            else:
                for i in range(images.shape[0]):
                    variation = self.variation[np.random.randint(len(self.variation))]
                    images[i], annotations[i] = self.run_perturbation(images[i],annotations[i],variation)
            images += min_val
        return images, annotations

    def run_perturbation(self, image, annotation, variation):
        image_row, image_col = image.shape[-2:]
        if variation not in self.M_image.keys():
            M_image = cv2.getRotationMatrix2D((int(image_row) / 2, int(image_col) / 2), variation, 1)
            self.M_image[variation] = M_image
        else:
            M_image = self.M_image[variation]
        if variation != 0:
            # images = cv2.warpAffine(images,M_image, (int(shape_size_image), int(shape_size_image)))
            image = cv2.warpAffine(image, M_image, (int(image_row), int(image_col)),flags=cv2.INTER_LINEAR)
            output_annotation = np.zeros(annotation.shape, dtype=annotation.dtype)
            for val in range(1, int(annotation.max()) + 1):
                temp = copy.deepcopy(annotation).astype('int')
                temp[temp != val] = 0
                temp[temp > 0] = 1
                im = temp
                if np.max(im) != 0:
                    im = cv2.warpAffine(im, M_image,
                                        (int(image_row), int(image_col)),
                                        flags=cv2.INTER_NEAREST)
                    im[im > 0.1] = val
                    im[im < val] = 0
                    output_annotation[im == val] = val
                # output_annotation[annotations == val] = val
            annotation = output_annotation
        return image, annotation


class Shift_Images_Processor(Image_Processor):
    def __init__(self,by_patient=False, variation=0, positive_to_negative=False):
        '''
        :param by_patient: Perform the same scaling across all images in stack, or one by one
        :param variation: Range of shift variations in rows and columns, up to and including! So 5 is 0,1,2,3,4,5
        :param positive_to_negative: if True and variation is 30, will range from -30 to +30
        '''
        self.by_patient = by_patient
        if variation != 0:
            if positive_to_negative:
                self.variation_range = np.asarray(range(-variation,variation+1))
            else:
                self.variation_range = np.asarray(range(variation+1))
        else:
            self.variation_range = None

    def post_load_process(self, images, annotations):
        if self.variation_range is not None:
            min_val = np.min(images)
            images -= min_val
            if not self.by_patient:
                for i in range(len(images)):
                    variation_row = self.variation_range[np.random.randint(len(self.variation_range))]
                    variation_col = self.variation_range[np.random.randint(len(self.variation_range))]
                    images[i, :, :], annotations[i, :, :] = self.make_perturbation(images[i, :, :],annotations[i, :, :],
                                                                                   variation_row, variation_col)
            else:
                variation_row = self.variation_range[np.random.randint(len(self.variation_range))]
                variation_col = self.variation_range[np.random.randint(len(self.variation_range))]
                images, annotations = self.make_perturbation(images, annotations, variation_row, variation_col)
            images += min_val
        return images, annotations

    def make_perturbation(self, images, annotations, variation_row, variation_col):
        if len(images.shape) == 2:
            images = interpolation.shift(images, [variation_row, variation_col])
            annotations = interpolation.shift(annotations.astype('int'),
                                              [variation_row, variation_col])
        elif len(images.shape) == 3:
            images = interpolation.shift(images, [0, variation_row, variation_col])
            annotations = interpolation.shift(annotations.astype('int'),
                                              [0, variation_row, variation_col])
        return images, annotations


class Random_2D_Deformation_Processor(Image_Processor):
    def __init__(self, by_patient=False, image_size=512, variation=None):
        '''
        :param image_shape: shape of images row/col
        :param by_patient: perform randomly on each image in stack, or on the entire collective
        :param variation: range of values np.round(np.arange(start=0, stop=2.6, step=0.5),2)
        '''
        self.image_size = image_size
        self.by_patient = by_patient
        self.variation = variation

    def run_perturbation(self, images, annotations, variation):
        # generate random parameter --- will be the same for all slices of the same patients
        # for 3D use dz with same pattern than dx/dy
        random_state = np.random.RandomState(None)

        if len(images.shape) > 2:
            temp_img = images
        else:
            temp_img = images[:, :, None]

        shape = temp_img.shape
        sigma = self.image_size * 0.1
        alpha = self.image_size * variation
        dx = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        # dz = np.zeros_like(dx) #2d not used
        # dz = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), 512*0.10, mode="constant", cval=0) * 512*variation

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        # indices_3d = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

        if len(images.shape) > 2:
            images = interpolation.map_coordinates(temp_img, indices, order=1, mode='constant',
                                                   cval=float(np.min(images))).reshape(shape)
        else:
            images = interpolation.map_coordinates(temp_img, indices, order=1, mode='constant',
                                                   cval=float(np.min(images))).reshape(shape)[:, :, 0]

        output_annotation = np.zeros(annotations.shape, dtype=annotations.dtype)

        for val in range(1, int(annotations.max()) + 1):
            temp = copy.deepcopy(annotations).astype('int')
            temp[temp != val] = 0
            temp[temp > 0] = 1

            if len(annotations.shape) > 2:
                im = interpolation.map_coordinates(temp, indices, order=0, mode='constant', cval=0).reshape(shape)
            else:
                im = interpolation.map_coordinates(temp[:, :, None], indices, order=0, mode='constant',
                                                   cval=0).reshape(
                    shape)[:, :, 0]

            im[im > 0.1] = val
            im[im < val] = 0
            output_annotation[im == val] = val
        annotations = output_annotation
        return images, annotations

    def post_load_process(self, images, annotations):
        if self.variation is not None:
            min_val = np.min(images)
            images -= min_val
            if self.by_patient:
                variation = self.variation[np.random.randint(len(self.variation))]
                for i in range(images.shape[0]):
                    images[i], annotations[i] = self.run_perturbation(images[i],annotations[i],variation)
            else:
                for i in range(images.shape[0]):
                    variation = self.variation[np.random.randint(len(self.variation))]
                    images[i], annotations[i] = self.run_perturbation(images[i],annotations[i],variation)
            images += min_val
        return images, annotations


if __name__ == '__main__':
    pass
