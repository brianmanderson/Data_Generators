import numpy as np
from scipy.ndimage import interpolation, filters
import cv2, math, copy
from Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt


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
    def __init__(self, lower_bound=-np.inf, upper_bound=np.inf):
        '''
        :param lower_bound: Lower bound to threshold images, normally -3.55 if Normalize_Images is used previously
        :param upper_bound: Upper bound to threshold images, normally 3.55 if Normalize_Images is used previously
        '''
        self.lower = lower_bound
        self.upper = upper_bound

    def post_load_process(self, images, annotations):
        images[images<self.lower] = self.lower
        images[images>self.upper] = self.upper
        return images, annotations


class Add_Noise_To_Images(Image_Processor):
    def __init__(self, noise=0.0):
        self.noise = noise

    def post_load_process(self, images, annotations):
        if self.noise != 0.0:
            noisy_image = self.noise * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
            images += noisy_image
        return images, annotations


class Perturbation_Class(Image_Processor):
    def __init__(self,pertubartions,image_shape, by_patient):
        '''
        :param pertubartions: Dictionary of keys for perturbations, examples are 'Shift', 'Rotation', '2D_Random'
        :param image_shape:
        :param by_patient:
        '''
        self.by_patient = by_patient
        self.pertubartions = pertubartions
        self.output_annotation_template = np.zeros(image_shape)
        self.output_images_template = np.zeros(image_shape)
        self.M_image = {}
        self.image_shape = image_shape

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
        if not self.by_patient:
            for i in range(len(images)):
                images[i, :, :], annotations[i, :, :] = self.make_pertubartions(images[i, :, :],annotations[i, :, :])
        else:
            images, annotations = self.make_pertubartions(images, annotations)
        return images, annotations

    def make_pertubartions(self,images,annotations):
        min_val = np.min(images)
        images -= min_val # This way any rotation gets a 0, irrespective of previous normalization
        for key in self.pertubartions.keys():
            variation = self.pertubartions[key][np.random.randint(0, len(self.pertubartions[key]))]
            if key == 'Scale':    # 'Scale': np.round(np.arange(start=-0.15, stop=0.20, step=0.05),2)
                if variation != 0:
                    output_image = np.zeros(images.shape, dtype=images.dtype)
                    if len(images.shape) > 2:
                        for image in range(images.shape[0]):
                            im = images[image, :, :]
                            if np.max(im) != 0:
                                im = self.scale_image(im, variation, 'linear')
                            output_image[image, :, :] = im
                    else:
                        output_image = self.scale_image(images, variation, 'linear')

                    images = output_image
                    output_annotation = np.zeros(annotations.shape, dtype=annotations.dtype)

                    for val in range(1, int(annotations.max()) + 1):
                        temp = copy.deepcopy(annotations).astype('int')
                        temp[temp != val] = 0
                        temp[temp > 0] = 1
                        if len(annotations.shape) > 2:
                            for image in range(annotations.shape[0]):
                                im = temp[image, :, :]
                                if np.max(im) != 0:
                                    im = self.scale_image(im, variation, 'nearest')

                                    im[im > 0.1] = val
                                    im[im < val] = 0
                                    output_annotation[image, :, :][im == val] = val
                        else:
                            im = temp
                            if np.max(im) != 0:
                                im = self.scale_image(im, variation, 'nearest')

                                im[im > 0.1] = val
                                im[im < val] = 0
                                output_annotation[im == val] = val
                    annotations = output_annotation
            elif key is 'h_flip':   # 'h_flip': [0, 1]
                if variation != 0:
                    images = images[:, ::-1]
                    annotations = annotations[:, ::-1]


        images += min_val
        output_image = images
        output_annotation = annotations
        return output_image, output_annotation


class Rotate_Images_2D_Processor(Image_Processor):
    def __init__(self, image_size=512, by_patient=False, variation=None):
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
    def __init__(self,by_patient=False, variation=0, positive_negative=False):
        '''
        :param by_patient: Perform the same scaling across all images in stack, or one by one
        :param variation: Range of shift variations in rows and columns, up to and including! So 5 is 0,1,2,3,4,5
        :param positive_negative: if True and variation is 30, will range from -30 to +30
        '''
        self.by_patient = by_patient
        if variation != 0:
            if positive_negative:
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
    def __init__(self, image_size=512, by_patient=False, variation=None):
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
