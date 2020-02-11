from tensorflow.keras.utils import Sequence, to_categorical
from .Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from skimage.measure import block_reduce
import cv2, os, copy, pickle
import numpy as np
from scipy.ndimage import interpolation
import SimpleITK as sitk


class Image_Processor(object):

    def pre_process(self, image, annotation):
        return image, annotation

    def nusance_process(self, image, annotation):
        return image, annotation

def load_obj(path):
    if path.find('.pkl') == -1:
        path += '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        out = {}
        return out


class image_loader(object):
    def __init__(self, shuffle=False, is_2D=True, batch_size=5, Data_Set_Readers=None):
        assert Data_Set_Readers is not None, 'Need to pass dictionaries from Data_Set_Reader'
        if type(Data_Set_Readers) is not list:
            Data_Set_Readers = [Data_Set_Readers]
        self.data_set_readers = Data_Set_Readers
        self.shuffle = shuffle
        if is_2D:
            self.consolidate_patients_2D()
            self.distribute_images()

    def consolidate_patients_2D(self):
        self.files_in_reader = []
        for data_reader in self.data_set_readers:
            files = np.asarray(data_reader.patient_dict.values())
            self.files_in_reader.append(files)  # Get all file images
        self.files_in_reader = np.asarray(self.files_in_reader)
        self.min_files = min([len(i) for i in self.files_in_reader])

    def distribute_patients_2D(self):
        if self.shuffle:
            for i in self.files_in_reader:
                np.random.shuffle(i)
        self.file_batches = []
        for i in range(self.min_files):

        patient_list = list(self.patient_dict.keys())
        xxx = 1
        output = []
        for i in range(len(patient_list)):
            output.append(self.patient_dict[patient_list[i]])
            xxx += 1
            if xxx > self.num_patients:
                self.file_batches.append(output)
                output = []
                xxx = 1

    def prep_batches(self,batch_size=1):
        self.load_file_list = self.file_list[:]
        self.file_batches = []
        if not self.by_patient:
            perm = np.arange(len(self.file_list))
            if self.shuffle_images:
                np.random.shuffle(perm)
            self.load_file_list = list(np.asarray(self.file_list)[perm])
            while len(self.load_file_list) > batch_size:
                file_list = []
                for _ in range(batch_size):
                    file_list.append(self.load_file_list[0])
                    del self.load_file_list[0]
                self.file_batches.append(file_list)
        else:
            keys = list(self.patient_dict.keys())
            i = 0
            temp_batches = []
            for key in keys:
                temp_batches.append(self.patient_dict[key])
                i += 1
                if i >= batch_size:
                    self.file_batches.append(temp_batches)
                    temp_batches, i = [], 0


class image_loader_old(object):
    def __init__(self,random_start=True, all_images=False):
        self.patient_dict_indexes = {}
        self.image_dictionary = {}
        self.random_start = random_start
        self.all_images = all_images

    def load_image(self, batch_size=0, image_names=None):
        images, annotations = np.ones([batch_size, self.image_size, self.image_size],dtype='float32')*-1000, \
                              np.zeros([batch_size, self.image_size, self.image_size],dtype='int8')
        temp_blank = lambda i: np.zeros([self.image_size, self.image_size, i.shape[-1] + 1])
        add = 0
        start = 0
        finish = len(image_names)
        if not self.random_start:
            batch_size = finish
        description = ''
        file = image_names[0]
        ext = '.npy'
        if file.find('.nii.gz') != -1:
            ext = '.nii.gz'
        if self.by_patient and batch_size != 0 and not self.all_images:
            broken_up = file.split('\\')
            if len(broken_up) == 1:
                broken_up = file.split('/')
            broken_up = broken_up[-1].split('_')
            if broken_up[-1].find('image') == 0: # Making everything back compatible with the new style of passing data
                slice_num = int(broken_up[-2])
                description = ''
                for i in broken_up[:-2]:
                    description += i + '_'
                description = description[:-1]
            else:
                slice_num = int(broken_up[-1].split('.')[0])
                description = ''
                for i in broken_up[:-1]:
                    description += i + '_'
                description = description[:-1]
        if len(image_names) > batch_size:
            if description not in self.patient_dict_indexes:
                start = len(image_names) - (batch_size + add)
                start = np.random.randint(start)
                if start < add:
                    start = add
                finish = int(start + batch_size)
                start = int(start)
                if finish > len(image_names):
                    finish = int(len(image_names))
            else:
                values = self.patient_dict_indexes[description][self.wanted_indexes[-1]]
                np.random.shuffle(values)
                new_file = file.replace(slice_num+ext,str(values[0])+ext)
                if os.path.exists(new_file):
                    start = image_names.index(new_file)
                    finish = min([int(start+batch_size),len(image_names)])
        k = None
        make_changes = True
        for index, i in enumerate(range(start,finish)):
            if i < 0 or i > len(image_names):
                print('start:' + str(start) + 'total images: ' + str(len(image_names)) + '_i:' + str(i))
            if image_names[i] not in self.image_dictionary or not self.save_and_reload:
                image_name = image_names[i]
                if image_name.find('_image' + ext) == -1:
                    if ext == '.npy':
                        data = np.load(image_name)
                    else:
                        data_handle = sitk.ReadImage(image_name)
                        data = sitk.ReadImage(data_handle)
                    images_temp = data[0, :, :][None,...]
                    annotations_temp = data[1, :, :][None,...]
                else:
                    if ext == '.npy':
                        images_temp = np.load(image_name)
                        annotations_temp = np.load(image_name.replace('_image.npy','_annotation.npy'))
                    else:
                        images_temp_handle = sitk.ReadImage(image_name)
                        images_temp = sitk.GetArrayFromImage(images_temp_handle)[None,...]
                        annotations_temp_handle = sitk.ReadImage(image_name.replace('_image.nii.gz','_annotation.nii.gz'))
                        annotations_temp = sitk.GetArrayFromImage(annotations_temp_handle)[None,...]
                if (make_changes or not self.by_patient) or (images_temp.shape[1] != self.image_size or images_temp.shape[2] != self.image_size):
                    if images_temp.shape[1] >= self.image_size*2 and images_temp.shape[2] >= self.image_size*2:
                        if len(annotations_temp.shape) == 3:
                            block = (2,2)
                        else:
                            block = (2,2,1)
                        images_temp = block_reduce(images_temp[0,...], block, np.average).astype('float32')[None,...]
                        annotations_temp = block_reduce(annotations_temp[0,...].astype('int'), block, np.max).astype('int')[None,...]
                    elif images_temp.shape[1] <= self.image_size / 2 or images_temp.shape[2] <= self.image_size / 2:
                        images_temp, annotations_temp = self.give_resized_images(images_temp, annotations_temp)
                    if images_temp.shape[0] != 1:
                        images_temp = images_temp[None,...]
                        annotations_temp = annotations_temp[None,...]
                    images_temp, annotations_temp = self.convert_image_size(images_temp, annotations_temp,
                                                                            self.image_size)
                if len(annotations_temp.shape) > 3 and annotations_temp.shape[-1] != annotations_temp.shape[-2]:
                    if k is None:
                        k = temp_blank(annotations_temp)
                    k[..., 1:] = annotations_temp
                    annotations_temp = np.argmax(k, axis=-1)
                self.image_dictionary[image_names[i]] = [images_temp.astype('float32'), annotations_temp]
            else:
                images_temp, annotations_temp = self.image_dictionary[image_names[i]]
            images[index] = np.squeeze(images_temp)
            annotations[index] = np.squeeze(annotations_temp)


        if self.perturbations:
            if not self.by_patient:
                for i in range(len(image_names)):
                    images[i,:,:], annotations[i,:,:] = self.pertubartion_class.make_pertubartions(images[i,:,:],annotations[i,:,:])
            else:
                images, annotations = self.pertubartion_class.make_pertubartions(images,annotations)

        if self.three_channel and images.shape[-1] != 3:
            images_stacked = np.stack([images,images,images],axis=-1)
        else:
            images_stacked = np.expand_dims(images,axis=-1)
        images = images_stacked

        if images.shape[0] != batch_size:
            i = 0
            while images.shape[0] < batch_size:
                if i == 0:
                    images = np.concatenate((images, np.expand_dims(images[-1, :, :, :], axis=0)),
                                                  axis=0)
                    annotations = np.concatenate(
                        (annotations, np.expand_dims(annotations[-1, :, :], axis=0)), axis=0)
                    i = 1
                elif i == 1:
                    images = np.concatenate((np.expand_dims(images[0, :, :, :], axis=0), images),
                                                  axis=0)
                    annotations = np.concatenate(
                        (np.expand_dims(annotations[0, :, :], axis=0), annotations), axis=0)
                    i = 0
            while images.shape[0] > batch_size:
                if i == 0:
                    images = images[1:, :, :, :]
                    annotations = annotations[1:, :, :]
                    i = 1
                elif i == 1:
                    images = images[:-1, :, :, :]
                    annotations = annotations[:-1, :, :]
                    i = 0
        if self.final_steps:
            images, annotations = self.final_steps(images,annotations)
        return images, annotations

    def get_bounding_box_indexes(self, annotation):
        '''
        :param annotation: A binary image of shape [# images, # rows, # cols, channels]
        :return: the min and max z, row, and column numbers bounding the image
        '''
        indexes = np.where(np.any(annotation, axis=(1, 2)) == True)[0]
        min_z_s, max_z_s = indexes[0], indexes[-1]
        '''
        Get the row values of primary and secondary
        '''
        indexes = np.where(np.any(annotation, axis=(0, 2)) == True)[0]
        min_r_s, max_r_s = indexes[0], indexes[-1]
        '''
        Get the col values of primary and secondary
        '''
        indexes = np.where(np.any(annotation, axis=(0, 1)) == True)[0]
        min_c_s, max_c_s = indexes[0], indexes[-1]
        return min_z_s, max_z_s, min_r_s, max_r_s, min_c_s, max_c_s

    def pad_images(self, images, annotations, output_size=None,value=0):
        if not output_size:
            print('did not provide a desired size')
            return images, annotations
        holder = output_size - np.asarray(images.shape)
        val_differences = [[max([int(i / 2), 0]), max([int(i / 2), 0])] for i in holder]
        images, annotations = np.pad(images, val_differences, 'constant', constant_values=(value)), \
                        np.pad(annotations, val_differences, 'constant', constant_values=(0))
        holder = output_size - np.asarray(images.shape)
        final_pad = [[0, i] for i in holder]
        images, annotations = np.pad(images, final_pad, 'constant', constant_values=(value)), \
                              np.pad(annotations, final_pad, 'constant', constant_values=(0))
        return images, annotations

    def load_images(self,index,batch_size=0):
        if self.by_patient:
            image_names_all = self.file_batches[index]
            if type(image_names_all[0]) != list:
                image_names_all = [image_names_all]
            image_names = image_names_all[0]
            if self.all_images:
                batch_size = len(image_names)
            images_out, annotations_out = self.load_image(batch_size=batch_size, image_names=image_names)
            if len(image_names_all) > 1:
                images_out = np.expand_dims(images_out,axis=0)
                annotations_out = np.expand_dims(annotations_out,axis=0)
            for i in range(1,len(image_names_all)):
                image_names = image_names_all[i]
                images, annotations = self.load_image(batch_size=batch_size, image_names=image_names)
                images_out = np.concatenate([images_out,np.expand_dims(images,axis=0)],axis=0)
                annotations_out = np.concatenate([annotations_out, np.expand_dims(annotations, axis=0)], axis=0)
        else:
            image_names = self.file_batches[index]
            images_out, annotations_out = self.load_image(batch_size=batch_size, image_names=image_names)
        self.images = images_out
        self.annotations = annotations_out
        return images_out, annotations_out

    def return_images(self):
        return self.images, self.annotations

class Data_Set_Reader(object):
    def __init__(self,path=None,expansion=0):
        assert path is not None, 'Need to pass a list of paths'
        assert os.path.exists(path), 'Path {} does not exist'.format(path)
        self.expansion = expansion
        self.start_stop_dict = {}
        self.patient_dict = {}
        self.file_list = []
        self.all_files = []
        self.start_stop_dict = {}
        if 'descriptions_start_and_stop.pkl' in os.listdir(path):
            self.start_stop_dict[path] = load_obj(os.path.join(path, 'descriptions_start_and_stop.pkl'))
        else:
            print('No start_and_stop description in {}'.format(path))
        self.load_file_list = self.file_list[:]
        self.get_file_list(path)
        self.make_patient_list()

    def get_file_list(self,path):
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.find('.nii.gz') == -1:
                    continue
                if file.find('_annotation.') == -1:
                    self.file_list.append(os.path.join(path, file))

    def make_patient_list(self):
        self.patient_dict = {}
        self.patient_dict_indexes = {}
        self.start_stop_dict_local = {}
        for file in self.file_list:
            broken_up = file.split('\\')
            if len(broken_up) == 1:
                broken_up = file.split('/')
            broken_up = broken_up[-1].split('_')
            slice_num = int(broken_up[-1].split('.')[0])
            description = ''
            for i in broken_up[:-1]:
                description += i + '_'
            description = description[:-1]
            values = None
            if description in self.start_stop_dict.keys() and description not in self.start_stop_dict_local.keys():
                start, stop = self.start_stop_dict[description]['start'], self.start_stop_dict[description]['stop']
                self.start_stop_dict_local[description] = {'start stop':[start, stop],'values':values}
            elif description in self.start_stop_dict_local.keys():
                start,stop = self.start_stop_dict_local[description]['start stop']
                values = self.start_stop_dict_local[description]['values']
            else:
                print(description)
                start, stop = 0,999
                print('no start or stop')
            start -= self.expansion
            stop += self.expansion
            if description in self.patient_dict.keys():
                files = self.patient_dict[description]
            else:
                files = {}
            if np.any(values):
                if np.max(np.abs(slice_num-values) < self.expansion):
                    files[slice_num] = file
            elif slice_num >= start:
                if slice_num <= stop:
                    files[slice_num] = file
            self.patient_dict[description] = files
        for pat in self.patient_dict.keys():
            slice_vals = list(self.patient_dict[pat].keys())
            indexes = [i[0] for i in sorted(enumerate(slice_vals), key=lambda x: x[1])]
            file_names = []
            for key in self.patient_dict[pat].keys():
                file_names.append(self.patient_dict[pat][key])
            file_names = list(np.asarray(file_names)[indexes])
            self.patient_dict[pat] = file_names
        self.all_files = self.patient_dict.values()


    def __len__(self):
        return len(self.file_list)

    def shuffle(self):
        self.prep_batches()
        if self.by_patient:
            perm = np.arange(len(self.file_batches))
            np.random.shuffle(perm)
            self.file_batches = list(np.asarray(self.file_batches)[perm])


class Pertubartion_Class:
    def __init__(self,pertubartions,image_shape):
        self.pertubartions = pertubartions
        self.output_annotation_template = np.zeros(image_shape)
        self.output_images_template = np.zeros(image_shape)
        self.M_image = {}
        self.image_shape = image_shape

    def make_pertubartions(self,images,annotations):
        min_val = np.min(images)
        images -= min_val # This way any rotation gets a 0, irrespective of previous normalization
        for key in self.pertubartions.keys():

            variation = self.pertubartions[key][np.random.randint(0, len(self.pertubartions[key]))]
            if key == 'Rotation':
                shape_size_image = shape_size_annotation = self.image_shape[1]
                if variation not in self.M_image.keys():
                    M_image = cv2.getRotationMatrix2D((int(shape_size_image) / 2, int(shape_size_image) / 2), variation,1)
                    self.M_image[variation] = M_image
                else:
                    M_image = self.M_image[variation]
                if variation != 0:
                    # images = cv2.warpAffine(images,M_image, (int(shape_size_image), int(shape_size_image)))
                    output_image = np.zeros(images.shape,dtype=images.dtype)
                    if len(images.shape) > 2:
                        for image in range(images.shape[0]):
                            im = images[image, :, :]
                            if np.max(im) != 0:
                                im = cv2.warpAffine(im, M_image, (int(shape_size_image), int(shape_size_image)),flags=cv2.INTER_LINEAR)
                            output_image[image, :, :] = im
                    else:
                        output_image = cv2.warpAffine(images, M_image, (int(shape_size_image), int(shape_size_image)),flags=cv2.INTER_LINEAR)
                    images = output_image
                    output_annotation = np.zeros(annotations.shape,dtype=annotations.dtype)
                    for val in range(1, int(annotations.max()) + 1):
                        temp = copy.deepcopy(annotations).astype('int')
                        temp[temp != val] = 0
                        temp[temp > 0] = 1
                        if len(annotations.shape) > 2:
                            for image in range(annotations.shape[0]):
                                im = temp[image, :, :]
                                if np.max(im) != 0:
                                    im = cv2.warpAffine(im, M_image,
                                                        (int(shape_size_annotation), int(shape_size_annotation)),flags=cv2.INTER_NEAREST)
                                    im[im > 0.1] = val
                                    im[im < val] = 0
                                    output_annotation[image, :, :][im == val] = val
                        else:
                            im = temp
                            if np.max(im) != 0:
                                im = cv2.warpAffine(im, M_image,
                                                    (int(shape_size_annotation), int(shape_size_annotation)),flags=cv2.INTER_NEAREST)
                                im[im > 0.1] = val
                                im[im < val] = 0
                                output_annotation[im == val] = val
                        # output_annotation[annotations == val] = val
                    annotations = output_annotation
            elif key == 'Shift':
                variation_row = variation
                variation_col = self.pertubartions[key][np.random.randint(-len(self.pertubartions[key]), len(self.pertubartions[key]))]
                if len(images.shape) == 2:
                    output_image = interpolation.shift(images,[variation_row, variation_col])
                    annotations = interpolation.shift(annotations.astype('int'),
                                                                    [variation_row, variation_col])
                elif len(images.shape) == 3:
                    output_image = interpolation.shift(images, [0, variation_row, variation_col])
                    annotations = interpolation.shift(annotations.astype('int'),
                                                                    [0, variation_row, variation_col])
                images = output_image
            elif key == 'Scale':
                if variation != 0:
                    output_image = np.zeros(images.shape, dtype=images.dtype)
                    if len(images.shape) > 2:
                        for image in range(images.shape[0]):
                            im = images[image, :, :]
                            if np.max(im) != 0:
                                temp_scale = cv2.resize(images, None, fx=1 + variation, fy=1 + variation,
                                                        interpolation=cv2.INTER_LINEAR)
                                center = (temp_scale.shape[0] // 2, temp_scale.shape[1] // 2)
                                if variation > 0:
                                    im = temp_scale[int(center[0] - 512 / 2):int(center[0] + 512 / 2),
                                         int(center[1] - 512 / 2):int(center[1] + 512 / 2)]
                                elif variation < 0:
                                    im = np.pad(temp_scale, [
                                        (abs(int(center[0] - 512 / 2)), abs(int(center[0] - 512 / 2))),
                                        (abs(int(center[1] - 512 / 2)),
                                         abs(int(center[1] - 512 / 2)))], mode='constant',
                                                constant_values=np.min(temp_scale))
                            output_image[image, :, :] = im
                    else:
                        temp_scale = cv2.resize(images, None, fx=1 + variation, fy=1 + variation,
                                                interpolation=cv2.INTER_LINEAR)
                        center = (temp_scale.shape[0] // 2, temp_scale.shape[1] // 2)
                        if variation > 0:
                            output_image = temp_scale[int(center[0] - 512 / 2):int(center[0] + 512 / 2),
                                           int(center[1] - 512 / 2):int(center[1] + 512 / 2)]
                        elif variation < 0:
                            output_image = np.pad(temp_scale, [
                                (abs(int(center[0] - 512 / 2)), abs(int(center[0] - 512 / 2))),
                                (abs(int(center[1] - 512 / 2)),
                                 abs(int(center[1] - 512 / 2)))], mode='constant', constant_values=np.min(temp_scale))
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
                                    temp_scale = cv2.resize(im, None, fx=1 + variation, fy=1 + variation,
                                                            interpolation=cv2.INTER_NEAREST)
                                    center = (temp_scale.shape[0] // 2, temp_scale.shape[1] // 2)
                                    if variation > 0:
                                        im = temp_scale[int(center[0] - 512 / 2):int(center[0] + 512 / 2),
                                             int(center[1] - 512 / 2):int(center[1] + 512 / 2)]
                                    elif variation < 0:
                                        im = np.pad(temp_scale, [
                                            (abs(int(center[0] - 512 / 2)), abs(int(center[0] - 512 / 2))),
                                            (abs(int(center[1] - 512 / 2)),
                                             abs(int(center[1] - 512 / 2)))], mode='constant',
                                                    constant_values=np.min(temp_scale))

                                    im[im > 0.1] = val
                                    im[im < val] = 0
                                    output_annotation[image, :, :][im == val] = val
                        else:
                            im = temp
                            if np.max(im) != 0:
                                temp_scale = cv2.resize(im, None, fx=1 + variation, fy=1 + variation,
                                                        interpolation=cv2.INTER_NEAREST)
                                center = (temp_scale.shape[0] // 2, temp_scale.shape[1] // 2)
                                if variation > 0:
                                    im = temp_scale[int(center[0] - 512 / 2):int(center[0] + 512 / 2),
                                         int(center[1] - 512 / 2):int(center[1] + 512 / 2)]
                                elif variation < 0:
                                    im = np.pad(temp_scale, [
                                        (abs(int(center[0] - 512 / 2)), abs(int(center[0] - 512 / 2))),
                                        (abs(int(center[1] - 512 / 2)),
                                         abs(int(center[1] - 512 / 2)))], mode='constant',
                                                constant_values=np.min(temp_scale))
                                im[im > 0.1] = val
                                im[im < val] = 0
                                output_annotation[im == val] = val
                    annotations = output_annotation

        images += min_val
        output_image = images
        output_annotation = annotations
        return output_image, output_annotation


class Train_Data_Generator2D(Sequence):
    def __init__(self, image_size=512, batch_size=5, perturbations=None, num_of_classes=2, data_paths=None,clip=0,expansion=0,
                 whole_patient=False, shuffle=False, flatten=False, noise=0.0, normalize_to_255=False,z_images=16,auto_normalize=False,
                 all_for_one=False, three_channel=True, using_perturb_engine=False,on_VGG=False,normalize_to_value=None,
                 resize_class=None,add_filename_extension=True, is_test_set=False, reduced_interest=False, mean_val=0, std_val=1):
        self.z_images = z_images
        self.auto_normalize = auto_normalize
        self.max_images = np.inf
        self.normalize_to_value = normalize_to_value
        self.reduced_interest = reduced_interest
        self.resize_class = resize_class
        self.using_perturb_engine = using_perturb_engine
        if type(clip) == int:
            clip = [clip for _ in range(4)]
        self.clip = clip
        self.noise = noise
        self.flatten = flatten
        self.shuffle = shuffle
        self.all_for_one = all_for_one
        self.num_of_classes = num_of_classes
        self.on_VGG = on_VGG
        extension = 'Single_Images3D'
        self.mean_val = mean_val
        self.std_val = std_val
        self.normalize_to_255 = normalize_to_255
        if self.using_perturb_engine:
            extension += '\\Perturbations'
        self.image_size = image_size
        self.batch_size = batch_size
        self.perturbations = perturbations
        self.image_list = []
        models = {}
        for path in data_paths:
            if path.find(extension) == -1 and add_filename_extension:
                path = os.path.join(path,extension)
            models[path] = Data_Set_Reader(shuffle_images=shuffle,expansion=expansion,
                path=path, by_patient=whole_patient, is_test_set=is_test_set)
        self.training_models = models
        self.train_dataset_reader = Data_Set_Reader(perturbations=self.perturbations,verbose=False,
                                                    image_size=image_size,three_channel=three_channel,
                                                    by_patient=whole_patient,resize_class=resize_class, is_test_set=is_test_set,
                                                    shuffle_images=shuffle)
        self.get_image_lists()

    def get_image_lists(self):
        list_len = []
        self.image_list = []
        for key in self.training_models.keys():
            if self.shuffle:
                self.training_models[key].shuffle()
            self.training_models[key].prep_batches(self.batch_size)
            list_len.append(len(self.training_models[key].file_batches))
        if self.all_for_one:
            for i in range(min(list_len)):
                images = []
                for key in self.training_models.keys():
                    # self.image_list.append(self.training_models[key].file_batches[i])
                    images += self.training_models[key].file_batches[i]
                self.image_list.append(images)
        else:
            for i in range(min(list_len)):
                for key in self.training_models.keys():
                    try:
                        self.image_list.append(self.training_models[key].file_batches[i])
                    except:
                        print(i)
                        print(len(self.training_models[key].file_batches))
        self.train_dataset_reader.file_batches = self.image_list

    def __getitem__(self,index):
        train_images, annotations = self.train_dataset_reader.load_images(index, batch_size=self.batch_size) # wanting multi-patient batches, use the 3D model
        # Center it about VGG 19
        if self.reduced_interest:
            for i in range(train_images.shape[-1]):
                temp = train_images[:,:,:,i]
                temp[annotations[:,:,:,0]==0] = 0
                train_images[:,:,:,i] = temp
            annotations[annotations == 3] = 0
        if not self.using_perturb_engine:
            annotations = to_categorical(annotations,self.num_of_classes)
        if self.flatten:
            class_weights_dict = {0:1,1:20}
            class_weights = np.ones([annotations.shape[0],annotations.shape[1],annotations.shape[2]])
            for i in range(self.num_of_classes):
                class_weights[annotations[:,:,:,i] == 1] = class_weights_dict[i]
            annotations = np.reshape(annotations,
                                        [train_images.shape[0],self.image_size*self.image_size*self.num_of_classes])
            class_weights = np.reshape(class_weights,[train_images.shape[0],self.image_size*self.image_size,1])
            return train_images, annotations, class_weights
        if self.auto_normalize:
            data = train_images.flatten()
            data.sort()
            ten_percent = int(len(data) / 10)
            data = data[int(ten_percent * 5):]
            self.mean_val = np.mean(data)
            self.std_val = np.std(data)
        if self.mean_val != 0 or self.std_val != 1:
            train_images = (train_images-self.mean_val)/self.std_val
            if self.noise != 0:
                train_images += self.noise * np.random.normal(loc=0.0, scale=1.0, size=train_images.shape)
            train_images[train_images>3.55] = 3.55
            train_images[train_images<-3.55] = -3.55 # 6 sigma
            if self.normalize_to_255:
                train_images = (train_images + 3.55)/(3.55*2)
                train_images *= 255
                train_images[train_images<0] = 0
                train_images[train_images>255] = 255
            if self.normalize_to_value:
                train_images *= self.normalize_to_value
        if max(self.clip) > 0:
            if len(train_images.shape) == 5:
                train_images, annotations = train_images[:,:,self.clip[0]:-self.clip[2],self.clip[1]:-self.clip[3],:], \
                                            annotations[:,:,self.clip[0]:-self.clip[2],self.clip[1]:-self.clip[3],:]
            else:
                train_images, annotations = train_images[:, self.clip[0]:-self.clip[2],self.clip[1]:-self.clip[3], :], \
                                            annotations[:,self.clip[0]:-self.clip[2],self.clip[1]:-self.clip[3], :]
        if self.on_VGG:
            train_images[:, :, :, 0] -= 123.68
            train_images[:, :, :, 1] -= 116.78
            train_images[:, :, :, 2] -= 103.94
        return train_images, annotations

    def __len__(self):
        int(len(self.image_list))
        return min([self.max_images,int(len(self.image_list))])

    def on_epoch_end(self):
        self.get_image_lists()

if __name__ == '__main__':
    xxx = 1