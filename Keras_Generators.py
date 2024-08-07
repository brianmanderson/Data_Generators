from tensorflow.python.keras.utils import Sequence, to_categorical
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.models import load_model
from skimage import morphology
import os, glob, pickle
from .Image_Processors_Module.src.Processors.KerasGeneratorProcessors import *


def dice_coef_3D(y_true, y_pred, smooth=0.0001):
    intersection = K.sum(y_true[...,1:] * y_pred[...,1:])
    union = K.sum(y_true[...,1:]) + K.sum(y_pred[...,1:])
    return (2. * intersection + smooth) / (union + smooth)


def load_obj(path):
    if path.find('.pkl') == -1:
        path += '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        out = {}
        return out


def remove_non_liver(annotations, threshold=0.5, volume_threshold=9999999):
    annotations = copy.deepcopy(annotations)
    if len(annotations.shape) == 4:
        annotations = annotations[...,0]
    if not annotations.dtype == 'int':
        annotations[annotations < threshold] = 0
        annotations[annotations > 0] = 1
        annotations = annotations.astype('int')
    labels = morphology.label(annotations, neighbors=4)
    area = []
    max_val = 0
    for i in range(1,labels.max()+1):
        new_area = labels[labels == i].shape[0]
        if new_area > volume_threshold:
            continue
        area.append(new_area)
        if new_area == max(area):
            max_val = i
    labels[labels != max_val] = 0
    labels[labels > 0] = 1
    annotations = labels
    return annotations


def get_bounding_box_indexes(annotation):
    '''
    :param annotation: A binary image of shape [# images, # rows, # cols, channels]
    :return: the min and max z, row, and column numbers bounding the image
    '''
    annotation = np.squeeze(annotation)
    if annotation.dtype != 'int':
        annotation[annotation>0.1] = 1
        annotation = annotation.astype('int')
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


def pad_images(images,annotations,output_size=None,value=0):
    if not output_size:
        print('did not provide a desired size')
        return images, annotations
    holder = output_size - np.asarray(images.shape)
    if np.max(holder) == 0:
        return images, annotations
    val_differences = [[max([int(i/2 - 1),0]), max([int(i/2 - 1),0])] for i in holder]
    if np.max(val_differences) > 0:
        images, annotations = np.pad(images, val_differences, 'constant', constant_values=(value)), \
                        np.pad(annotations, val_differences, 'constant', constant_values=(0))
    holder = output_size - np.asarray(images.shape)
    final_pad = [[0, i] for i in holder]
    if np.max(final_pad) > 0:
        images, annotations = np.pad(images, final_pad, 'constant', constant_values=(value)), \
                        np.pad(annotations, final_pad, 'constant', constant_values=(0))
    return images, annotations


def pull_cube_from_image(images, annotation, desired_size=(16,32,32), samples=10):
    output_images = np.ones([samples,desired_size[0],desired_size[1],desired_size[2],1])*np.min(images)
    output_annotations = np.zeros([samples, desired_size[0], desired_size[1], desired_size[2], annotation.shape[-1]])
    pat_locations, z_locations, r_locations, c_locations = np.where(annotation[...,-1] == 1)
    for i in range(samples):
        index = np.random.randint(len(z_locations))
        z_start = max([0, int(z_locations[index] - desired_size[0] / 2)])
        z_stop = min([z_start + desired_size[0], images.shape[1]])
        r_start = max([0, int(r_locations[index] - desired_size[1] / 2)])
        r_stop = min([r_start + desired_size[1], images.shape[2]])
        c_start = max([0, int(c_locations[index] - desired_size[2] / 2)])
        c_stop = min([c_start + desired_size[2], images.shape[3]])
        output_images[i, :z_stop - z_start, :r_stop - r_start, :c_stop - c_start, ...] = images[pat_locations[index],z_start:z_stop,r_start:r_stop,c_start:c_stop,...]
        output_annotations[i, :z_stop - z_start, :r_stop - r_start, :c_stop - c_start, ...] = annotation[pat_locations[index],z_start:z_stop,r_start:r_stop,c_start:c_stop,...]
    return output_images, output_annotations


def center_image_based_on_annotation(images,annotation,mask,layers=2,extensions=(0,0,0)):
    '''
    :param images: 1, images, rows, columns, channels
    :param annotation: 1, images, rows, cols, channels
    :param mask: images, rows, cols
    :param layers:
    :param extensions:
    :return:
    '''
    if mask.dtype != 'int':
        mask[mask>0.1] = 1
        mask = mask.astype('int')
    mask = remove_non_liver(mask)
    z_start, z_stop, r_start, r_stop, c_start, c_stop = get_bounding_box_indexes(np.expand_dims(mask,axis=-1))
    max_image_number = 150
    if z_stop - z_start > max_image_number:
        dif = int((max_image_number - (z_stop - z_start)))
        if np.random.random() > 0.5:
            z_stop += dif
        else:
            z_start -= dif
    power_val = 2 ** layers
    z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start
    z_total, r_total, c_total = z_total + extensions[0], r_total + extensions[1], c_total + extensions[2]
    remainder_z, remainder_r, remaineder_c = power_val - z_total % power_val if z_total % power_val != 0 else 0, \
                                             power_val - r_total % power_val if r_total % power_val != 0 else 0, \
                                             power_val - c_total % power_val if c_total % power_val != 0 else 0
    min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r + extensions[1], c_total + remaineder_c + extensions[2]
    dif_z = min_images - (z_stop - z_start + 1)
    dif_r = min_rows - (r_stop - r_start + 1)
    dif_c = min_cols - (c_stop - c_start + 1)
    extension = min([min([z_start, images.shape[1] - z_stop]), int(dif_z / 2)])  # Keep everything centered
    z_start, z_stop = z_start - extension, z_stop + extension
    # self.z_start, self.z_stop = int(self.z_start - mult*extension), int(self.z_stop + mult*extension)
    extension = min([min([r_start, images.shape[2] - r_stop]), int(dif_r / 2)])  # Keep everything centered
    r_start, r_stop = r_start - extension, r_stop + extension
    # self.r_start, self.r_stop = int(self.r_start - mult*extension), int(self.r_stop + mult*extension)
    extension = min([min([c_start, images.shape[3] - c_stop]), int(dif_c / 2)])  # Keep everything centered
    # self.c_start, self.c_stop = int(self.c_start - mult * extension), int(self.c_stop + mult * extension)
    c_start, c_stop = c_start - extension, c_stop + extension
    if min_images - (z_stop - z_start) == 1:
        if z_start > 0:
            z_start -= 1
            # self.z_start -= mult
        elif z_stop < images.shape[0]:
            z_stop += 1
            # self.z_stop += mult
    if min_rows - (r_stop - r_start) == 1:
        if r_start > 0:
            r_start -= 1
            # self.r_start -= mult
        elif r_stop < images.shape[1]:
            r_stop += 1
            # self.r_stop += mult
    if min_cols - (c_stop - c_start) == 1:
        if c_start > 0:
            c_start -= 1
            # self.c_start -= mult
        elif c_stop < images.shape[2]:
            c_stop += 1
            # self.c_stop += mult
    images, annotation = images[:, z_start:z_stop, r_start:r_stop, c_start:c_stop], \
                    annotation[:, z_start:z_stop, r_start:r_stop, c_start:c_stop, :]
    images, annotation = pad_images(images, annotation, [1, min_images, min_rows, min_cols, images.shape[-1]],value=-3.55)
    return images, annotation


def cartesian_to_polar(xyz):
    '''
    :param x: x_values in single array
    :param y: y_values in a single array
    :param z: z_values in a single array
    :return: polar coordinates in the form of: radius, rotation away from the z axis, and rotation from the y axis
    '''
    # xyz = np.stack([x, y, z], axis=-1)
    input_shape = xyz.shape
    reshape = False
    if len(input_shape) > 2:
        reshape = True
        xyz = np.reshape(xyz,[np.prod(xyz.shape[:-1]),3])
    polar_points = np.empty(xyz.shape)
    # ptsnew = np.hstack((xyz, np.empty(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    polar_points[:,0] = np.sqrt(xy + xyz[:,2]**2)
    polar_points[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    polar_points[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    if reshape:
        polar_points = np.reshape(polar_points,input_shape)
    return polar_points


def polar_to_cartesian(polar_xyz):
    '''
    :param polar_xyz: in the form of radius, elevation away from z axis, and elevation from y axis
    :return: x, y, and z intensities
    '''
    cartesian_points = np.empty(polar_xyz.shape)
    from_y = polar_xyz[:,2]
    xy_plane = np.sin(polar_xyz[:,1])*polar_xyz[:,0]
    cartesian_points[:,2] = np.cos(polar_xyz[:,1])*polar_xyz[:,0]
    cartesian_points[:,0] = np.sin(from_y)*xy_plane
    cartesian_points[:,1] = np.cos(from_y)*xy_plane
    return cartesian_points


def get_bounding_box(train_images_out_base, train_annotations_out_base, include_mask=True,
                     image_size=512, sub_sample=[64,64,64], random_start=True):
    '''
    :param train_images_out_base: shape[1, #images, image_size, image_size, channels]
    :param train_annotations_out_base: shape[1, #images, image_size, image_size, #classes]
    :param include_mask:
    :param image_size:
    :param sub_sample: the box dimensions to include the organ
    :param random_start: Makes a random sub section
    :return: list of indicies which indicate the bounding box of the organ
    '''
    if len(train_images_out_base.shape) == 4:
        train_images_out_base = np.expand_dims(train_images_out_base, axis=0)
        train_annotations_out_base = np.expand_dims(train_annotations_out_base, axis=0)
    train_images_out = train_images_out_base
    train_annotations_out = train_annotations_out_base
    min_row, min_col, min_z, max_row, max_col, max_z = 0, 0, 0, image_size, image_size, train_images_out.shape[1]
    if include_mask:
        mask_comparison = np.squeeze((np.argmax(train_annotations_out, axis=-1)),axis=0)
        itemindex = np.where(mask_comparison > 0)
        min_z, max_z = min(itemindex[0]), max(itemindex[0])
        min_row, max_row = min(itemindex[1]), max(itemindex[1])
        min_col, max_col = min(itemindex[2]), max(itemindex[2])
        if random_start:
            min_row = min_row - int(sub_sample[1] / 2) if min_row - int(sub_sample[1] / 2) > 0 else 0
            min_col = min_col - int(sub_sample[2] / 2) if min_col - int(sub_sample[2] / 2) > 0 else 0
            min_z = min_z - int(sub_sample[0]/2) if min_z - int(sub_sample[0]/2) > 0 else 0
            max_row = max_row + int(sub_sample[1]/2) if max_row + sub_sample[1]/2 < image_size else image_size
            max_col = max_col + int(sub_sample[2]/2) if max_col + sub_sample[2]/2 < image_size else image_size
            max_z = max_z + sub_sample[0]/2 if max_z + sub_sample[0]/2 < train_images_out.shape[1] else train_images_out.shape[1]
            got_region = False
            while not got_region:
                z_start = np.random.randint(min_z, max_z - sub_sample[0]) if max_z - sub_sample[0] > min_z else min_z
                row_start = np.random.randint(min_row,max_row - sub_sample[1])
                col_start = np.random.randint(min_col,max_col - sub_sample[2])
                if z_start < 0:
                    z_start = 0
                col_stop = col_start + sub_sample[2]
                row_stop = row_start + sub_sample[1]
                z_stop = z_start + sub_sample[0] if z_start + sub_sample[0] <= train_images_out.shape[1] else train_images_out.shape[1]
                # train_images_out = train_images_out[:, z_start:z_stop, row_start:row_stop, col_start:col_stop, :]
                # train_annotations_out = train_annotations_out[:, z_start:z_stop, row_start:row_stop, col_start:col_stop, :]
                if not include_mask:
                    got_region = True
                elif np.any(mask_comparison[z_start:z_stop, row_start:row_stop, col_start:col_stop] > 0):
                    got_region = True
            return z_start, z_stop, row_start, row_stop, col_start, col_stop
        else:
            return min_z, max_z, min_row, max_row, min_col, max_col
    else:
        return min_z, max_z, min_row, max_row, min_col, max_col


class Data_Set_Reader(object):
    def __init__(self,path=None, verbose=True, expansion=0, wanted_indexes=None):
        '''
        :param path:
        :param by_patient:
        :param verbose:
        :param num_patients:
        :param is_test_set:
        :param random_start:
        :param expansion:
        :param shuffle_images:
        :param wanted_indexes: a tuple of indexes wanted (2) will pull disease only if 1 is liver
        '''
        self.wanted_indexes = wanted_indexes
        self.expansion = expansion
        self.start_stop_dict = {}
        self.patient_dict = {}
        self.verbose = verbose
        self.file_batches = []
        self.data_path = path
        self.file_list = []
        self.file_ext = '.npy'
        if path:
            if 'descriptions_start_and_stop.pkl' in os.listdir(path):
                self.start_stop_dict = load_obj(os.path.join(path, 'descriptions_start_and_stop.pkl'))
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.find('.npy') == -1 and file.find('.nii.gz') == -1:
                        continue
                    if file.find('_annotation.') == -1:
                        if file.find('.nii.gz') != -1:
                            self.file_ext = '.nii.gz'
                        self.file_list.append(os.path.normpath(os.path.join(path, file)))
            elif self.verbose:
                print(path)
                print('Wrong path')
        self.load_file_list = []
        self.make_patient_list()

    def make_patient_list(self):
        self.patient_dict = {}
        self.patient_dict_indexes = {}
        self.start_stop_dict_local = {}
        if not self.file_list and self.verbose:
            print('No files found')
        for file in self.file_list:
            broken_up = file.split('\\')
            if len(broken_up) == 1:
                broken_up = file.split('/')
            broken_up = broken_up[-1].split('_')
            if broken_up[-1].find('image') == 0: # Making everything back compatible with the new style of passing Data
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
            start, stop = None, None
            values = None
            if description in self.start_stop_dict.keys() and description not in self.start_stop_dict_local.keys():
                if self.wanted_indexes:
                    for index in self.wanted_indexes:
                        if index in self.start_stop_dict[description]:
                            values = self.start_stop_dict[description][index]
                            self.patient_dict_indexes[description] = {index:values}
                            if not start:
                                start, stop = min(values), max(values)
                            else:
                                start, stop = min([start,min(values)]), max([stop,max(values)])
                else:
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
                if np.min(np.abs(slice_num-values)) <= self.expansion:
                    files[slice_num] = file
            elif slice_num >= start and slice_num <= stop:
                files[slice_num] = file
            self.patient_dict[description] = files
        for pat in self.patient_dict.keys():
            slice_vals = list(self.patient_dict[pat].keys())
            indexes = [i[0] for i in sorted(enumerate(slice_vals), key=lambda x: x[1])]
            file_names = [self.patient_dict[pat][key] for key in self.patient_dict[pat].keys()]
            file_names = list(np.asarray(file_names)[indexes])
            self.patient_dict[pat] = file_names
            self.load_file_list += self.patient_dict[pat]


class Data_Generator_Class(Sequence):
    def __init__(self, by_patient=False, whole_patient=False, wanted_indexes=None,data_paths=None, num_patients=1,
                 expansion=np.inf,shuffle=False, batch_size=1, max_batch_size=np.inf,
                 image_processors=None, split_data_evenly_from_paths=False, random_start=True, by_patient_2D=False,
                 random_wiggle_3D=0):
        '''
        :param by_patient: (True/False), load by 3D patient or 2D slices
        :param whole_patient: load entire patient?
        :param wanted_indexes: tuple specifying desired indexes, can be left at None
        :param data_paths: Data paths to pull from
        :param num_patients: if by_patient, how many patients to pull
        :param expansion: how many slices to expand above and below positive annotations
        :param shuffle: shuffle images/patients?
        :param batch_size: number of z_images, if whole_patient this is overridden
        :param save_and_reload: save in a dictionary, default True
        :param image_processors: a list of Data processors, see Image_Processors.py
        :param split_data_evenly_from_paths: in beta
        :param random_start: default, other options in beta
        '''
        self.random_wiggle_3D = random_wiggle_3D
        if by_patient_2D:
            assert num_patients == 1, 'Specified that 2D image output is wanted, but num_patients is > 1'
        self.by_patient_2D = by_patient_2D
        self.random_start = random_start
        self.num_patients = num_patients
        self.max_batch_size = max_batch_size
        self.split_data_evenly_from_paths = split_data_evenly_from_paths
        if whole_patient:
            by_patient = True
        self.by_patient = by_patient
        if image_processors is None:
            image_processors = []
        self.image_dictionary = {}
        self.preload_patient_dict = {}
        self.image_processors = image_processors
        self.max_patients = np.inf
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.whole_patient = whole_patient
        self.is_auto_encoder = False # This can change in sub classes
        self.wanted_indexes = wanted_indexes
        if type(data_paths) is not list:
            data_paths = [data_paths]
        self.patient_dict = {}
        self.patient_dict_indexes = {}
        self.file_list = []
        self.expansion = expansion
        self.training_models = self.get_training_models(data_paths,expansion, wanted_indexes)
        self.get_image_lists()

    def get_training_models(self, data_paths, expansion, wanted_indexes):
        models = {}
        self.start_stop_dicts = {}
        for path in data_paths:
            if path.find('Single_Images3D') == -1:
                path = os.path.join(path,'Single_Images3D') #Make them all 3D
            path = os.path.normpath(path)
            if len(os.listdir(path)) == 0:
                print('Nothing in Data path:' + path)
            self.preload_patient_dict[path] = []
            data_reader = Data_Set_Reader(path=path, expansion=expansion, wanted_indexes=wanted_indexes)
            models[path] = data_reader
            self.start_stop_dicts[path] = data_reader.start_stop_dict
            self.patient_dict[path] = models[path].patient_dict
            self.patient_dict_indexes.update(models[path].start_stop_dict)
            self.file_list += models[path].load_file_list
        for processor in self.image_processors:
            processor.set_start_stop_dict(self.start_stop_dicts)
        return models

    def get_image_lists(self):
        self.image_list = []
        file_batches = []
        batch_split = self.batch_size
        if self.by_patient:
            batch_split = self.num_patients
            if not self.split_data_evenly_from_paths:
                for path in self.patient_dict.keys():
                    for patient in self.patient_dict[path].keys():
                        patient_images = self.patient_dict[path][patient]
                        if self.whole_patient:
                            self.image_list.append(patient_images)
                        else:
                            start = 0
                            if self.random_wiggle_3D != 0:
                                start = np.random.randint(self.random_wiggle_3D)
                            batch = 0
                            batch_images = []
                            pulled = False
                            for i in range(start, len(patient_images)):
                                if batch < self.batch_size:
                                    batch_images.append(patient_images[i])
                                    pulled = False
                                    batch += 1
                                else:
                                    batch = 0
                                    self.image_list.append(batch_images)
                                    pulled = True
                                    batch_images = []
                            if not pulled and batch > self.expansion:
                                self.image_list.append(batch_images)

        else:
            if not self.split_data_evenly_from_paths:
                self.image_list = self.file_list
        if self.shuffle:
            perm = np.arange(len(self.image_list))
            np.random.shuffle(perm)
            self.image_list = list(np.asarray(self.image_list)[perm])
        i = 0
        temp_batch = []
        for image_list in self.image_list:
            if i < batch_split:
                temp_batch.append(image_list)
                i += 1
            if i == batch_split:
                file_batches.append(temp_batch)
                i = 0
                temp_batch = []
        self.file_batches = file_batches

    def load_images_process(self, image_names, batch_size=0):
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
        if self.by_patient and batch_size != 0:
            broken_up = file.split('\\')
            if len(broken_up) == 1:
                broken_up = file.split('/')
            broken_up = broken_up[-1].split('_')
            if broken_up[-1].find('image') == 0: # Making everything back compatible with the new style of passing Data
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
                if self.wanted_indexes:
                    values = self.patient_dict_indexes[description][self.wanted_indexes[-1]]
                    batch_range = np.arange(-batch_size,0)
                    go = True
                    while go:
                        np.random.shuffle(values)
                        np.random.shuffle(batch_range)
                        for value, batch_step in zip(values, batch_range):
                            val = value + batch_step
                            new_file = file.replace('{}_image{}'.format(slice_num, ext), '{}_image{}'.format(val, ext))
                            if new_file in image_names:
                                start = image_names.index(new_file)
                                finish = min([int(start+batch_size),len(image_names)])
                                go = False
                                break
                else:
                    start = self.patient_dict_indexes[description]['start']
                    stop = self.patient_dict_indexes[description]['stop']
                    start += np.random.randint(0,stop-start)
                    finish = min([int(start+batch_size),len(image_names)])
        wanted_names = []
        for index, i in enumerate(range(start,finish)):
            if i < 0 or i > len(image_names):
                print('start:' + str(start) + 'total images: ' + str(len(image_names)) + '_i:' + str(i))
            if image_names[i] not in self.image_dictionary:
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
                        # for image_processor in self.image_processors:
                        #     images_temp_handle, annotations_temp_handle = image_processor.sitk_processes(images_temp_handle,annotations_temp_handle)
                for image_processors in self.image_processors:
                    images_temp, annotations_temp = image_processors.preload_single_image_process(images_temp, annotations_temp)
                images_temp = images_temp[...,None]
                self.image_dictionary[image_names[i]] = [images_temp, annotations_temp]
            wanted_names.append(image_names[i])
        images_temp, annotations_temp = self.image_dictionary[wanted_names[0]]
        images, annotations = np.ones((len(wanted_names),) + images_temp.shape[1:],dtype=images_temp.dtype)*-1000, \
                              np.zeros((len(wanted_names),) + annotations_temp.shape[1:],dtype=annotations_temp.dtype)
        for i, key in enumerate(wanted_names):
            images[i],annotations[i] = self.image_dictionary[key]
        return wanted_names, images, annotations

    def patient_preload_process(self, image_names):
        wanted_names, images, annotations = self.load_images_process(image_names, batch_size=len(image_names))
        for image_processors in self.image_processors:
            images, annotations = image_processors.pre_load_whole_image_process(images, annotations)
        for i, key in enumerate(wanted_names):
            self.image_dictionary[key] = copy.deepcopy([images[i][None,...], annotations[i][None,...]])
        return None

    def load_image(self, batch_size=0, image_names=None):
        wanted_names, images, annotations = self.load_images_process(image_names, batch_size=batch_size)
        for image_processors in self.image_processors:
            images, annotations = image_processors.post_load_process(images, annotations)
        return images, annotations

    def get_patient_name(self, image_names):
        file = image_names[0]
        broken_up = file.split('\\')
        if len(broken_up) == 1:
            broken_up = file.split('/')
            broken_up[1] = '/' + broken_up[1]
            broken_up = broken_up[1:]
        file_key = ''.join(['{}_'.format(i) for i in broken_up[-1].split('_')[:-2]])[:-1]
        path_key = os.path.normpath(file.split(file_key)[0])
        return path_key, file_key

    def load_images(self,index):
        batch_size = self.batch_size
        if self.by_patient:
            image_names_all = self.file_batches[index]
            for i in range(len(image_names_all)):
                image_names = image_names_all[i]
                path_key, file_key = self.get_patient_name(image_names)
                if file_key not in self.preload_patient_dict[path_key]:
                    self.patient_preload_process(self.patient_dict[path_key][file_key])
                    self.preload_patient_dict[path_key].append(file_key)
                if self.whole_patient:
                    batch_size = len(image_names)
                if batch_size > self.max_batch_size:
                    batch_size = self.max_batch_size
                images, annotations = self.load_image(batch_size=batch_size, image_names=image_names)
                if len(images.shape) < 5:
                    images = np.expand_dims(images, axis=0)
                    annotations = np.expand_dims(annotations, axis=0)
                if i == 0:
                    images_out, annotations_out = images, annotations
                else:
                    images_out = np.concatenate([images_out, images], axis=0)
                    annotations_out = np.concatenate([annotations_out, annotations], axis=0)
                for image_processor in self.image_processors:
                    images_out, annotations_out = image_processor.post_load_all_patient_process(images_out,
                                                                                                annotations_out,
                                                                                                path_key=path_key,
                                                                                                file_key=file_key)
            if self.by_patient_2D:
                images_out, annotations_out = images_out[0,...], annotations_out[0,...]
        else:
            image_names = self.file_batches[index]
            images_out, annotations_out = self.load_image(batch_size=batch_size, image_names=image_names)
        return images_out, annotations_out

    def __getitem__(self,index):
        train_images_out, train_annotations_out = self.load_images(index)  # how many images to pull
        return train_images_out, train_annotations_out

    def __len__(self):
        num = len(self.file_batches)
        return num

    def on_epoch_end(self):
        self.get_image_lists()


class Train_DVF_Generator(Sequence):
    def __init__(self, path, save_and_load = False, batch_size = 1, split=False, pool_base = 2, layers=1, reduce=False,
                 mask_images=False,multi_pool=False,polar_coordinates=False,distance_map=False,min_rows_cols = None,is_2D=False,
                 flatten=False, get_CT_images=False, min_val=-1000, max_val=300, mean_val=100, std_val=40, perturbations=None,noise=0,layers_dict=None,output_size=None):
        '''
        :param path: Path to Data
        :param save_and_load: Should we save the images to prevent loading them each time
        :param batch_size: What batch size to grab?
        :param split: Should the images come in concatentated, or as [primary,secondary]
        :param layers: Number of layers in 3D network, to make sure the images are divisible evenly
        '''
        self.is_2D = is_2D
        self.min_rows_cols = min_rows_cols
        if min_rows_cols:
            print('Changing min size to be ' + str(min_rows_cols))
        self.polar_coordinates = polar_coordinates
        self.use_distance_map = distance_map
        self.output_size = output_size
        self.multi_pool = multi_pool
        self.layers_dict = layers_dict
        self.paths = glob.glob(os.path.join(path,'*field*'))
        self.data = {}
        self.batch_size = batch_size
        self.save_and_load = save_and_load
        if save_and_load:
            print('Saving and reloading Data')
        self.split = split
        self.layers = layers
        self.pool_base = pool_base
        self.reduce = reduce
        self.flatten = flatten
        self.get_CT_images = get_CT_images
        self.min, self.max, self.mean_val, self.std_val = min_val, max_val, mean_val, std_val
        self.mask_images = mask_images
        self.perturbations = perturbations
        self.M_image = {}
        self.noise = noise

    def make_pertubartion(self,images,variation,is_annotations=False):
        shape_size_image = images.shape[1]
        if variation not in self.M_image.keys():
            M_image = cv2.getRotationMatrix2D((int(shape_size_image) / 2, int(shape_size_image) / 2), variation,1)
            self.M_image[variation] = M_image
        else:
            M_image = self.M_image[variation]
        if variation != 0:
            output_image = np.zeros(images.shape)
            if len(images.shape) > 2:
                for image in range(images.shape[0]):
                    im = images[image,...]
                    im = cv2.warpAffine(im, M_image, (int(shape_size_image), int(shape_size_image)))
                    if is_annotations:
                        im[im > 0.1] = 1
                        im[im < 1] = 0
                    output_image[image, ...] = im
            else:
                output_image = cv2.warpAffine(images, M_image, (int(shape_size_image), int(shape_size_image)))
            images = output_image

        output_image = images
        return output_image

    def load_images(self,path):
        '''
        :param path:
        :param primary: The CT image of the primary
        :param primary_liver: The binary mask of the primary liver
        :return:
        '''
        # primary, primary_liver, secondary, secondary_liver, field_0, field_1, field_2 = np.load(path)
        if self.use_distance_map:
            primary_liver, secondary_liver = np.load(path.replace('field_reduced','liver_primary_distance_map_reduced')),\
                                             np.load(path.replace('field_reduced','liver_secondary_distance_map_reduced'))
            mapped = 3
            primary_liver, secondary_liver = primary_liver[...,:mapped], secondary_liver[...,:mapped]
            images = np.concatenate([primary_liver, secondary_liver], axis=-1)[None,...]
        else:
            primary_liver, secondary_liver  = np.load(path.replace('field_reduced','liver_primary_reduced')),\
                                              np.load(path.replace('field_reduced','liver_secondary_reduced'))
        field = np.load(path)
        if self.get_CT_images:
            primary, secondary = np.load(path.replace('field_reduced','primary_reduced')), \
                                 np.load(path.replace('field_reduced','secondary_reduced'))
        # field = np.concatenate([np.expand_dims(field_0,axis=-1),np.expand_dims(field_1,axis=-1),np.expand_dims(field_2,axis=-1)],axis=-1)
        mean_val, std_val = [-0.22746165,-0.977084339,0.2880331], [3.509797,3.2545,5.4878]
        # mean_val, std_val = [0, 0, 0], [1,1,1]
        field_shape =(field.shape[0],field.shape[1],field.shape[2],1)
        mean_val, std_val = np.tile(mean_val,field_shape), np.tile(std_val,field_shape)
        field = (field-mean_val)/std_val
        if self.perturbations:
            key = 'Rotation'
            variation = self.perturbations[key][np.random.randint(0, len(self.perturbations[key]))]
            if self.get_CT_images:
                primary = self.make_pertubartion(images=primary,variation=variation,is_annotations=False)
                secondary = self.make_pertubartion(images=secondary, variation=variation, is_annotations=False)
            primary_liver = self.make_pertubartion(images=primary_liver,variation=variation,is_annotations=True)
            secondary_liver = self.make_pertubartion(images=secondary_liver, variation=variation, is_annotations=True)
            field = self.make_pertubartion(images=field, variation=variation, is_annotations=False)
        self.primary_liver = primary_liver
        # min_z_p, max_z_p, min_r_p, max_r_p, min_c_p, max_c_p = get_bounding_box_indexes(np.expand_dims(primary_liver,axis=-1))
        # self.z_start, self.z_stop, self.r_start, self.r_stop, self.c_start, self.c_stop = min_z_p, max_z_p, min_r_p, max_r_p, min_c_p, max_c_p
        if self.reduce:
            primary_liver = block_reduce(primary_liver.astype('int'), tuple([2, 2, 2]), np.max)
            secondary_liver = block_reduce(secondary_liver.astype('int'), tuple([2, 2, 2]), np.max)
            field = block_reduce(field, tuple([2, 2, 2, 1]), np.average)
        field = np.expand_dims(field, axis=0)
        primary_liver, secondary_liver = np.expand_dims(primary_liver, axis=-1), np.expand_dims(secondary_liver,axis=-1)
        if not self.use_distance_map:
            images = np.concatenate([primary_liver, secondary_liver], axis=-1)
            images = np.expand_dims(images, axis=0)
        min_z_p, max_z_p, min_r_p, max_r_p, min_c_p, max_c_p = get_bounding_box_indexes(primary_liver)
        min_z_s, max_z_s, min_r_s, max_r_s, min_c_s, max_c_s = get_bounding_box_indexes(secondary_liver)
        z_start, z_stop = min([min_z_p, min_z_s]), max([max_z_s, max_z_p])
        r_start, r_stop = min([min_r_p,min_r_s]), max([max_r_p,max_r_s])
        c_start, c_stop = min([min_c_s, min_c_p]), max([max_c_p, max_c_s])

        z_start, z_stop = 0, primary_liver.shape[0]
        r_start, c_start = max([0,r_start-50]), max([0,c_start-50])
        r_stop, c_stop = min([512,r_stop+50]), min([512,c_stop+50])

        power_val_x = power_val_y = power_val_z = 0
        # power_val = self.pool_base**self.layers
        if self.layers_dict:
            power_val_z, power_val_x, power_val_y = (1,1,1)
            for layer in self.layers_dict:
                if layer == 'Base':
                    continue
                if 'Pooling' in self.layers_dict[layer]:
                    pooling = self.layers_dict[layer]['Pooling']
                else:
                    pooling = [self.pool_base for _ in range(3)]
                if not self.multi_pool:
                    power_val_z *= pooling[0]
                    power_val_x *= pooling[1]
                    power_val_y *= pooling[2]
                if self.multi_pool:
                    z, x, y = pooling
                    power_val_z, power_val_x, power_val_y = max([z,power_val_z]), max([x, power_val_x]), max([y, power_val_y])
        z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start
        remainder_z, remainder_r, remainder_c = power_val_z - z_total % power_val_z if z_total % power_val_z != 0 else 0, \
                                                 power_val_x - r_total % power_val_x if r_total % power_val_x != 0 else 0, \
                                                 power_val_y - c_total % power_val_y if c_total % power_val_y != 0 else 0
        min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r, c_total + remainder_c
        if self.output_size:
            min_images, min_rows, min_cols = self.output_size
        if self.min_rows_cols:
            min_rows, min_cols = max([self.min_rows_cols,min_rows]), max([self.min_rows_cols,min_cols])
        # min_rows, min_cols = self.min_rows_cols, self.min_rows_cols
        if not self.use_distance_map:
            out_images = np.zeros([1,min_images,min_rows,min_cols,2],dtype=images.dtype)
        else:
            out_images = np.zeros([1,min_images,min_rows,min_cols,images[0].shape[-1]],dtype='float')
        out_field = np.ones([1,min_images,min_rows,min_cols,3],dtype=field.dtype)
        out_field *= np.median(field,axis=(0,1,2,3))
        keep_centered = False
        if keep_centered:
            dif_z = min_images - (z_stop - z_start)
            z_start = max([z_start-dif_z//2,0])
            z_stop = min([images.shape[1] + dif_z//2,images.shape[1]])

            dif_r = min_rows - (r_stop - r_start)
            r_start = max([r_start-dif_r//2,0])
            r_stop = min([images.shape[2] + dif_r//2,images.shape[2]])

            dif_c = min_cols - (c_stop - c_start)
            c_start = max([c_start-dif_c//2,0])
            c_stop = min([images.shape[3] + dif_c//2,images.shape[3]])
            if min_images - (z_stop - z_start) == 1:
                if z_start > 0:
                    z_start -= 1
                elif z_stop < images.shape[0]:
                    z_stop += 1
            if min_rows - (r_stop - r_start) == 1:
                if r_start > 0:
                    r_start -= 1
                elif r_stop < images.shape[1]:
                    r_stop += 1
            if min_cols - (c_stop - c_start) == 1:
                if c_start > 0:
                    c_start -= 1
                elif c_stop < images.shape[2]:
                    c_stop += 1
        while z_stop - z_start < min_images:
            dif_z = min_images - (z_stop - z_start)
            if z_start != 0:
                z_start = max([0,z_start-dif_z])
            elif z_stop != images.shape[1]:
                z_stop = min([images.shape[1],z_stop+dif_z])
            else:
                break
        while r_stop - r_start < min_rows:
            dif_r = min_rows -(r_stop - r_start)
            if r_start != 0:
                r_start = max([0,r_start-dif_r])
            elif r_stop != images.shape[2]:
                r_stop = min([images.shape[2],r_stop+dif_r])
            else:
                break
        while c_stop - c_start < min_cols:
            dif_c = min_cols - (c_stop - c_start)
            if c_start != 0:
                c_start = max([0,c_start-dif_c])
            elif c_stop != images.shape[3]:
                c_stop = min([images.shape[3],c_stop+dif_c])
            else:
                break
        out_images[:,:(z_stop-z_start),:r_stop-r_start,:c_stop-c_start] = images[:, z_start:z_stop, r_start:r_stop, c_start:c_stop]
        out_field[:,:(z_stop-z_start),:r_stop-r_start,:c_stop-c_start,:] = field[:, z_start:z_stop, r_start:r_stop, c_start:c_stop, :]
        # images, field = images[:, z_start:z_stop, r_start:r_stop, c_start:c_stop], \
        #                 field[:, z_start:z_stop, r_start:r_stop, c_start:c_stop, :]
        # images,field = pad_images(images,field,[1,min_images,min_rows,min_cols,images.shape[-1]])
        if self.get_CT_images:
            if self.reduce:
                primary = block_reduce(primary, tuple([2, 2, 2]), np.average)
                secondary = block_reduce(secondary, tuple([2, 2, 2]), np.average)
            primary, secondary = np.expand_dims(primary, axis=-1), np.expand_dims(secondary,axis=-1)
            primary = (primary - self.mean_val)/self.std_val
            secondary = (secondary - self.mean_val)/self.std_val
            primary[primary<-3.55] = -3.55
            primary[primary > 3.55] = 3.55
            secondary[secondary < -3.55] = -3.55
            secondary[secondary>3.55] = 3.55
            primary[primary_liver==0] = 0
            secondary[secondary_liver==0] = 0
            images_CT = np.expand_dims(np.concatenate([primary, secondary], axis=-1),axis=0)
            out_images_CT = np.zeros(out_images.shape,dtype=images_CT.dtype)
            out_images_CT[:,:(z_stop-z_start),:r_stop-r_start,:c_stop-c_start] = images_CT[:, z_start:z_stop, r_start:r_stop, c_start:c_stop]

            # images_CT, _ = pad_images(images_CT, -2.55*np.ones(images_CT.shape),[1,min_images,min_rows,min_cols,images.shape[-1]])
            if self.noise != 0:
                out_images_CT += self.noise * np.random.normal(loc=0.0, scale=1.0, size=out_images_CT.shape)
            return out_images_CT, out_images[...,1][...,None] # Take the secondary image, since we are doing primary -> secondary
        else:
            return out_images,out_field
    def __getitem__(self,index):
        if self.get_CT_images:
            if index in self.data:
                data_out, masked_image = self.data[index]
            else:
                images, mask = self.load_images(self.paths[index])
                images = [np.expand_dims(images[..., 0], axis=-1), np.expand_dims(images[..., 1], axis=-1)]
                masked_image = copy.deepcopy(images[1])
                data_out = images + [mask], masked_image
                if self.save_and_load:
                    self.data[index] = data_out, masked_image
                # masked_image[mask==0] = images[1].min()
            return data_out, masked_image
        else:
            if index in self.data:
                images,field = self.data[index]
            else:
                images, field = self.load_images(self.paths[index]) #images, field =
                if not self.use_distance_map:
                    images = [np.expand_dims(images[..., 0], axis=-1), np.expand_dims(images[..., 1], axis=-1)]
                else:
                    dep = images.shape[-1]//2
                    images = [images[...,:dep],images[...,dep:]]
                if self.save_and_load:
                    self.data[index] = images,field
            # field[np.tile(images[0], 3) == 0] = 0  # Mask output
            if self.polar_coordinates:
                field = np.reshape(cartesian_to_polar(np.reshape(field, [np.prod(field.shape[:-1]), field.shape[-1]])),
                                   field.shape)
            return images, field

    def __len__(self):
        len(self.paths)
        return len(self.paths)


class Train_Data_Generator3D(Data_Generator_Class):

    def __init__(self, batch_size=1, perturbations=None, whole_patient=True,verbose=False,
                 noise=None,prediction_class=None,output_size = None,save_and_reload=True,
                 data_paths=None, shuffle=False, all_for_one=False, write_predictions = False,is_auto_encoder=False,
                 num_patients=1,is_test_set=False, expansion=0, mean_val=None, std_val=None,
                 max_image_size=999,skip_correction=False, normalize_to_value=None, wanted_indexes=None, z_images=32,
                 image_processors=None):
        '''
        :param batch_size:
        :param perturbations:
        :param whole_patient:
        :param verbose:
        :param prediction_class:
        :param output_size:
        :param data_paths:
        :param shuffle:
        :param all_for_one:
        :param write_predictions:
        :param is_auto_encoder:
        :param num_patients:
        :param is_test_set:
        :param expansion:
        :param clip:
        :param max_image_size:
        :param skip_correction:
        :param normalize_to_value:
        :param wanted_indexes:
        :param z_images:
        '''
        KeyError('Use Data_Generator_Class, not 3D or 2D Model')
        super().__init__(whole_patient=whole_patient, save_and_reload=save_and_reload,
                         data_paths=data_paths, num_patients=num_patients,is_test_set=is_test_set, expansion=expansion,
                         shuffle=shuffle, batch_size=batch_size, all_for_one=all_for_one, wanted_indexes=wanted_indexes,
                         image_processors=image_processors)
        self.perturbations = perturbations
        self.is_test_set = is_test_set
        self.index_data = {}
        self.loaded_model = None
        self.output_size = output_size
        self.is_auto_encoder = is_auto_encoder
        self.max_image_size = max_image_size
        self.verbose = verbose
        self.write_predictions = write_predictions
        self.prediction_class = prediction_class
        if mean_val is not None or std_val is not None:
            raise KeyError('Use Normalize_Images in the Image_Processors module instead of mean_val or std_val!')
        if noise is not None:
            raise KeyError('Use Add_Noise_To_Images in the Image_Processors module instead of noise!')
        self.normalize_to_value = normalize_to_value
        self.skip_correction = skip_correction
        if not data_paths:
            raise NameError('No training paths defined')
        self.all_for_one = all_for_one
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.z_images = z_images
        if self.is_auto_encoder:
            self.train_dataset_reader_no_changes = copy.deepcopy(self.train_dataset_reader)
        self.get_image_lists()

    def __getitem__(self, index):
        train_images_out, train_annotations_out = self.train_dataset_reader.load_images(index,self.z_images)  # how many images to pull
        return train_images_out, train_annotations_out


class Image_Clipping_and_Padding(Sequence):
    def __init__(self, layers_dict, generator, return_mask=False, liver_box=False, mask_image=False,
                 bounding_box_expansion=(5,10,10), threshold_value=None, remove_liver_layer=False):
        '''
        :param layers_dict: Dictionary of layers for model, Layer_0, Layer_1, Base, etc.
        :param generator: a Data generator
        :param return_mask: return the mask used for masking input Data
        :param liver_box: use a bounding box
        :param mask_image: mask the image Data
        :param bounding_box_expansion: z, x, y expansions for bounding box
        '''
        self.bounding_box_expansion = bounding_box_expansion
        self.remove_liver_layer = remove_liver_layer
        self.mask_image = mask_image
        self.patient_dict = {}
        self.liver_box = liver_box
        self.threshold_value = threshold_value
        self.generator = generator
        power_val_z, power_val_x, power_val_y = (1,1,1)
        pool_base = 2
        for layer in layers_dict:
            if layer == 'Base':
                continue
            if 'Pooling' in layers_dict[layer]:
                if 'Encoding' not in layers_dict[layer]['Pooling']:
                    pooling = layers_dict[layer]['Pooling']
                else:
                    pooling = [pool_base for _ in range(3)]
            else:
                pooling = (1, 1, 1)
            power_val_z *= pooling[0]
            power_val_x *= pooling[1]
            power_val_y *= pooling[2]
        self.return_mask = return_mask
        self.power_val_z, self.power_val_x, self.power_val_y = power_val_z, power_val_x, power_val_y

    def __getitem__(self, item):
        x,y = self.generator.__getitem__(item) # Have perturbations being applied, need to keep loading
        if self.liver_box:
            liver = np.argmax(y,axis=-1)
            z_start, z_stop, r_start, r_stop, c_start, c_stop = get_bounding_box_indexes(liver)
            z_start = max([0,z_start-self.bounding_box_expansion[0]])
            z_stop = min([z_stop+self.bounding_box_expansion[0],x.shape[1]])
            r_start = max([0,r_start-self.bounding_box_expansion[1]])
            r_stop = min([512,r_stop+self.bounding_box_expansion[1]])
            c_start = max([0,c_start-self.bounding_box_expansion[2]])
            c_stop = min([512,c_stop+self.bounding_box_expansion[2]])
        else:
            z_start = 0
            z_stop = x.shape[1]
            r_start = 0
            r_stop = x.shape[2]
            c_start = 0
            c_stop = x.shape[3]
        z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start
        remainder_z, remainder_r, remainder_c = self.power_val_z - z_total % self.power_val_z if z_total % self.power_val_z != 0 else 0, \
                                                self.power_val_x - r_total % self.power_val_x if r_total % self.power_val_x != 0 else 0, \
                                                self.power_val_y - c_total % self.power_val_y if c_total % self.power_val_y != 0 else 0
        min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r, c_total + remainder_c
        if self.threshold_value is None:
            threshold_val = np.min(x)
        else:
            threshold_val = self.threshold_value
        out_images = np.ones([1,min_images,min_rows,min_cols,x.shape[-1]],dtype=x.dtype)*threshold_val
        out_annotations = np.zeros([1, min_images, min_rows, min_cols, y.shape[-1]], dtype=y.dtype)
        out_annotations[..., 0] = 1
        out_images[:,0:z_stop-z_start,:r_stop-r_start,:c_stop-c_start,:] = x[:,z_start:z_stop,r_start:r_stop,c_start:c_stop,:]
        out_annotations[:,0:z_stop-z_start,:r_stop-r_start,:c_stop-c_start,:] = y[:,z_start:z_stop,r_start:r_stop,c_start:c_stop,:]
        if self.mask_image:
            out_images[out_annotations[...,0] == 1] = self.threshold_value
        if self.return_mask:
            mask = np.sum(out_annotations[...,1:],axis=-1)[...,None]
            if self.remove_liver_layer:  # In future predictions we do not want to predict liver, so toss it out
                out_annotations = out_annotations[..., (0, 2)]
                out_annotations[...,0] = 1-np.sum(out_annotations[...,1:],axis=-1)
            mask = np.repeat(mask,out_annotations.shape[-1],axis=-1)
            sum_vals = np.zeros(mask.shape)
            sum_vals[...,0] = 1 - mask[...,0]
            return [out_images,mask, sum_vals], out_annotations
        if self.remove_liver_layer:
            out_annotations = out_annotations[...,(0,2)]
        return out_images, out_annotations

    def __len__(self):
        return len(self.generator)

    def on_epoch_end(self):
        self.generator.on_epoch_end()
        return None


class Generator_From_Predictions(Sequence):
    def __init__(self, generator=None, model=None, is_validation=False):
        self.generator = generator
        self.model = model
        self.data_dict = {}
        self.is_validation = is_validation


    def __getitem__(self, index):
        if self.is_validation:
            if index not in self.data_dict:
                x, y = self.generator.__getitem__(index)
                x = self.model.predict(x)
                self.data_dict[index] = x,y
            x,y = self.data_dict[index]
        else:
            x, y = self.generator.__getitem__(index)
            x = self.model.predict(x)
        return x,y

    def __len__(self):
        # len(self.generator)
        return len(self.generator)

    def on_epoch_end(self):
        self.generator.on_epoch_end()


class Predict_From_Trained_Model(object):
    def __init__(self,model_path,Bilinear_model=None): #gpu=0,graph1=Graph(),session1=Session(config=ConfigProto(log_device_placement=False)),
        print('loaded vgg model ' + model_path)
        self.vgg_model_base = load_model(model_path, custom_objects={'BilinearUpsampling':Bilinear_model,'dice_coef_3D':dice_coef_3D})
        print('finished loading')

    def predict(self,images):
        return self.vgg_model_base.predict(images)


if __name__ == '__main__':
    xxx = 1
