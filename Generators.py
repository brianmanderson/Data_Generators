from keras.utils import Sequence, np_utils
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.measure import block_reduce
import cv2, os, copy, glob, pickle
import numpy as np
from scipy.ndimage import interpolation

def plot_scroll_Image(x):
    '''
    :param x: input to view of form [rows, columns, # images]
    :return:
    '''
    if x.dtype not in ['float32','float64']:
        x = copy.deepcopy(x).astype('float32')
    if len(x.shape) > 3:
        x = np.squeeze(x)
    if len(x.shape) == 3:
        if x.shape[0] != x.shape[1]:
            x = np.transpose(x,[1,2,0])
        elif x.shape[0] == x.shape[2]:
            x = np.transpose(x, [1, 2, 0])
    fig, ax = plt.subplots(1, 1)
    if len(x.shape) == 2:
        x = np.expand_dims(x,axis=0)
    tracker = IndexTracker(ax, x)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return fig,tracker
    #Image is input in the form of [#images,512,512,#channels]

def load_obj(path):
    if path.find('.pkl') == -1:
        path += '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        out = {}
        return out

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = np.where(self.X != 0)[-1]
        if len(self.ind) > 0:
            self.ind = self.ind[len(self.ind)//2]
        else:
            self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap='gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


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


class image_loader(object):
    def __init__(self,image_size=512,perturbations=None, three_channel=False, by_patient=False,
                 resize_class=None, random_start=True, final_steps=None, all_images=False):
        self.patient_dict_indexes = {}
        self.image_dictionary = {}
        self.resize_class = resize_class
        self.random_start = random_start
        self.by_patient = by_patient
        self.image_size = image_size
        if perturbations:
            self.perturbations = perturbations
            self.pertubartion_class = Pertubartion_Class(perturbations, [image_size, image_size])
        self.perturbations = perturbations
        self.three_channel = three_channel
        self.final_steps = final_steps
        self.all_images = all_images

    def convert_image_size(self, images, annotations, image_size):
        if images.shape[1] != image_size:
            difference_1 = image_size - images.shape[1]
            if difference_1 > 0:
                images = np.concatenate((images, images[:, :int(difference_1 / 2), :]),
                                        axis=1)
                images = np.concatenate((images[:, -int(difference_1 / 2):, :], images),
                                        axis=1)
                annotations = np.concatenate((annotations, annotations[:, :int(difference_1 / 2), :]),
                                        axis=1)
                annotations = np.concatenate((annotations[:, -int(difference_1 / 2):, :], annotations),
                                        axis=1)
            elif difference_1 < 0:
                images = images[:, :int(difference_1 / 2), :]
                images = images[:, abs(int(difference_1 / 2)):, :]
                annotations = annotations[:, :int(difference_1 / 2), :]
                annotations = annotations[:, abs(int(difference_1 / 2)):, :]
        if images.shape[2] != image_size:
            difference_2 = image_size - images.shape[2]
            if difference_2 > 0:
                images = np.concatenate((images, images[:, :, :int(difference_2 / 2)]),
                                        axis=2)
                images = np.concatenate((images[:, :, -int(difference_2 / 2):], images),
                                        axis=2)
                annotations = np.concatenate((annotations, annotations[:, :, :int(difference_2 / 2)]),
                                        axis=2)
                annotations = np.concatenate((annotations[:, :, -int(difference_2 / 2):], annotations),
                                        axis=2)
            elif difference_2 < 0:
                images = images[:, :, :int(difference_2 / 2)]
                images = images[:, :, abs(int(difference_2 / 2)):]
                annotations = annotations[:, :, :int(difference_2 / 2)]
                annotations = annotations[:, :, abs(int(difference_2 / 2)):]
        return images, annotations

    def give_resized_images(self, images_temp, annotations_temp):
        if images_temp.shape[0] != 1:
            images_temp = np.expand_dims(images_temp, axis=0)
            annotations_temp = np.expand_dims(annotations_temp, axis=0)
        if images_temp.shape[-1] != 1:
            images_temp = np.expand_dims(images_temp, axis=-1)
            annotations_temp = np.expand_dims(annotations_temp, axis=-1)
        images_temp, annotations_temp = self.convert_image_size(images_temp, annotations_temp, 512)
        images_temp = self.resize_class.resize_images(images_temp)
        annotations_temp = self.resize_class.resize_images(annotations_temp)
        images_temp = images_temp[:, :, :, 0]
        annotations_temp = annotations_temp[:, :, :, 0]
        return images_temp, annotations_temp
    def load_image(self, batch_size=0, image_names=None):
        images, annotations = np.ones([batch_size, self.image_size, self.image_size],dtype='float32')*-1000, \
                              np.zeros([batch_size, self.image_size, self.image_size],dtype='int8')
        add = 0
        start = 0
        finish = len(image_names)
        if not self.random_start:
            batch_size = finish
        description = ''
        if self.by_patient and batch_size != 0 and not self.all_images:
            file = image_names[0]
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
                slice_num = int(broken_up[-1][:-4])
                description = ''
                for i in broken_up[:-1]:
                    description += i + '_'
                description = description[:-1]
        if len(image_names) > batch_size:
            print('looking here')
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
                new_file = file.replace(slice_num+'.npy',str(values[0])+'.npy')
                if os.path.exists(new_file):
                    start = image_names.index(new_file)
                    finish = min([int(start+batch_size),len(image_names)])

        make_changes = True
        for index, i in enumerate(range(start,finish)):
            if i < 0 or i > len(image_names):
                print('start:' + str(start) + 'total images: ' + str(len(image_names)) + '_i:' + str(i))
            if image_names[i] not in self.image_dictionary:
                image_name = image_names[i]
                if image_name.find('_image.npy') == -1:
                    data = np.load(image_name)
                    images_temp = data[0, :, :][None,...]
                    annotations_temp = data[1, :, :][None,...]
                else:
                    images_temp = np.load(image_name)
                    annotations_temp = np.load(image_name.replace('_image.npy','_annotation.npy'))
                if (make_changes or not self.by_patient) or (images_temp.shape[1] != self.image_size or images_temp.shape[2] != self.image_size):
                    if images_temp.shape[1] > self.image_size and images_temp.shape[2] > self.image_size:
                        images_temp = block_reduce(images_temp[0,...], (2, 2), np.average).astype('float32')[None,...]
                        annotations_temp = block_reduce(annotations_temp[0,...].astype('int'), (2, 2), np.max).astype('int8')[None,...]
                    elif images_temp.shape[1] <= self.image_size / 2 or images_temp.shape[2] <= self.image_size / 2:
                        images_temp, annotations_temp = self.give_resized_images(images_temp, annotations_temp)
                    if images_temp.shape[0] != 1:
                        images_temp = images_temp[None,...]
                        annotations_temp = annotations_temp[None,...]
                    images_temp, annotations_temp = self.convert_image_size(images_temp, annotations_temp,
                                                                            self.image_size)
                self.image_dictionary[image_names[i]] = copy.deepcopy([images_temp.astype('float32'), annotations_temp])
            else:
                images_temp, annotations_temp = copy.deepcopy(self.image_dictionary[image_names[i]])
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
        # dif_images, dif_rows, dif_cols = int(output_size[1]-images.shape[1]), int(output_size[2] - images.shape[2]), \
        #                                  int(output_size[3] - images.shape[3])
        # dif_images, dif_rows, dif_cols = max([0, int(dif_images / 2 - 1)]), max([0, int(dif_rows / 2 - 1)]), max(
        #     [0, int(dif_cols / 2 - 1)])
        # if len(images.shape) == 4:
        #     val_differences = [[0, 0], [dif_images, dif_images], [dif_rows, dif_rows], [dif_cols, dif_cols], [0, 0]]
        # else:
        #     val_differences = [[dif_images, dif_images], [dif_rows, dif_rows], [dif_cols, dif_cols], [0, 0]]
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

class Data_Set_Reader(image_loader):
    def __init__(self,path=None,image_size=512,perturbations=None, three_channel=False, by_patient=False,verbose=True,
                 num_patients=1, resize_class=None,is_test_set=False, random_start=True,
                 expansion = 0,final_steps=None, shuffle_images=True, wanted_indexes=None):
        '''
        :param path:
        :param image_size:
        :param perturbations:
        :param three_channel:
        :param by_patient:
        :param verbose:
        :param num_patients:
        :param resize_class:
        :param is_test_set:
        :param random_start:
        :param expansion:
        :param final_steps:
        :param shuffle_images:
        :param wanted_indexes: a tuple of indexes wanted (2) will pull disease only if 1 is liver
        '''
        super().__init__(image_size=image_size,perturbations=perturbations, three_channel=three_channel,
                              by_patient=by_patient,resize_class=resize_class, random_start=random_start,
                              final_steps=final_steps, all_images=is_test_set)
        self.wanted_indexes = wanted_indexes
        self.shuffle_images = shuffle_images
        self.expansion = expansion
        if resize_class:
            self.resize_class = resize_class
        self.start_stop_dict = {}
        self.num_patients = num_patients
        self.patient_dict = {}
        self.verbose = verbose
        self.file_batches = []
        self.by_patient = by_patient
        self.data_path = path
        self.file_list = []
        if path:
            if 'descriptions_start_and_stop.pkl' in os.listdir(path):
                self.start_stop_dict = load_obj(os.path.join(path, 'descriptions_start_and_stop.pkl'))
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.find('.npy') != -1 and file.find('_annotation.npy') == -1:
                        self.file_list.append(os.path.join(path, file))
            elif self.verbose:
                print(path)
                print('Wrong path')
        self.load_file_list = copy.deepcopy(self.file_list)
        self.make_patient_list()
        if self.by_patient:
            self.prep_batches()

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
            if broken_up[-1].find('image') == 0: # Making everything back compatible with the new style of passing data
                slice_num = int(broken_up[-2])
                description = ''
                for i in broken_up[:-2]:
                    description += i + '_'
                description = description[:-1]
            else:
                slice_num = int(broken_up[-1][:-4])
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
                if np.max(np.abs(slice_num-values) < self.expansion):
                    files[slice_num] = file
            elif slice_num >= start and slice_num <= stop:
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
        if not self.by_patient:
            self.file_list = []
            for pat in self.patient_dict.keys():
                self.file_list += self.patient_dict[pat]

    def distribute_patients(self):
        self.file_batches = []
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
        self.load_file_list = copy.deepcopy(self.file_list)
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
        images += min_val
        output_image = images
        output_annotation = annotations
        return output_image, output_annotation


class Train_Data_Generator2D(Sequence):
    def __init__(self, image_size=512, batch_size=5, perturbations=None, num_of_classes=2, data_paths=None,clip=0,expansion=0,
                 whole_patient=False, shuffle=False, flatten=False, noise=0.0, normalize_to_255=False,z_images=16,
                 all_for_one=False, three_channel=True, using_perturb_engine=False,on_VGG=False,normalize_to_value=None,
                 resize_class=None,add_filename_extension=True, is_test_set=False, reduced_interest=False, mean_val=0, std_val=1):
        self.z_images = z_images
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
                    self.image_list.append(self.training_models[key].file_batches[i])
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
            annotations = np_utils.to_categorical(annotations,self.num_of_classes)
        if self.flatten:
            class_weights_dict = {0:1,1:20}
            class_weights = np.ones([annotations.shape[0],annotations.shape[1],annotations.shape[2]])
            for i in range(self.num_of_classes):
                class_weights[annotations[:,:,:,i] == 1] = class_weights_dict[i]
            annotations = np.reshape(annotations,
                                        [train_images.shape[0],self.image_size*self.image_size*self.num_of_classes])
            class_weights = np.reshape(class_weights,[train_images.shape[0],self.image_size*self.image_size,1])
            return train_images, annotations, class_weights
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
        return int(len(self.image_list))

    def on_epoch_end(self):
        self.get_image_lists()

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
    return min_z_s, int(max_z_s + 1), min_r_s, int(max_r_s + 1), min_c_s, int(max_c_s + 1)

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


class Train_DVF_Generator(Sequence):
    def __init__(self, path, save_and_load = False, batch_size = 1, split=False, pool_base = 2, layers=1, reduce=False,
                 mask_images=False,multi_pool=False,polar_coordinates=False,distance_map=False,min_rows_cols = None,is_2D=False,
                 flatten=False, get_CT_images=False, min_val=-1000, max_val=300, mean_val=100, std_val=40, perturbations=None,noise=0,layers_dict=None,output_size=None):
        '''
        :param path: Path to data
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
            print('Saving and reloading data')
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

class Train_Data_Generator_class(Sequence):

    def __init__(self, image_size=512, perturbations=None, three_channel=True,whole_patient=True,num_of_classes=2,wanted_indexes=None,
                 data_paths=None, num_patients=1,is_test_set=False, expansion=0,shuffle=False, batch_size=1, all_for_one=False):
        '''
        :param image_size: Image size
        :param three_layer: make three layer
        :param whole_patient: want whole patient
        :param data_paths: data paths
        :param only_valid:
        :param num_patients:
        :param is_test_set:
        :param expansion:
        :param normalize:
        :param center_mask:
        :param final_steps:
        :param save_and_load:
        '''
        self.max_patients = np.inf
        self.image_size = image_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.all_for_one = all_for_one
        self.num_of_classes = num_of_classes
        self.whole_patient = whole_patient
        self.is_auto_encoder = False # This can change in sub classes
        self.wanted_indexes = wanted_indexes
        self.train_dataset_reader = Data_Set_Reader(perturbations=perturbations,
                                                    image_size=image_size, three_channel=three_channel,
                                                    by_patient=whole_patient,
                                                    num_patients=num_patients, is_test_set=is_test_set,
                                                    expansion=expansion,
                                                    final_steps=None, verbose=False, wanted_indexes=wanted_indexes)
        self.training_models = self.get_training_models(data_paths,is_test_set,whole_patient,num_patients,expansion, wanted_indexes)

    def get_training_models(self, data_paths, is_test_set, whole_patient, num_patients, expansion, wanted_indexes):
        models = {}
        for path in data_paths:
            if path.find('Single_Images3D') == -1:
                path = os.path.join(path,'Single_Images3D') #Make them all 3D
            if len(os.listdir(path)) == 0:
                print('Nothin in data path:' + path)
            models[path] = Data_Set_Reader(
                path=path, by_patient=whole_patient,  num_patients=num_patients,
                is_test_set=is_test_set, expansion=expansion, wanted_indexes=wanted_indexes) #Always 1
            self.train_dataset_reader.patient_dict_indexes.update(models[path].patient_dict_indexes)
        return models


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
                    self.image_list.append(self.training_models[key].file_batches[i])
        self.train_dataset_reader.file_batches = self.image_list
        if self.is_auto_encoder:
            self.train_dataset_reader_no_changes.file_batches = self.image_list

    def __getitem__(self,index):
        pass

    def __len__(self):
        num = min([self.max_patients,int(len(self.image_list))])
        return num

    def on_epoch_end(self):
        self.get_image_lists()

    def return_corrected_image(self, train_images_full_size, train_annotations):
        if self.image_size != train_images_full_size.shape[1]:
            train_images = block_reduce(train_images_full_size, (1, 2, 2, 1), np.average)
        else:
            train_images = train_images_full_size

        under_review = True
        if train_images.shape[0] != self.batch_size and not under_review:
            i = 0
            while train_images.shape[0] < self.batch_size:
                if i == 0:
                    train_images = np.concatenate((train_images, np.expand_dims(train_images[0, :, :, :], axis=0)),
                                                  axis=0)
                    train_annotations = np.concatenate(
                        (train_annotations, np.expand_dims(train_annotations[0, :, :, :], axis=0)), axis=0)
                    i = 1
                elif i == 1:
                    train_images = np.concatenate((np.expand_dims(train_images[-1, :, :, :], axis=0), train_images),
                                                  axis=0)
                    train_annotations = np.concatenate(
                        (np.expand_dims(train_annotations[-1, :, :, :], axis=0), train_annotations), axis=0)
                    i = 0
            while train_images.shape[0] > self.batch_size:
                if i == 0:
                    train_images = train_images[1:, :, :, :]
                    train_annotations = train_annotations[1:, :, :, :]
                    i = 1
                elif i == 1:
                    train_images = train_images[:-1, :, :, :]
                    train_annotations = train_annotations[:-1, :, :, :]
                    i = 0
        if np.max(train_annotations) == 1:
            xxx = 1
        train_annotations = np_utils.to_categorical(train_annotations, self.num_of_classes)
        if self.wanted_indexes:
            train_annotations = train_annotations[...,self.wanted_indexes]
        if self.whole_patient:
            train_images = np.expand_dims(train_images, axis=0)
            train_annotations = np.expand_dims(train_annotations, axis=0)
        return train_images, train_annotations


class Train_Data_Generator3D(Train_Data_Generator_class):

    def __init__(self, image_size=512, batch_size=1, perturbations=None, three_layer=True,whole_patient=True,verbose=False,
                 num_classes=2, flatten=False,noise=0.0,prediction_class=None,output_size = None,
                 data_paths=None, shuffle=False, all_for_one=False, write_predictions = False,is_auto_encoder=False,
                 num_patients=1,is_test_set=False, expansion=0, clip=0,mean_val=0, std_val=1,
                 max_image_size=999,skip_correction=False, normalize_to_value=None, wanted_indexes=None, z_images=32):
        super().__init__(image_size=image_size, perturbations=perturbations, three_channel=three_layer,whole_patient=whole_patient, num_of_classes=num_classes,
                 data_paths=data_paths, num_patients=num_patients,is_test_set=is_test_set, expansion=expansion,shuffle=shuffle, batch_size=batch_size, all_for_one=all_for_one, wanted_indexes=wanted_indexes)
        '''
        :param image_size: Size of the image that you want as output, recommend 512 or 256
        :param batch_size: Number of batches, usually stuck at 1 unless specify generator.random_start = False outside of this
        :param three_layer: Make this a 3 channel output
        :param whole_patient: Do you want the whole patient
        :param vgg_model: Use the vgg model as pre-prediction? (Out dated)
        :param use_vgg: Out dated
        :param num_classes: Number of output classes
        :param flatten: Flatten the data
        :param data_paths: Paths to data
        :param only_valid: Only give valid images
        :param shuffle: True/False
        :param all_for_one:
        :param use_arg_max:
        :param num_patients:
        :param sub_sample:
        :param is_test_set:
        :param additions:
        :param change_pixel_values:
        :param resolutions:
        :param expansion:
        :param clip:
        :param normalize:
        :param skip_correction: Skip the final shaping, etc.
        :param save_and_load: Save the data and load each time
        :param wanted_indexes: Tuple of desired indexes, (1) gives liver
        '''
        self.perturbations = perturbations
        self.loaded_model = None
        self.output_size = output_size
        self.is_auto_encoder = is_auto_encoder
        self.max_image_size = max_image_size
        self.verbose = verbose
        self.write_predictions = write_predictions
        self.prediction_class = prediction_class
        self.mean_val, self.std_val = mean_val, std_val
        self.normalize_to_value = normalize_to_value
        self.skip_correction = skip_correction
        if type(clip) == int:
            clip = [clip for _ in range(4)]
        self.clip = clip
        if not data_paths:
            raise NameError('No training paths defined')
        self.all_for_one = all_for_one
        self.shuffle = shuffle
        self.flatten = flatten
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.z_images = z_images
        self.noise = noise
        if self.is_auto_encoder:
            self.train_dataset_reader_no_changes = copy.deepcopy(self.train_dataset_reader)
        self.get_image_lists()

    def __getitem__(self, index):
        file_name = self.train_dataset_reader.file_batches[index][0][0]
        non_noisy_image = None

        train_images_full_size, train_annotations_full_size = self.train_dataset_reader.load_images(index,self.z_images) # how many images to pull
        if not self.skip_correction:
            if len(train_images_full_size.shape) == 5:
                train_images_out, train_annotations_out = np.empty(train_images_full_size.shape), np.empty(train_images_full_size.shape[:-1] + (self.num_of_classes,))
                for i in range(train_images_full_size.shape[0]):
                    train_images_out[i], train_annotations_out[i] = self.return_corrected_image(train_images_full_size[i,...],
                                                                                                train_annotations_full_size[i, ...])
            else:
                train_images_out, train_annotations_out = self.return_corrected_image(train_images_full_size, train_annotations_full_size)
            if self.output_size:
                train_images_out, train_annotations_out = pull_cube_from_image(train_images_out,
                                                                               train_annotations_out,
                                                                               samples=self.output_size[0],
                                                                               desired_size=self.output_size[1:])
            if self.is_auto_encoder:
                non_noisy_image = copy.deepcopy(train_images_out)
                if self.mean_val != 0 or self.std_val != 1:
                    non_noisy_image = (non_noisy_image - self.mean_val) / self.std_val
                    non_noisy_image[non_noisy_image > 3.55] = 3.55
                    non_noisy_image[non_noisy_image < -3.55] = -3.55
            if self.mean_val != 0 or self.std_val != 1 or train_images_out.max() == 1:
                train_images_out = (train_images_out - self.mean_val) / self.std_val
                if self.noise != 0:
                    train_images_out += self.noise * np.random.normal(loc=0.0, scale=1.0, size=train_images_out.shape)
                train_images_out[train_images_out > 3.55] = 3.55
                train_images_out[train_images_out < -3.55] = -3.55
                # train_images_out[train_annotations_out[..., -1] == 0] = 0 # Don't mask training images
                if self.normalize_to_value:
                    train_images_out = (train_images_out - -3.55) / (2*3.55) * self.normalize_to_value
                    # train_images_out[train_annotations_out[..., -1] == 0] = 0
                    if self.is_auto_encoder:
                        non_noisy_image = (non_noisy_image - -3.55) / (2*3.55) * self.normalize_to_value
                        non_noisy_image[train_annotations_out[..., -1] == 0] = 0
                        # non_noisy_image[train_annotations_out[..., -1] == 0] = 0 # Leave all pixels involved
                        if self.flatten:
                            train_images_out = train_images_out.reshape(train_images_out.shape[0],np.prod(train_images_out.shape[1:]))
                            train_annotations_out = train_annotations_out.reshape(train_annotations_out.shape[0],
                                                               np.prod(train_annotations_out.shape[1:]))
        else:
            train_images_out, train_annotations_out = train_images_full_size, train_annotations_full_size
        if max(self.clip) > 0:
            if self.is_auto_encoder:
                non_noisy_image = non_noisy_image[:, self.clip[0]:-self.clip[0], self.clip[1]:-self.clip[1],self.clip[2]:-self.clip[2], :]
                train_annotations_out = train_annotations_out[:, self.clip[0]:-self.clip[0], self.clip[1]:-self.clip[1], self.clip[2]:-self.clip[2], :]
        if self.is_auto_encoder:
            if self.loaded_model:
                train_images_out = self.loaded_model.predict([train_images_out, np.ones(train_annotations_out.shape[:-1])[...,None]])
            return [train_images_out, train_annotations_out[...,-1][...,None]], non_noisy_image
        if train_images_out.shape[:-1] != train_annotations_out.shape[:-1]:
            print(file_name)
            x,y = self.__getitem__(index)
        return train_images_out, train_annotations_out


if __name__ == '__main__':
    xxx = 1