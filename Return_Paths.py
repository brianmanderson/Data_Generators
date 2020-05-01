import os

class Path_Return_Class(object):
    def __init__(self, base_path, morfeus_path, save_model=True, is_keras_model=True):
        self.base_path = base_path
        self.morfeus_path = morfeus_path
        self.save_model = save_model
        self.is_keras_model = is_keras_model
        for path in [base_path,morfeus_path]:
            assert os.path.exists(path), print(path + ' path does not exist')

    def define_model_things(self, model_name, *args, **kwargs):
        '''
        :param args: Send in a list of arguments, I recommend Model_Name, Architecture, Image_Characteristics, Hyper_parameters
        :return:
        '''
        self.model_distinctions = [model_name]
        for key in kwargs:
            self.model_distinctions.append('{}_{}'.format(kwargs[key],key))
        for arg in args:
            if type(arg) is list:
                for i in arg:
                    self.model_distinctions.append(i)
            else:
                self.model_distinctions.append(arg)
        self.create_out_paths()

    def create_out_paths(self):
        base_out_path = os.path.join(self.base_path, 'Keras', self.model_distinctions[0])
        tensorboard_output = os.path.join(self.morfeus_path, 'Keras', self.model_distinctions[0], 'Tensorboard')
        model_path_out = os.path.join(base_out_path,'Models')

        for i in self.model_distinctions[1:]:
            tensorboard_output = os.path.join(tensorboard_output,i)
            model_path_out = os.path.join(model_path_out,i)

        if self.save_model:
            self.make_paths(model_path_out,tensorboard_output)
        else:
            self.make_paths(tensorboard_output)
        if self.is_keras_model:
            model_path_out = os.path.join(model_path_out,'weights-improvement-best.hdf5')
        self.model_path_out = model_path_out
        self.tensorboard_path_out = tensorboard_output
        return None

    def make_paths(self, *args):
        for i in args:
            if not os.path.exists(i):
                os.makedirs(i)
        return None


def find_base_dir():
    base_path = '.'
    for _ in range(20):
        if 'Morfeus' in os.listdir(base_path):
            break
        else:
            base_path = os.path.join(base_path,'..')
    return base_path

def find_raid_dir():
    base_path = os.path.join('.')
    for _ in range(20):
        if 'raid' in os.listdir(base_path):
            break
        else:
            base_path = os.path.join(base_path,'..')
    return base_path


if __name__ == '__main__':
    pass
