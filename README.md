## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!

### This code builds off of images found here https://github.com/brianmanderson/Make_Single_Images

A collection of the data generators which I created

To get 3D images that pull all images from the patient about the mask [1] with 0 expansion

    from Base_Deeplearning_Code.Data_Generators.Generators import Data_Generator_Class
    from Base_Deeplearning_Code.Data_Generators.Image_Processors import *
    paths = [os.path.join(base_path, 'Train', 'Contrast', 'Single_Images3D')]
    
    # Load up any processors
    mean_val, std_val = 67, 36
    image_processors_train = [
                              Normalize_Images(mean_val=mean_val,std_val=std_val), 
                              Ensure_Image_Proportions(512, 512),
                              Annotations_To_Categorical(num_of_classes=2)
                              ]
    train_generator = Data_Generator_Class(by_patient=True,num_patients=1, whole_patient=True, shuffle=False,
                                           data_paths=paths, expansion=0, wanted_indexes=[1],
                                           image_processors=image_processors_train)
