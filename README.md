## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!

### This code builds off of images found here https://github.com/brianmanderson/Make_Single_Images

A collection of the data generators which I created

To get 3D images use
        paths = [os.path.join(base_path, 'Train', 'Contrast', 'Single_Images3D'), os.path.join(base_path, 'Train', 'Non_Contrast','Single_Images3D')]
        train_generator = Train_Data_Generator(batch_size=1,
                                             whole_patient=True, shuffle=False,
                                             image_size=512, num_patients=1,z_images=z_images,
                                             data_paths=paths, is_CT=True, num_classes=num_classes,clip=clip,perturbations=perturbations,
                                             flatten=False, mean_val=mean_val, std_val=std_val, expansion=expansion, three_layer=False,
                                             is_test_set=True,all_for_one=False,verbose=False,output_size=desired_output,
                                             noise=0.05, write_predictions=False, max_image_size=max_images, normalize_to_value=1) 
