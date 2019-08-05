## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!

### This code builds off of images found here https://github.com/brianmanderson/Make_Single_Images

A collection of the data generators which I created

To get 3D images use

    from Generators import Train_Data_Generator2D

    paths = [os.path.join(base_path, 'Train', 'Contrast', 'Single_Images3D')]

    test_generator = Train_Data_Generator2D(batch_size=5, num_of_classes=2,
                                      data_paths=paths_test_generator,normalize_to_value=1,expansion=10,
                                      is_CT=is_CT, mean_val=mean_val, std_val=std_val)
    Batch_size determines how many images are thrown in a batch together, z_images determines how many images are printed out
    if z_images > batch_size for 2D purposes, you will get blacked out images
    expansion asks how many images to include above and below segmented images
