## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!

### This code builds off of images found here https://github.com/brianmanderson/Make_Single_Images

A collection of the data generators which I created

To get 3D images use

    from Image_Array_And_Mask_From_Dicom import DicomImagestoData

    paths = [os.path.join(base_path, 'Train', 'Contrast', 'Single_Images3D')]

    test_generator = Train_Data_Generator(batch_size=batch_size, num_of_classes=num_classes,z_images=z_images,data_paths=paths,normalize_to_value=1,mean_val=80,std_val=42)
