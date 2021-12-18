# Import Libraries

import os
from path import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

d = Path(__file__).parent.parent.parent.parent
Path(d).chdir()

class ModifyBuild:

    def __init__(self, data):

        self.data = data

    def create_paths(self):

        for dirpath, dirnames, filenames in os.walk(data):
            print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

        train_dir = os.path.join(data + "/Training")
        print('Training Dir:', train_dir)
        test_dir = os.path.join(data + "/Training")
        print('Test Dir:', test_dir)

        # Create augmented train and test data generators and rescale the data
        train_datagen_augmented = ImageDataGenerator(rescale=1./255,
                                                     rotation_range=45,
                                                     width_shift_range=0.2,
                                                     height_shift_range=0.2,
                                                     shear_range=0.2,
                                                     zoom_range=0.2,
                                                     horizontal_flip=True,
                                                     fill_mode='nearest')

        test_datagen = ImageDataGenerator(rescale=1./255)

        # Load the image data
        train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                           target_size=(128, 128),
                                                                           class_mode="categorical",
                                                                           batch_size=200,
                                                                           shuffle=True,
                                                                           color_mode="grayscale",
                                                                           seed=44)

        test_data = test_datagen.flow_from_directory(test_dir,
                                                     target_size=(128,128),
                                                     class_mode="categorical",
                                                     batch_size = 200,
                                                     color_mode="grayscale",
                                                     seed=44)

        return train_data_augmented, test_data


data = os.path.join(d, 'project/volume/data/raw/archive')
modify = ModifyBuild(data)
train, test = modify.create_paths()
