from keras.preprocessing.image import ImageDataGenerator

# Data preprocessing 
# 1. Read pictures file
# 2. Decode Jpeg into RGB
# 3. Convert Rgb girds into tensors
# 4. Rescale pixel values into [0, 1]

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir  = '/home/filippo/Documents/PoliMi/ANNDL/shared_folder/ANNDL/datasets/cats_and_dogs_small/train'
validation_dir = '/home/filippo/Documents/PoliMi/ANNDL/shared_folder/ANNDL/datasets/cats_and_dogs_small/validation'



train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),  # we are resising the images to 150x150!
        batch_size=20, 
        class_mode='binary' # to impose that labels are binary, nedeed in orded to use binary crossentropy ...
)

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),  # we are resising the images to 150x150!
        batch_size=1, 
        class_mode='binary' # to impose that labels are binary, nedeed in orded to use binary crossentropy ...
    
)