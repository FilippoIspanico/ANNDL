from keras import layers, models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
from keras.saving import load_model
from plot_history import plot_history

model = models.Sequential(); 
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())


model.compile(loss = 'binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1.e-4), metrics = ['acc'])




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
        batch_size=20, 
        class_mode='binary' # to impose that labels are binary, nedeed in orded to use binary crossentropy ...
    
)

################################################################### 
# 
#Training! 


history = model.fit_generator(
  train_generator,
  steps_per_epoch=100,
  epochs=20,
  validation_data=validation_generator,
  validation_steps=50
)

model.save('cats_and_dogs_small_1.h5')
#####################################################################

model = load_model('cats_and_dogs_small_1.h5')

plot_history(history)
