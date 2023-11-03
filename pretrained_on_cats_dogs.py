from keras.applications import VGG16
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from plot_history import plot_history
train_dir ='/home/filippo/Documents/PoliMi/ANNDL/shared_folder/ANNDL/datasets/cats_and_dogs_small/train'
validation_dir ='/home/filippo/Documents/PoliMi/ANNDL/shared_folder/ANNDL/datasets/cats_and_dogs_small/validation'

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

print('# of trainable weights before freezing: ', len(model.trainable_weights))

conv_base.trainable = False
print('# of trainable weights after freezing: ', len(model.trainable_weights))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range= 0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )   

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

model.compile(loss = 'binary_crossentropy', optimizer=optimizers.RMSprop(lr = 2e-5), metrics=['acc'])

history = model.fit_generator(
    train_generator, 
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

plot_history(history=history, smoothing=True)

conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if(layer.name == 'block5_conv1'):
        set_trainable = True
    if(set_trainable):
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss = 'binary_crossentropy', optimizer=optimizers.RMSprop(lr = 1e-5), metrics=['acc'])

history = model.fit_generator(
    train_generator, 
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('cats_and_dogs_small_2.h5')