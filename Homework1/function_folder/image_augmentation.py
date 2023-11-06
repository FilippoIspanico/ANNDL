from function_folder.librarys import *

# Build the neural network layer by layer
#input_layer = tfkl.Input(shape=input_shape, name='Input')

#preprocessing = preprocessing(input_layer)

#x = tfkl.Conv2D(filters=32, kernel_size=3, padding='same', name='conv0')(preprocessing)


def image_augmenter():
    preprocessing = tf.keras.Sequential([
    tfkl.RandomFlip("horizontal"),
    tfkl.RandomTranslation(0.2,0.2),
    tfkl.RandomRotation(0.2),
    tfkl.RandomZoom(0.2),
    tfkl.RandomBrightness(0.5, value_range=(0,1)),
    tfkl.RandomContrast(0.75),
    ], name='preprocessing')
    return preprocessing

