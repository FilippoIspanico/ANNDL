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


    train_generator = ImageDataGenerator(rescale = 1./255)
    train_generator = train_generator.flow(
            x=npzobj['data'],
            y=npzobj['labels'],
            batch_size=20,
            shuffle=True,
            sample_weight=None,
            seed=None,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            ignore_class_split=False,
            subset=None
        )
