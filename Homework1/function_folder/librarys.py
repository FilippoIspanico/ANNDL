def librarys_upload():
    seed = 42
    import os


    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    import numpy as np
    np.random.seed(seed)

    import logging

    import random
    random.seed(seed)
    # Import other libraries
    import cv2
    from tensorflow.keras.applications.mobilenet import preprocess_input
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    # others
    import numpy
    import pandas
    from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
    import matplotlib.pyplot as plt
    from PIL import Image