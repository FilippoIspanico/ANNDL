from function_folder.librarys import *
def MobileNetV2():
    mobile = tfk.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    pooling='avg',
    )
    return mobile;
