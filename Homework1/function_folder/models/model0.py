from function_folder.librarys import *
def MobileNetV2():
    mobile = tfk.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet",
    pooling='avg',
    )
    return mobile;

def VGG16_max():
    mobile = tfk.applications.VGG16(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet",
    pooling='avg',
    )
    return mobile;
