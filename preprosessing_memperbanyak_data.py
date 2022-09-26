# import library
import os
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import img_as_ubyte
from skimage import util
from skimage import io
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

# augmentasi pergambar yang akan di perbanyak
def random_rotation(image_array: ndarray):
    random_degree = random.uniform(-25, 25)
    samples = expand_dims(image_array, 0)
    datagen = ImageDataGenerator(rotation_range=random_degree,horizontal_flip=True)
    it = datagen.flow(samples, batch_size=1)
    pyplot.subplot(330 + 1)
    batch = it.next()
    image = batch[0].astype('uint8')
    return image

# aktifkan jika diperlukan
# Brightness 
# def random_brightness(image_array: ndarray):
#     samples = expand_dims(image_array, 0)
#     datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
#     it = datagen.flow(samples, batch_size=1)
#     pyplot.subplot(330 + 1)
#     batch = it.next()
#     image = batch[0].astype('uint8')
#     return image
# zoom
# def random_zoom(image_array: ndarray):
#     random_degree = random.uniform(-25, 25)
#     samples = expand_dims(image_array, 0)
#     datagen = ImageDataGenerator(rotation_range=random_degree,zoom_range=[0.5,1.0])
#     it = datagen.flow(samples, batch_size=1)
#     pyplot.subplot(330 + 1)
#     batch = it.next()
#     image = batch[0].astype('uint8')
#     return image

available_transformations = {
	'rotate' : random_rotation,
	# 'brightness'  : random_brightness,
	# 'zoom' : random_zoom 
}


# folder gambar yang akan diperbanyak
path_data = "dataset/train/"


# proses perbanyak data
# dalam data train terdapat tiga sub folder
for direc in os.listdir(path_data):
    # masuk kedalam sub folder
    folder_path = path_data+direc+"/"
    image = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path,f))]
    num_files = len(image) * 2 # jumlah kelipatan data
    num_generate_files = 0
    while num_generate_files < num_files:
        # prses
        image_path = random.choice(image)
        image_to_transform = sk.io.imread(image_path)
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        trasformed_image = None
        while num_transformations < num_transformations_to_apply:
            key = random.choice(list(available_transformations))
            trasformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1
            new_path = '%s/augmentation_image_%s.jpg' % (folder_path, num_generate_files)
            io.imsave(new_path,img_as_ubyte(trasformed_image))
            num_generate_files += 1