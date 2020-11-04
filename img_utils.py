import os
import shutil
import pydicom
from os import walk
import numpy as np
import png


def read_dict():
    """
    Utility function to read dictionary with this structure {ID: {type of hemorrhage: probability}}
    from text file. Kindly shared by the other group (Daniel and Patrick)
    :return:
    """
    with open('../small_data_set_balanced_probs.txt', 'r') as f:
        content = f.read()
        probabilities = eval(content)
    return probabilities


def move_files(probs):
    """
    Utility function to reorganize pictures into folders by healthy or hemorrhagic class.
    To be used for the ImageFolder of the healthy/hemorrhagic classifier
    :param probs: dictionary with this structure: {ID: {type of hemorrhage: probability}}
    """
    path = '../brain_tiny_dataset_class/png/'
    for _, _, files in os.walk(path):
        for file in files:
            # Reads the ID
            id = file[3:-4]
            try:
                # Reads dictionary of probabilities
                result = probs[id]
                # Moves pictures in 2 folders
                if result['epidural'] > 0 or result['intraparenchymal'] > 0 \
                    or result['intraventricular'] > 0 or result['subarachnoid'] > 0 \
                    or result['subdural'] > 0:
                    shutil.move(path + file, '../brain_tiny_dataset_class/hemorrhage/' + file)
                else:
                    shutil.move(path + file, '../brain_tiny_dataset_class/healthy/' + file)
            except KeyError:
                continue


# From https://stackoverflow.com/questions/60219622/python-convert-dcm-to-png-images-are-too-bright
def save_as_png(path):
    """
    Converts .dcm images to pixel array, rescales pixels
    in range 0-255, downcasts them to uint8 and saves them on disk
    :param path: Path of the directory of pictures to convert
    """
    for _, _, filename in walk(path):
        for f in filename:
            medical_image = pydicom.dcmread(path + f)
            shape = medical_image.pixel_array.shape
            # Convert to float to avoid overflow or underflow losses
            brain_image = medical_image.pixel_array.astype(float)
            # Rescaling grey scale between 0-255
            scaled_image = (np.maximum(brain_image, 0) / brain_image.max()) * 255.0
            # Convert to uint
            scaled_image = np.uint8(scaled_image)
            # Write the PNG file
            with open(f'{path}png/{f.strip(".dcm")}.png', 'wb') as png_file:
                w = png.Writer(shape[1], shape[0], greyscale=True)
                w.write(png_file, scaled_image)


save_as_png('../small_dataset/')
probs = read_dict()
move_files(probs)