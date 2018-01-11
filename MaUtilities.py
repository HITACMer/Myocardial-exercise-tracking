# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:01:45 2017

@author: ma

A utilities library for simple data process.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal

image_path = './IU22Frame/%d.png'
save_path = "./IU22Result/%d.png"

# amplify a image(array) using bilinear interpolation.
def BilinearInterpolation(feature_array, new_shape=(224, 224)):
    #image_array = np.float32(image_array)
    if len(feature_array.shape) == 3:
        return np.asarray([np.asarray(Image.fromarray(feature_array[:, :, i]).resize(new_shape, Image.BILINEAR))
                        for i in range(feature_array.shape[-1])]).transpose(1, 2, 0)
    if len(feature_array.shape) == 4:
        return np.asarray([np.asarray(Image.fromarray(feature_array[0, :, :, i]).resize(new_shape, Image.BILINEAR))
                        for i in range(feature_array.shape[-1])]).transpose(1, 2, 0)

'''
# add a Hann window to feature.
def hann2D(feature):
    assert feature.shape[:2] == (224, 224), "Hann2D : feature must be (224, 224)."
    window = signal.hann(224 * 224).reshape((224, 224))
    if len(feature.shape) == 3:
        window = copy_2D_to_3D(window, feature.shape[-1])
        return feature * window * window.transpose(1, 0, 2)
    if len(feature.shape) == 2:
        return feature * window * window.transpose()
'''

# add a Hann window to feature.
def hann2D(feature):
    width = feature.shape[0]
    height = feature.shape[1]
    window = np.dot(signal.hann(width).reshape((width, 1)), signal.hann(height).reshape((1, height)))
    if len(feature.shape) == 3:
        window = copy_2D_to_3D(window, feature.shape[-1])

    return feature * window

# Translate an image file to an numpy array.
def image_to_array(file_name):
    # RGB
    if type(file_name) == int:
        file_name = image_path % file_name
    image = Image.open(file_name)
    return np.asarray(image)[..., 0: 3]

# def save_rected_image(frame, positions):
#     plt.subplot(111)
#     path = image_path % frame
#     plt.imshow(image_to_array(path))
#     width = 20
#     colors = ['red', 'green', 'blue', 'yellow']
#     color_index = 0
#     for position in positions:
#         rect = plt.Rectangle((position[1] - width // 2, position[0] - width // 2), width, width,
#                              linewidth=1, alpha=1, facecolor='none', edgecolor=colors[color_index % len(colors)])
#         plt.subplot(111).add_patch(rect)
#         color_index += 1
#
#     plt.savefig(save_path % frame)
#     plt.close()

def save_rected_image(frame, positions):
    path = save_path % frame
    image = Image.open(path)
    painter = ImageDraw.Draw(image)
    width = 5
    colors = ['red', 'green', 'blue', 'yellow']
    color_index = 0
    for position in positions:
        painter.rectangle([(position[1] - width//2, position[0] - width//2), (position[1] + width//2, position[0] + width//2)], fill=colors[color_index % len(colors)], outline=colors[color_index % len(colors)])
        color_index += 1
    image.save(path)
    del painter

# Display images. Support the number of frames and filename and array. When given a coordinate tuple behind a valid type,
# the function will draw an rectangle on last image centering at the coordinate.
def display(*input):
    plt.figure('Display')
    #image_tobe_saved = []

    num_rect = 0
    for i in range(len(input)):
        if type(input[i]) == tuple:
            num_rect += 1
    num_image = len(input) - num_rect
    assert num_image < 10, "The number of images must smaller than 10."
    marked_image = 0
    for i in range(len(input)):
        if type(input[i]) == tuple:
            assert i is not 0, "Rect must draw on an existed image."
            marked_image += 1
            position = 100 + 10 * num_image + i + 1 - marked_image
            current_image = plt.subplot(position)
            center = input[i]
            width = 20
            rect = plt.Rectangle((center[1] - width//2, center[0] - width//2), width, width,
                                 linewidth = 1, alpha = 1, facecolor = 'none', edgecolor = 'yellow')
            current_image.add_patch(rect)
            print(input[i])
            continue

        position = 100 + 10 * num_image + i + 1 - marked_image
        plt.subplot(position)

        if type(input[i]) == np.ndarray:
            plt.imshow(input[i])
            #image_tobe_saved += input([i])
        else:
            path = None
            if type(input[i]) == int:
                path = image_path % input[i]
            elif type(input[i]) == str:
                path = input[i]
            else:
                raise TypeError('Unsupport type')
            plt.imshow(image_to_array(path))
            #image_tobe_saved += image_to_array(path)
    plt.show()

    #return image_tobe_saved

# Resize an array to 'new_shape' using bilinear interpolation.
def resize_image(feature_array, new_shape=(224, 224)):
    shape_PIL = (new_shape[1], new_shape[0])
    if len(feature_array.shape) == 3:
        return np.asarray([np.asarray(Image.fromarray(feature_array[:, :, i]).resize(shape_PIL, Image.BILINEAR))
                        for i in range(feature_array.shape[-1])]).transpose(1, 2, 0)
    if len(feature_array.shape) == 4:
        return np.asarray([np.asarray(Image.fromarray(feature_array[0, :, :, i]).resize(shape_PIL, Image.BILINEAR))
                        for i in range(feature_array.shape[-1])]).transpose(1, 2, 0)

    #return np.expand_dims(image, axis = 0)

# Return the indice of the greatest element in a tensor.
def get_argmax_pos(a):
    return np.int64(np.transpose(np.where(a == np.max(a)))[0])

# Return an Gaussian matrix of 'shape' whose center is 'center'(x, y).
def Gaussian_2D(center, shape, sigma = 0.1):
    #signal.gaussian()
    confidence_score = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            confidence_score[i, j] = (i - center[0])**2 + (j - center[1])**2
    confidence_score = confidence_score * -1. / (2 * sigma**2)
    confidence_score = np.exp(confidence_score)
    return confidence_score

# Judge equality of two same-shape array ignoring of tiny difference.
def is_equal(a, b):
    assert a.shape == b.shape, "Two arrays must be the same shape."
    return np.sum(abs(a - b)) < 0.000001


# Return a cropped array whose shape is 'shape' & given center. Support string of file name & array.
def crop_image(image, center, shape=(224, 224), mode='gray'):
    if type(image) == int:
        image = image_path % image
    if type(image) == str:
        image = image_to_array(image)
    cropped_image = None
    if type(shape) == int:
        shape = (shape, shape)
    assert image.shape[0] >= shape[0] and image.shape[1] >= shape[1], "Uncorrect crop shape"
    rectified_x = max(center[0], shape[0] // 2)
    rectified_x = min(rectified_x, image.shape[0] - shape[0] // 2)
    rectified_y = max(center[1], shape[1] // 2)
    rectified_y = min(rectified_y, image.shape[1] - shape[1] // 2)
    rectified_center = (rectified_x, rectified_y)
    if len(image.shape) == 3:
        cropped_image = image[rectified_center[0]-shape[0]//2 : rectified_center[0]+shape[0]//2, rectified_center[1]-shape[1]//2 : rectified_center[1]+shape[1]//2, :]
        if mode == 'gray':
            cropped_image = copy_2D_to_3D(cropped_image[..., 0], 3)
    if len(image.shape) == 2:
        cropped_image = image[rectified_center[0]-shape[0]//2 : rectified_center[0]+shape[0]//2, rectified_center[1]-shape[1]//2 : rectified_center[1]+shape[1]//2]

    return cropped_image

# Given a shape, return its center coordinate(tuple).
def get_shape_center(shape):
    return (shape[0] // 2, shape[1] // 2)

# For a cube-shape tensor, compression to a 2D array by summerizing each layer.
def summerize_each_layer(cube):
    assert len(cube.shape) == 3, "Cube must be 3dim."
    return np.sum(cube, axis = 2)

# Return a 3D array copyed from a 2D array i.e. Pile up num_layers 2D arrays.
def copy_2D_to_3D(array_2D, num_layers):
    return np.tile(array_2D, (num_layers, 1, 1)).transpose((1, 2, 0))

# Print detail info of the given np_array.
def show_detail(np_array, comment=None):
    print(comment, "shape:", np_array.shape, "dtype:", np_array.dtype, "max:", np.max(np_array), "min:", np.min(np_array))