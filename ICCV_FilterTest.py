import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import MaUtilities as mu
import time, os, shutil
from PIL import Image, ImageDraw

import threading, time

'''
In this program, (x, y) of an image(array) is (vertical, horizontal), i.e. the same with numpy and the contrast with PIL.
'''

lock = threading.Lock()

base_model = VGG19(include_top=False, weights='imagenet')

low_model = Model(input=base_model.input, output=base_model.get_layer('block2_conv2').output) # (1, 56, 56, 256)
mid_model = Model(input=base_model.input, output=base_model.get_layer('block3_conv4').output) # (1, 28, 28, 512)
high_model = Model(input=base_model.input, output=base_model.get_layer('block4_conv4').output) # (1, 14, 14, 512)


def init_filter(feature, downer_target_pos):
    # feature: (height, width, layers)
    x = feature
    y = mu.copy_2D_to_3D(mu.Gaussian_2D(downer_target_pos, feature.shape[:2]), feature.shape[-1])
    #y = mu.copy_2D_to_3D(gaussian_2D, feature.shape[-1])
    # FFT for each channel
    X = fft2(x, axes=(-3, -2))
    Y = fft2(y, axes=(-3, -2))

    W_numerator = Y * np.conj(X)
    W_denominator = mu.copy_2D_to_3D(mu.summerize_each_layer(X * np.conj(X)), W_numerator.shape[-1])
    W_denominator = W_denominator + np.ones(W_denominator.shape) * lamda

    W = W_numerator / W_denominator

    return [W_numerator, W_denominator]#, W

def filter_update(Wnds, frame, global_pos, window_shape):
    features, downer_target_pos = get_frame_feature(frame, global_pos, window_shape)
    global eta
    if 22 < frame < 29 or 75 < frame < 88 or 222-22 < frame < 222-29 or 222-75 < frame < 222-88:
        eta = 0.75
    else:
        eta = 0.3
    updated_Wnds = []
    for feature, Wnd in zip(features, Wnds):
        X = fft2(feature, axes=(-3, -2))
        Y = fft2(mu.copy_2D_to_3D(mu.Gaussian_2D(downer_target_pos, X.shape[:2]), feature.shape[-1]), axes=(-3, -2))

        W_numerator = eta * Y * np.conj(X) + (1 - eta) * Wnd[0]
        W_denominator = eta * mu.copy_2D_to_3D(mu.summerize_each_layer(X * np.conj(X)), Wnd[0].shape[-1]) + (1 - eta) * Wnd[1]
        W_denominator += np.ones(W_denominator.shape) * lamda

        updated_Wnds.append([W_numerator, W_denominator])

    return updated_Wnds

# low mid high
def get_frame_feature(frame, target_pos, shape):
    downer_target = get_downer_coordinate(target_pos, image_shape, shape)
    #print("downer target coordinate: ", downer_target)
    x = mu.crop_image(frame, target_pos, shape=shape, mode='gray')  # 1.png (253, 437)
    x = mu.resize_image(x)
    x = np.expand_dims(x, axis=0)
    x = np.float64(x)
    x = preprocess_input(x)

    models = [low_model, mid_model, high_model]
    features = []
    for model in models:
        #lock.acquire()
        feature = model.predict(x)
        #lock.release()
        #x = featured
        # note that for PIL, x is horizontal while y is vertical, which is contrast with numpy.
        feature = mu.resize_image(feature, shape)
        feature = mu.hann2D(feature)

        features.append(feature)

    return features, downer_target
'''
def get_upper_coordinate(local_coordinate, search_window_shape, search_center):
    search_window_shape_half = (search_window_shape[0] // 2, search_window_shape[1] // 2)
    return tuple([x-y+z for x,y,z in zip(local_coordinate, search_window_shape_half, search_center)])
'''

def get_upper_coordinate(local_coordinate, local_window_shape, last_target_pos, upper_window_shape):
    local_window_shape_half = (local_window_shape[0] // 2, local_window_shape[1] // 2)

    local_window_center_x = max(last_target_pos[0], local_window_shape_half[0])
    local_window_center_x = min(local_window_center_x, upper_window_shape[0] - local_window_shape_half[0])
    local_window_center_y = max(last_target_pos[1], local_window_shape_half[1])
    local_window_center_y = min(local_window_center_y, upper_window_shape[1] - local_window_shape_half[1])
    # related to upper window
    local_window_center = (local_window_center_x, local_window_center_y)

    return tuple([x-y+z for x,y,z in zip(local_coordinate, local_window_shape_half, local_window_center)])

def get_downer_coordinate(global_coordinate, global_shape, downer_window_shape):
    assert global_shape[0] > downer_window_shape[0] and global_shape[1] > downer_window_shape[1], "Uncorrect window shape."
    (x, y) = (0, 0)
    # x
    if global_coordinate[0] <= downer_window_shape[0] // 2:
        x = global_coordinate[0]
    elif global_coordinate[0] >= global_shape[0] - downer_window_shape[0] // 2:
        x = downer_window_shape[0] - (global_coordinate[0] - (global_shape[0] - downer_window_shape[0] // 2))
    else:
        x = downer_window_shape[0] // 2
    
    # y
    if global_coordinate[1] <= downer_window_shape[1] // 2:
        y = global_coordinate[1]
    elif global_coordinate[1] >= global_shape[1] - downer_window_shape[1] // 2:
        y = downer_window_shape[1] - (global_coordinate[1] - (global_shape[1] - downer_window_shape[1] // 2))
    else:
        y = downer_window_shape[1] // 2

    return (x, y)


def get_optimal_pos(fs, global_pos, feature_shape):
    # global_pos is the center of last patch(search_window)
    # convert current position coordinate to upper range position coordinate
    # get_up_pos = lambda local_pos, half_search_width, search_center : tuple([x-y+z for x,y,z in zip(local_pos, half_search_width, search_center)])
    search_window_shape = (20, 30)
    if search_window_shape > feature_shape:
        search_window_shape = feature_shape
    #search_width = 50
    f_low, f_mid, f_high = fs
    # center of the bottom range, based on f, i.e. search_window.
    target_pos_f = tuple(mu.get_argmax_pos(f_high))

    # f_cropped is the minimum search range
    f_high_cropped = mu.crop_image(f_high, target_pos_f, search_window_shape)
    f_mid_cropped  = mu.crop_image(f_mid, target_pos_f, search_window_shape)
    f_low_cropped  = mu.crop_image(f_low, target_pos_f, search_window_shape)
    # based on the minimum search range
    search_optimal_pos_bottom = mu.get_argmax_pos(f_high_cropped * feature_weights[0] + f_mid_cropped * feature_weights[1] + f_low_cropped * feature_weights[2]) #1 0.2 0.2
    # the maximum value pos produced by f_high and f_mid becomes the new search center
    # convert the coordinate to upper range coordinate
    target_pos_f = get_upper_coordinate(search_optimal_pos_bottom, search_window_shape, target_pos_f, feature_shape)

    f_mid_cropped  = mu.crop_image(f_mid, target_pos_f, search_window_shape)
    f_low_cropped  = mu.crop_image(f_low, target_pos_f, search_window_shape)
    search_optimal_pos_bottom = mu.get_argmax_pos(f_mid_cropped * 5 + f_low_cropped * 1)
    #target_pos_f  = get_upper_coordinate(search_optimal_pos_bottom, search_window_shape, target_pos_f, feature_shape)
    # convert coordinate to global coordinate(based on the whole image)
    global_pos = get_upper_coordinate(target_pos_f, feature_shape, global_pos, image_shape)

    return global_pos

def track_frame(training_frame, window_shape, end_frame, *training_target_poses):
    file = open('error', 'a+')
    for global_pos in training_target_poses:
        # global_pos is the center of the search window
        #global_pos = training_target_poses
        # Crop a patch centering at 'global_pos' and of size 'size', then get its CNN features with the same size
        features, downer_target_pos = get_frame_feature(training_frame, global_pos, shape=window_shape)
        # low, mid, high
        Wnds = [init_filter(feature, downer_target_pos) for feature in features]
        track_poses = []
        for frame_i in range(training_frame + 1, end_frame + 1):
            # zs : low mid high
            zs, _ = get_frame_feature(frame_i, global_pos, shape=window_shape)
            fs = []
            for Wnd, z in zip(Wnds, zs):
                # W: (window_shape, 256) (window_shape, 512) (window_shape, 512)
                W = Wnd[0] / Wnd[1]
                Z = fft2(z, axes=(-3, -2))
                #f = ifft2(mu.summerize_each_layer(W * Z))
                #f = np.real(fftshift(ifft2(mu.summerize_each_layer(W * Z))))
                # f: window_shape
                f = np.real(ifft2(mu.summerize_each_layer(W * Z)))
                fs.append(f)
            current_frame_argmax_poses = [get_upper_coordinate(mu.get_argmax_pos(f), f.shape[:2], global_pos, image_shape) for f in fs]
            #current_frame_argmax_poses = [local_to_global_pos(mu.get_argmax_pos(f), global_pos) for f in fs]
            # fs: low mid high
            new_pos = get_optimal_pos(fs, global_pos, window_shape)
            print("frame %d" % frame_i, new_pos, time.strftime("%H:%M:%S", time.localtime()))

            current_frame_argmax_poses = []
            current_frame_argmax_poses.append(new_pos)
            track_poses.append(new_pos)
            # low mid high optimal    RGBY
            lock.acquire()
            mu.save_rected_image(frame_i, current_frame_argmax_poses)
            lock.release()
            Wnds = filter_update(Wnds, frame_i, new_pos, window_shape=window_shape)
        error = 0
        max_error = 0
        first_error = 0
        file.write("point：" + str(global_pos) + "\n")
        for i in range(len(track_poses) // 2):
            print(track_poses[i], track_poses[-(i+1)])
            current_error = (track_poses[i][0] - track_poses[-(i+1)][0]) ** 2 + (track_poses[i][1] - track_poses[-(i+1)][1]) ** 2
            file.write(str(current_error) + "\n")
            print(current_error)
            if i == 0:
                first_error = current_error
            max_error = max(current_error, max_error)
            error += (track_poses[i][0] - track_poses[-(i+1)][0]) ** 2 + (track_poses[i][1] - track_poses[-(i+1)][1]) ** 2
        print("Begin-End error: %d\nTotal error: %d\nMax error: %d\nAverage error: %f\n" % (first_error, error, max_error, error / 221))
    file.close()



#track_frame(1, (126, 176), window_shape=feature_shape, end_frame=164)
# # motor
# mu.image_path = "./MotorFrame/%d.jpg"
# mu.save_path = "./MotorResult/%d.png"

# heart
mu.image_path = "./IU22Frame/%d.png"
mu.save_path = "./IU22Result/%d.png"
image_shape = mu.image_to_array(1).shape[:2]

for f in os.listdir("./IU22Frame/"):
    shutil.copyfile("./IU22Frame/" + f, "./IU22Result/" + f)


sigma = 0.3
lamda = 0.5
eta = 0.25
# high mid low
feature_weights = [1, 0, 0.5]
gaussian_2D = None

track_poses = [(357, 319), (295, 318), (263, 319), (198, 305), (174, 314), (145, 318), (124, 335), (129, 367), (195, 420), (275, 467)]
feature_shapes = [(50, 50), (40, 60), (50, 50), (50, 50), (50, 50), (50, 50), (50, 50), (30, 30), (50, 50), (50, 50)]

#track_frame(1, (30, 30), 1, track_poses[0])

t = threading.Thread(target=track_frame, args=(1, feature_shapes[0], 111, track_poses[0]))
#t.start()

for i in range(len(track_poses)):
    track_frame(1, feature_shapes[i], 221, track_poses[i])
    #t = threading.Thread(target=track_frame, args=(1, feature_shapes[i], 111, track_poses[i]))
    #t.start()
    #time.sleep(1)

