import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imresize
from numpy.fft import fft2, ifft2
from numpy.linalg import inv
import MaUtilities as mu

sigma = 0.1
lamda = 0.001
eta = 0.125

def init_filter(training_image):
    x = training_image
    y = mu.copy_2D_to_3D(mu.Gaussian_2D((training_image.shape[0] // 2, training_image.shape[1] // 2), training_image.shape[:2]), training_image.shape[-1])

    # FFT for each channel
    X = fft2(x, axes=(-3, -2))
    #print 'X:'
    #mu.show_detail(X)
    Y = fft2(y, axes=(-3, -2))
    #print 'Y:'
    #mu.show_detail(Y)

    conj_X = np.conj(X)

    W_numerator = mu.summerize_each_layer(Y * conj_X)
    #print 'W_numerator:'
    #mu.show_detail(W_numerator)

    W_denominator = mu.summerize_each_layer(X * conj_X)
    # regularization
    W_denominator = W_denominator + np.ones(W_denominator.shape) * lamda
    #print 'W_denominator:'
    #mu.show_detail(W_denominator)

    W = W_numerator / W_denominator

    return W_numerator, W_denominator, W

'''
1 (253, 437)
2 (254, 433)
3 (254, 431)
4 (248, 429)
5 (247, 428)

22 (235, 408)
'''

def filter_update(W_numerator_last, W_denominator_last, frame, global_pos, search_shape):
    updated_image = mu.crop_image(frame, global_pos, shape=search_shape)[..., 0]
    print(frame, global_pos)
    mu.show_detail(updated_image)
    hanned_updated_image = mu.hann2D(updated_image)
    # hanned_updated_image: (w, h)
    X = fft2(hanned_updated_image)
    Y = fft2(mu.Gaussian_2D((X.shape[0] // 2, X.shape[1] // 2), X.shape))

    W_numerator   = eta * Y * np.conj(X) + (1 - eta) * W_numerator_last
    W_denominator = eta * X * np.conj(X) + (1 - eta) * W_denominator_last
    W = W_numerator / W_denominator

    return W_numerator, W_denominator, W

def get_training_image_obviously_point():
    x = np.zeros((224, 224, 8))
    x[..., 0] = mu.crop_image(1, (253, 437))[..., 0]
    x[..., 1] = mu.crop_image(2, (254, 433))[..., 0]
    x[..., 2] = mu.crop_image(3, (254, 431))[..., 0]
    x[..., 3] = mu.crop_image(4, (248, 429))[..., 0]
    x[..., 4] = mu.crop_image(5, (247, 428))[..., 0]
    x[..., 5] = mu.crop_image(6, (246, 430))[..., 0]
    x[..., 6] = mu.crop_image(7, (244, 426))[..., 0]
    x[..., 7] = mu.crop_image(8, (247, 424))[..., 0]
    x = mu.hann2D(x)
    # training_end_pos = (247, 424)
    return x

def get_training_image_strenuous_exercise_point():
    x = np.zeros((224, 224, 8))
    x[..., 0] = mu.crop_image(1, (350, 453))[..., 0]
    x[..., 1] = mu.crop_image(2, (351, 453))[..., 0]
    x[..., 2] = mu.crop_image(3, (350, 451))[..., 0]
    x[..., 3] = mu.crop_image(4, (351, 448))[..., 0]
    x[..., 4] = mu.crop_image(5, (351, 450))[..., 0]
    x[..., 5] = mu.crop_image(6, (349, 446))[..., 0]
    x[..., 6] = mu.crop_image(7, (345, 444))[..., 0]
    x[..., 7] = mu.crop_image(8, (342, 444))[..., 0]
    x = mu.hann2D(x)

    return x

def get_training_image_myocardium():
    x = np.zeros((224, 224, 8))
    x[..., 0] = mu.crop_image(1, (350, 453))[..., 0]
    x[..., 1] = mu.crop_image(2, (351, 453))[..., 0]
    x[..., 2] = mu.crop_image(3, (350, 451))[..., 0]
    x[..., 3] = mu.crop_image(4, (351, 448))[..., 0]
    x[..., 4] = mu.crop_image(5, (351, 450))[..., 0]
    x[..., 5] = mu.crop_image(6, (349, 446))[..., 0]
    x[..., 6] = mu.crop_image(7, (345, 444))[..., 0]
    x[..., 7] = mu.crop_image(8, (342, 444))[..., 0]
    x = mu.hann2D(x)

    return x

def get_training_image(track_pos, search_shape):
    x = np.zeros((search_shape[0], search_shape[1], 1))
    x[..., 0] = mu.crop_image(1, track_pos, search_shape)[..., 0]
    x = mu.hann2D(x)

    return x

def track_frame(training_frames, training_end_pos, traced_frame, search_shape):
    file = open("F_Error", "a+")
    num_training_frames = training_frames.shape[-1]
    track_poses = []
    assert traced_frame > num_training_frames, "frame %d is a training frame." % traced_frame
    global_pos = training_end_pos
    W_numerator, W_denominator, W = init_filter(training_frames)
    for frame_i in range(num_training_frames + 1, traced_frame + 1):
        z = mu.crop_image(frame_i, global_pos, shape=search_shape)[..., 0]
        hanned_z = mu.hann2D(z)
        Z = fft2(hanned_z)
        f = ifft2(W * Z)
        # print np.max(f)
        patch_pos = tuple(mu.get_argmax_pos(f))
        global_pos = (global_pos[0] - search_shape[0] // 2 + patch_pos[0], global_pos[1] - search_shape[1] // 2 + patch_pos[1])
        print(global_pos)
        track_poses.append(global_pos)
        ##mu.display(frame_i, global_pos)
        #mu.display(abs(W), abs(f), z, patch_pos)
        # Online update
        W_numerator, W_denominator, W = filter_update(W_numerator, W_denominator, frame_i, global_pos, search_shape)
    error = 0
    max_error = 0
    first_error = 0
    file.write("point：" + str(global_pos) + "\n")
    for i in range(len(track_poses) // 2):
        print(track_poses[i], track_poses[-(i + 1)])
        current_error = (track_poses[i][0] - track_poses[-(i + 1)][0]) ** 2 + (track_poses[i][1] - track_poses[-(i + 1)][
            1]) ** 2
        file.write(str(current_error) + "\n")
        print(current_error)
        if i == 0:
            first_error = current_error
        max_error = max(current_error, max_error)
        error += (track_poses[i][0] - track_poses[-(i + 1)][0]) ** 2 + (track_poses[i][1] - track_poses[-(i + 1)][1]) ** 2
    print("Begin-End error: %d\nTotal error: %d\nMax error: %d\nAverage error: %f\n" % (
    first_error, error, max_error, error / 221))

    file.close()
'''
training_feature = get_training_image()
#mu.show_detail(training_feature)
#mu.display(training_feature[..., 2])
W_numerator, W_denominator, W = init_filter(training_feature)
mu.show_detail(W)

z = mu.crop_image(16, (247, 424))[..., 0]
hanned_z = mu.hann2D(z)
Z = fft2(z)
# Hann the filter (is it legal?)
f = ifft2(W * Z)
print np.max(f)
tracked_pos = tuple(mu.get_argmax_pos(f))
print tracked_pos

#mu.display('1.png', '2.png')
mu.display(abs(W), abs(f), z, tracked_pos)
#mu.display(mu.summerize_each_layer(abs(W)))
'''

track_poses = [(357, 319), (295, 318), (263, 319), (198, 305), (174, 314), (145, 318), (124, 335), (129, 367), (195, 420), (275, 467)]
feature_shapes = [(50, 50), (40, 60), (50, 50), (50, 50), (50, 50), (50, 50), (50, 50), (30, 30), (50, 50), (50, 50)]

total_poses = []
for i in range(len(track_poses)):
    track_frame(get_training_image(track_poses[i], feature_shapes[i]), track_poses[i], 221, feature_shapes[i])
#track_frame(get_training_image_strenuous_exercise_point(), (342, 444), 111)
