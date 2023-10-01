import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from collections import Counter
from pylab import savefig
import cv2 as cv
import os
import pandas as pd
from skimage import io
import matplotlib.pylab as plt

global img_counter
img_counter = 1

def upload():
    global img_counter
    img_counter = 1

def normal():
    global img_counter
    img_counter = 1

def grayscale():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]
    new_arr = r.astype(int) + g.astype(int) + b.astype(int)
    new_arr = (new_arr/3).astype('uint8')
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)



def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    for i in range(w):
        for j in range(h):
            r, g, b = im.getpixel((i, j))
            if r != g != b:
                return False
    return True


def zoomin():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img = img.convert("RGB")
    img_arr = np.asarray(img)
    new_size = ((img_arr.shape[0] * 2),
                (img_arr.shape[1] * 2), img_arr.shape[2])
    new_arr = np.full(new_size, 255)
    new_arr.setflags(write=1)

    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]

    new_r = []
    new_g = []
    new_b = []

    for row in range(len(r)):
        temp_r = []
        temp_g = []
        temp_b = []
        for i in r[row]:
            temp_r.extend([i, i])
        for j in g[row]:
            temp_g.extend([j, j])
        for k in b[row]:
            temp_b.extend([k, k])
        for _ in (0, 1):
            new_r.append(temp_r)
            new_g.append(temp_g)
            new_b.append(temp_b)

    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            new_arr[i, j, 0] = new_r[i][j]
            new_arr[i, j, 1] = new_g[i][j]
            new_arr[i, j, 2] = new_b[i][j]

    new_arr = new_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)


def zoomout():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img = img.convert("RGB")
    x, y = img.size
    new_arr = Image.new("RGB", (int(x / 2), int(y / 2)))
    r = [0, 0, 0, 0]
    g = [0, 0, 0, 0]
    b = [0, 0, 0, 0]

    for i in range(0, int(x/2)):
        for j in range(0, int(y/2)):
            r[0], g[0], b[0] = img.getpixel((2 * i, 2 * j))
            r[1], g[1], b[1] = img.getpixel((2 * i + 1, 2 * j))
            r[2], g[2], b[2] = img.getpixel((2 * i, 2 * j + 1))
            r[3], g[3], b[3] = img.getpixel((2 * i + 1, 2 * j + 1))
            new_arr.putpixel((int(i), int(j)), (int((r[0] + r[1] + r[2] + r[3]) / 4), int(
                (g[0] + g[1] + g[2] + g[3]) / 4), int((b[0] + b[1] + b[2] + b[3]) / 4)))
    new_arr = np.uint8(new_arr)
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)


def move_left():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
    g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
    b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)


def move_right():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
    g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
    b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)


def move_up():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
    g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
    b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)


def move_down():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    b = np.pad(b, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)


def brightness_addition():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr+100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)


def brightness_substraction():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr-100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)


def brightness_multiplication():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    img_arr = img_arr*1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)


def brightness_division():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    img_arr = img_arr/1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)


def convolution(img, kernel):
    h_img, w_img, _ = img.shape
    out = np.zeros((h_img-2, w_img-2), dtype=float)

    new_img = np.zeros((h_img-2, w_img-2, 3))
    if np.array_equal((img[:, :, 1], img[:, :, 0]), img[:, :, 2]) == True:
        array = img[:, :, 0]
        for h in range(h_img-2):
            for w in range(w_img-2):
                S = np.multiply(array[h:h+3, w:w+3], kernel)
                out[h, w] = np.sum(S)
        out_ = np.clip(out, 0, 255)
        for channel in range(3):
            new_img[:, :, channel] = out_
    else:
        for channel in range(3):
            array = img[:, :, channel]
            for h in range(h_img-2):
                for w in range(w_img-2):
                    S = np.multiply(array[h:h+3, w:w+3], kernel)
                    out[h, w] = np.sum(S)
            out_ = np.clip(out, 0, 255)
            new_img[:, :, channel] = out_
    new_img = np.uint8(new_img)
    return new_img


def edge_detection():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img, dtype=int)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)


def blur():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img, dtype=int)
    kernel = np.array(
        [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)


def sharpening():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img, dtype=int)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)


def histogram_rgb():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        g = img_arr.flatten()
        data_g = Counter(g)
        plt.bar(list(data_g.keys()), data_g.values(), color='black')
        plt.savefig(f'static/img/grey_histogram.jpg', dpi=300)
        plt.clf()
    else:
        r = img_arr[:, :, 0].flatten()
        g = img_arr[:, :, 1].flatten()
        b = img_arr[:, :, 2].flatten()
        data_r = Counter(r)
        data_g = Counter(g)
        data_b = Counter(b)
        data_rgb = [data_r, data_g, data_b]
        warna = ['red', 'green', 'blue']
        data_hist = list(zip(warna, data_rgb))
        for data in data_hist:
            plt.bar(list(data[1].keys()), data[1].values(), color=f'{data[0]}')
            plt.savefig(f'static/img/{data[0]}_histogram.jpg', dpi=300)
            plt.clf()



def df(img):  # to make a histogram (count distribution frequency)
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i, j]] += 1
    return values


def cdf(hist):  # cumulative distribution frequency
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1]+hist[i]
    # Now we normalize the histogram
    # What your function h was doing before
    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf


def histogram_equalizer():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = cv.imread(img_path, 0)
    my_cdf = cdf(df(img))
    # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
    image_equalized = np.interp(img, range(0, 256), my_cdf)
    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(img_path, image_equalized)


def threshold(lower_thres, upper_thres):
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)

    # Salin array yang dapat diubah
    img_arr_copy = np.copy(img_arr)

    condition = np.logical_and(np.greater_equal(img_arr_copy, lower_thres),
                               np.less_equal(img_arr_copy, upper_thres))

    # Tidak perlu mengatur writeable di sini

    img_arr_copy[condition] = 255

    # Buat gambar baru dari array yang sudah diubah
    new_img = Image.fromarray(img_arr_copy)

    # Ubah mode gambar ke RGB
    new_img = new_img.convert('RGB')

    img_counter += 1 
    img_path = f"static/img/img{img_counter}.jpg"
    new_img.save(img_path)

def split_image(image, rows, columns):
    width, height = image.size
    box_width = width // columns
    box_height = height // rows
    cropped_boxes = []

    for i in range(rows):
        for j in range(columns):
            left = j * box_width
            upper = i * box_height
            right = left + box_width
            lower = upper + box_height
            box = image.crop((left, upper, right, lower))
            cropped_boxes.append(box)

    return cropped_boxes

def cropping_susun(rows, columns):
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    original_image = Image.open(img_path)

    # Split the image into specified number of rows and columns
    cropped_boxes = split_image(original_image, rows, columns)

    # Create a new directory to store cropped images
    output_dir = "static/cropped_images"
    os.makedirs(output_dir, exist_ok=True)

    # Save each cropped box as a separate image
    for i, box in enumerate(cropped_boxes):
        # Convert the box to RGB mode before saving
        box = box.convert("RGB")
        output_path = os.path.join(output_dir, f"cropped_{i}.jpg")
        box.save(output_path)

def cropping_acak(rows, columns):
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    original_image = Image.open(img_path)

    # Split the image into specified number of rows and columns
    cropped_boxes = split_image(original_image, rows, columns)

    # Shuffle the cropped boxes
    np.random.shuffle(cropped_boxes)

    # Create a new directory to store shuffled cropped images
    output_dir = "static/cropped_images_random"
    os.makedirs(output_dir, exist_ok=True)

    # Save each shuffled cropped box as a separate image
    for i, box in enumerate(cropped_boxes):
        # Convert the box to RGB mode before saving
        box = box.convert("RGB")
        output_path = os.path.join(output_dir, f"cropped_random_{i}.jpg")
        box.save(output_path)

def restore_history(restore_int):
    global img_counter
    img_counter = restore_int


def identity():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    image = cv.imread(img_path)  # Load the image using OpenCV

    # Apply the identity kernel (no change)
    kernel = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])
    identity_image = cv.filter2D(src=image, ddepth=-1, kernel=kernel)

    img_counter += 1

    # Save the resulting image using OpenCV
    output_img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(output_img_path, identity_image)

def blur_kernel():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    image = cv.imread(img_path)  # Load the image using OpenCV

    kernel = np.ones((3, 3), np.float32) / 9

    blur = cv.filter2D(src=image, ddepth=-1, kernel=kernel)

    img_counter += 1

    # Save the resulting image using OpenCV
    output_img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(output_img_path, blur)

def blur_cv_blur():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    image = cv.imread(img_path)  # Load the image using OpenCV

    cv_blur = cv.blur(src=image, ksize=(5,5))

    img_counter += 1

    # Save the resulting image using OpenCV
    output_img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(output_img_path, cv_blur)


def gaussian_blur(kernelsize):
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    image = cv.imread(img_path)  # Load the image using OpenCV

    cv_gaussianblur = cv.GaussianBlur(src=image,ksize=(kernelsize,kernelsize),sigmaX=0)

    img_counter += 1

    # Save the resulting image using OpenCV
    output_img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(output_img_path, cv_gaussianblur)


def median_blur(kernelsize):
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    image = cv.imread(img_path)  # Load the image using OpenCV

    cv_median = cv.medianBlur(src=image, ksize=kernelsize)

    img_counter += 1

    # Save the resulting image using OpenCV
    output_img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(output_img_path, cv_median)


def sharp_kernel():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    image = cv.imread(img_path)  # Load the image using OpenCV

    kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

    sharp = cv.filter2D(src=image, ddepth=-1, kernel=kernel)

    img_counter += 1

    # Save the resulting image using OpenCV
    output_img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(output_img_path, sharp)


def bilateral_filter():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    image = cv.imread(img_path)  # Load the image using OpenCV

    bf = cv.bilateralFilter(src=image,d=9,sigmaColor=75,sigmaSpace=75)

    img_counter += 1

    # Save the resulting image using OpenCV
    output_img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(output_img_path, bf)

def zero_padding():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    image = cv.imread(img_path)  # Load the image using OpenCV

    image = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)

    img_counter += 1

    # Save the resulting image using OpenCV
    output_img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(output_img_path, image)

def low_filter_pass():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    image = cv.imread(img_path)  # Load the image using OpenCV

    # create the low pass filter
    lowFilter = np.ones((3,3),np.float32)/9
    # apply the low pass filter to the image
    lowFilterImage = cv.filter2D(image,-1,lowFilter)

    img_counter += 1

    # Save the resulting image using OpenCV
    output_img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(output_img_path, lowFilterImage)

def high_filter_pass():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    image = cv.imread(img_path)  # Load the image using OpenCV

    # create the high pass filter
    highFilter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])    
    # apply the high pass filter to the image
    highFilterImage = cv.filter2D(image,-1,highFilter)
    
    img_counter += 1

    # Save the resulting image using OpenCV
    output_img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(output_img_path, highFilterImage)

def band_filter_pass():
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    image = cv.imread(img_path)  # Load the image using OpenCV

    # create the band pass filter
    bandFilter = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    # apply the band pass filter to the image
    bandFilterImage = cv.filter2D(image,-1,bandFilter)

    img_counter += 1

    # Save the resulting image using OpenCV
    output_img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(output_img_path, bandFilterImage)

def custom_kernel(kernel):
    global img_counter
    img_path = f"static/img/img{img_counter}.jpg"
    image = cv.imread(img_path)  # Load the image using OpenCV

    # apply the custom high pass filter to the image
    customKernelImage = cv.filter2D(image, -1, kernel)

    img_counter += 1

    # Save the resulting image using OpenCV
    output_img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(output_img_path, customKernelImage)
