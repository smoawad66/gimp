import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from math import ceil, log, e, sqrt, pi

def image(name):
    return f'../images/{name}'

def get_value(value):
    return max(0, min(255, value))

# def show_image(image, title='Image', type=None, fig_size=(8, 4)):
#     plt.figure(figsize=fig_size)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(image, cmap=type)
#     plt.xlabel(title, fontsize=12)
#     plt.show()

# def show_images(images, ncol=2, fig_size=(8, 4)):
#     plt.figure(figsize=fig_size)
#     for i, (image, title, img_type) in enumerate(images):
#         plt.subplot(ceil(len(images)*1.0/ncol), ncol, i + 1) 
#         plt.xticks([])
#         plt.yticks([])
#         plt.imshow(image, cmap=img_type)
#         plt.xlabel(title, fontsize=fig_size[0])
#     plt.tight_layout()
#     plt.show()

# def show_histogram(img, title='Histogram', xlabel='Gray level', ylabel='Count'):
#     plt.figure(figsize=(16, 8))
#     bins = np.arange(0, 256, 1)
#     sns.histplot(img.flatten(), bins=bins, color='r')
#     plt.xticks(bins[::10])
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
    
# def path2Array(path):
    # return plt.imread(path)