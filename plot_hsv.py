from pkg_resources import resource_stream
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import skimage.color
from mpl_toolkits.mplot3d import Axes3D
from skimage.segmentation import mark_boundaries
from pysnic.algorithms.snic import snic
import os
import time
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
import cv2
from colormap import rgb2hex
from sklearn.decomposition import PCA, FastICA

from annotations import getFrameLabel
from annotations import getXMLLabel
from itertools import permutations
import copy


# def eaf_to_htk():


superpixel_rgb = []
superpixel_file = []
labels = [] #hand, green, blue, etc
file_names = [] #filenames for each  pixel
indices = [] #pixel location
x = []
y = []
two_d = []
fps = 60


l = list(permutations(range(0, 4)))
label_to_number_array_of_dictionaries = []

label_to_number = {"blue": 0.0, "green": 1.0, "hand": 2.0, "red": 3.0}
label_to_rgb = {"blue": [0,0,255], "green": [0,255,0], "red": [255,0,0], "hand": [0,0,0]}
for permutation in l:
    label_to_number_copy = copy.deepcopy(label_to_number)
    label_to_number_copy["blue"] = permutation[0]
    label_to_number_copy["green"] = permutation[1]
    label_to_number_copy["hand"] = permutation[2]
    label_to_number_copy["red"] = permutation[3]
    label_to_number_array_of_dictionaries.append(label_to_number_copy)
print(label_to_number_array_of_dictionaries)

def frame_to_time(frame):
    #time in ms?
    time = frame/fps*1000
    return time


for root, dirs, files in os.walk("./frames"):
    for file in files[1000:2000]:
        # print(file)
        #green1: hand isn't detected when item in hand

        frame_number = int(file[file.find("_") + 1:file.find(".")])
        # print(frame_to_time(frame_number))
        video_name = file[:file.find("_")]
        # print(file[:file.find("_")])
        label = None
        if "blue1" in video_name or "red1" in video_name or "green1" in video_name:
        # if "green1" in video_name:
        #     print(video_name)
            xmlTree = getXMLLabel("./Annotations/" + video_name + "_edited.eaf")
            label = getFrameLabel(xmlTree, frame_to_time(frame_number))
        # if label == "blue":
        #     print(label)


            image_path = "./frames/" + file

        # load image
            color_image = np.array(Image.open(resource_stream(__name__, image_path)))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            lab_image = skimage.color.rgb2lab(color_image).tolist()
            number_of_pixels = color_image.shape[0] * color_image.shape[1]
            hsv_image =  skimage.color.rgb2hsv(color_image)
            # SNIC parameters
            # number_of_segments = 2
            # compactness = 10.00
            # segmentation, _, centroids = snic(
            #     lab_image, number_of_segments, compactness,
            #     update_func=lambda num_pixels: print("processed %05.2f%%" % (num_pixels * 100 / number_of_pixels)))

            curr_image = color_image.copy()
            dims = hsv_image.shape
            for i in range(0, dims[0]):
                for j in range(0, dims[1]):
                    # subsample
                    if i % 1 == 0:
                    # BGR
                        hsv_value = np.array([[hsv_image[i, j, 0],
                                               hsv_image[i, j, 1],
                                               hsv_image[i, j, 2]]])
                        # rgb_value = np.array([[color_image[i, j, 0],
                        #                        color_image[i, j, 1],
                        #                        color_image[i, j, 2]]]) / 255.0

                        x.append(hsv_value[0][0])
                        y.append(hsv_value[0][1])
                        two_d.append([hsv_value[0][0], hsv_value[0][1]])
                        labels.append(label) #hand, green, blue, etc
                        file_names.append(file)
                        indices.append((i,j))
                    # for segment in np.unique(np.asarray(segmentation)):
                    #     curr_segmentation = segmentation.copy()
                    #
                    #     indices = np.argwhere(curr_segmentation == segment)
                    #     median = np.median(curr_image[indices[:,0], indices[:,1], :])
                    #     average = np.sum(curr_image[indices[:,0], indices[:,1], :], axis=0) / len(indices)
                    #     # print(median)
                    #     # print(average)
                    #     # print(indices)
                    #     #first pixel in the superpixel
                    #     pixel_value =  (color_image[indices[0][0], indices[0][1], :][0],
                    #                     color_image[indices[0][0], indices[0][1], :][1],
                    #                     color_image[indices[0][0], indices[0][1], :][2])
                    #     #some pixel in the middle of the superpixel
                    #     pixel_value =  (color_image[indices[len(indices)//2][0], indices[len(indices)//2][1], :][0],
                    #                     color_image[indices[len(indices)//2][0], indices[len(indices)//2][1], :][1],
                    #                     color_image[indices[len(indices)//2][0], indices[len(indices)//2][1], :][2])
                    #     pixel_value = average
                    #     superpixel_rgb.append(pixel_value)
                    #     curr_image[indices[:,0], indices[:,1], :] = pixel_value

# superpixel_rgb = np.asarray(superpixel_rgb)
# superpixel_hsv_xy = np.asarray(two_d)
# ica = FastICA()
# print(superpixel_hsv_xy.shape)
# S_ica_ = ica.fit(superpixel_hsv_xy).transform(superpixel_hsv_xy)
#
# print(superpixel_hsv_xy.shape)

# axis_list = [ica.mixing_]
# print(axis_list)


hs = np.asarray(two_d)
# x = np.sin(np.pi * hs[:,0])
theta = np.pi * 2 * hs[:,0] #hue
r = hs[:,1] #saturation
x = np.multiply(r,np.cos(theta))
y = np.multiply(r,np.sin(theta))
xy = np.asarray([x,y]).T

# r = np.sqrt(x**2+y**2) #saturation
# t = np.arctan2(y,x) #hue
# r = hs[:,1] #saturation
# t = 2*np.pi * hs[:,0] #hue
# # t =  2*np.pi * np.random.random((35726))
# rt = np.asarray([r,t]).T
# print(hs.shape)
# print(rt.shape)
# fig = plt.figure()
# norm = colors.Normalize(vmin=0,vmax=2*np.pi)
# ax = fig.add_subplot(111, projection='polar')
# plt.scatter(t, r, c=t,cmap='hsv', alpha=0.005, norm=norm)


# fig = plt.figure()

# plt.xlim(0, 2*np.pi)
# plt.ylim(0, 1)
# plt.scatter(x, y, c='b' ,alpha=0.005)

# plt.scatter(x, y, c=model.labels_.astype(float),alpha=0.005)


# Visualize it:
#kmeans per pixel at the moment (so a lot of noise as hand pixels are incorrectly labelled)
#TODO: kmeans on a different representation rather than per pixel
model = KMeans(n_clusters=4).fit(xy)

print(len(label_to_number_array_of_dictionaries))
max_accuracy = 0
#Do all permutations to find which clusters are associated with which class
#Input to Kmeans clustering on all the hs values
#
actual_dict = None
for curr_dict in label_to_number_array_of_dictionaries:
    new_labels = []
    for curr_label in labels:
        new_labels.append(float(curr_dict[curr_label]))
    np_labels = np.asarray(new_labels)
    # print(np_labels.shape)
    #
    # print(model.labels_.astype(float).shape)

    accuracy = np.sum(np_labels == model.labels_.astype(float))/len(np_labels)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        print(accuracy)
        print(curr_dict)
        actual_dict = curr_dict


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"


def image_to_class(predicted_labels, actual_dict):
    """
    :param actual_dict: correct mapping of label name to label number
    :return:
    """
    print(actual_dict)
    for root, dirs, files in os.walk("./frames"):
        print(len(set(file_names)))
        print(len(set(files)))
        for file in files[1000:2000]:
            if "blue1" in video_name: #or "red1" in video_name or "green1" in video_name:

            # if "green1" in file:

                image_path = "./frames/" + file
                # load image
                color_image = np.array(Image.open(resource_stream(__name__, image_path)))
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                height = color_image.shape[0]
                width = color_image.shape[1]

                #Create image of predicted labels
                curr_indices = []
                for idx, file_name in enumerate(file_names):
                    if file_name == file:
                        curr_indices.append(idx)
                predicted_label_image = np.zeros((height, width, 3))
                actual_label_image = np.zeros((height, width, 3))
                if len(curr_indices) != 0:
                    print("Number of Pixels:" + str(len(curr_indices)))
                    for index in curr_indices:
                        predicted_label_number = predicted_labels[index]
                        # label number to label name
                        predicted_label_name = get_key(predicted_label_number, actual_dict)
                        # label name to label color (global dict)
                        predicted_rgb = label_to_rgb[predicted_label_name]

                        actual_label_name = labels[index]
                        # label name to label color (global dict)
                        actual_rgb = label_to_rgb[actual_label_name]

                        x = indices[index][0]
                        y = indices[index][1]
                        predicted_label_image[x,y] = predicted_rgb
                        actual_label_image[x,y] = actual_rgb
                    f, axarr = plt.subplots(1, 3)
                    axarr[0].imshow(color_image)
                    axarr[0].set_title("Image")
                    axarr[1].imshow(predicted_label_image)
                    axarr[1].set_title("Predicted")
                    axarr[2].imshow(actual_label_image)
                    axarr[2].set_title("Ground Truth")
                    print(actual_label_name)
                    print(actual_rgb)
                    print(predicted_label_name)
                    plt.show()
                    cv2.waitKey(1)


image_to_class(model.labels_.astype(float), actual_dict)


#Plot image side-by-side with color model predictions





# produce a legend with the unique colors from the scatter


fig, ax = plt.subplots()

scatter = ax.scatter(theta, r, c=model.labels_.astype(float))
legend1 = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
ax.add_artist(legend1)


# if axis_list is not None:
#     colors = ['orange', 'red']
#     for color, axis in zip(colors, axis_list):
#         axis /= axis.std()
#         x_axis, y_axis = axis
#         # Trick to get legend to work
#         # plt.plot(x_axis, y_axis, linewidth=2, color='r')
#         plt.quiver((.5,.5), (.5, .5), x_axis, y_axis, zorder=11, width=0.01,
#                    scale=6, color='r')

plt.title("HS Plot")
plt.xlabel('h')
plt.ylabel('s')
plt.show()
cv2.waitKey(0)






#normalized
# superpixel_rgb = superpixel_rgb / superpixel_rgb.max(axis=0)
# superpixel_rgb = (superpixel_rgb - superpixel_rgb.mean(axis=0)) / superpixel_rgb.std(axis=0)

# model = KMeans(n_init=20, n_clusters=4).fit(superpixel_rgb)
# model = Birch(threshold=0.01, n_clusters=None).fit(superpixel_rgb)
# model = GaussianMixture(n_components=4)
# model = DBSCAN(eps=5, min_samples=5).fit(superpixel_rgb)
#model = SpectralClustering(n_clusters=4, n_init=10, gamma=1.0, affinity='nearest_neighbors', eigen_tol=0.0, assign_labels='kmeans').fit(superpixel_rgb)
#labels for GaussianMixture, Birch, KMeans
# label = model.fit_predict(superpixel_rgb)

#labels for DBSCAN
# label = model.labels_
# u_labels = np.unique(label)
#
# print(u_labels)
#plotting the results:
# for i in u_labels:
#     ax.scatter(superpixel_rgb[label == i , 0] , superpixel_rgb[label == i , 1], superpixel_rgb[label == i , 2],  label = 1)
# plt.legend()
# plt.show()

# for index, label in enumerate(superpixel_rgb):



# ax.scatter(superpixel_rgb[:, 0], superpixel_rgb[:, 1], superpixel_rgb[:, 2])
# plt.show()