from pkg_resources import resource_stream
from PIL import Image
import matplotlib.pyplot as plt
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


superpixel_rgb = []
superpixel_file = []
for root, dirs, files in os.walk("./frames"):
    for file in files:
        #green1: hand isn't detected when item in hand

        if "_" in file:
            print(file)
            image_path = "./frames/" + file

        # load image
            color_image = np.array(Image.open(resource_stream(__name__, image_path)))
            lab_image = skimage.color.rgb2lab(color_image).tolist()
            number_of_pixels = color_image.shape[0] * color_image.shape[1]

            # SNIC parameters
            number_of_segments = 5
            compactness = 10.00


            segmentation, _, centroids = snic(
                lab_image, number_of_segments, compactness,
                update_func=lambda num_pixels: print("processed %05.2f%%" % (num_pixels * 100 / number_of_pixels)))

            curr_image = color_image.copy()
            # show the output of SNIC

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlabel('R axis')
            ax.set_ylabel('G axis')
            ax.set_zlabel('B axis')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_zlim([0, 1])


            for segment in np.unique(np.asarray(segmentation)):
                curr_segmentation = segmentation.copy()

                indices = np.argwhere(curr_segmentation == segment)
                median = np.median(curr_image[indices[:,0], indices[:,1], :])
                average = np.sum(curr_image[indices[:,0], indices[:,1], :], axis=0) / len(indices)
                # print(median)
                # print(average)
                # print(indices)
                #first pixel in the superpixel
                pixel_value =  (color_image[indices[0][0], indices[0][1], :][0],
                                color_image[indices[0][0], indices[0][1], :][1],
                                color_image[indices[0][0], indices[0][1], :][2])
                #some pixel in the middle of the superpixel
                pixel_value =  (color_image[indices[len(indices)//2][0], indices[len(indices)//2][1], :][0],
                                color_image[indices[len(indices)//2][0], indices[len(indices)//2][1], :][1],
                                color_image[indices[len(indices)//2][0], indices[len(indices)//2][1], :][2])
                pixel_value = average
                superpixel_file.append(file)
                superpixel_rgb.append(pixel_value)
                curr_image[indices[:,0], indices[:,1], :] = pixel_value
                for i in range(0, len(indices)):
                    if i%30 == 0:
                        #BGR
                        pixel_value = np.array([[color_image[indices[i][0], indices[i][1], 2],
                                                 color_image[indices[i][0], indices[i][1], 1],
                                                 color_image[indices[i][0], indices[i][1], 0]]]) / 255.0
                        # ax.scatter(pixel_value[0],pixel_value[1],pixel_value[2], c=[pixel_value])
                        ax.scatter(pixel_value[0][0], pixel_value[0][1], pixel_value[0][2], c=[pixel_value[0]])
            cv2.imshow("Superpixel", curr_image)
            cv2.imshow("Original", color_image)




            plt.show()
            cv2.waitKey(0)

        #     time.sleep(1)

        # print(centroids)
        # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
        # ax0.imshow(mark_boundaries(color_image, np.array(segmentation)))
        # ax0.set_title('Original')
        # ax0.axis('off')
        # ax1.imshow(mark_boundaries(curr_image, np.array(segmentation)))
        # ax1.set_title('Statistic')
        # ax1.axis('off')
        # plt.show()
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('R axis')
ax.set_ylabel('G axis')
ax.set_zlabel('B axis')
superpixel_rgb = np.asarray(superpixel_rgb)

#normalized
# superpixel_rgb = superpixel_rgb / superpixel_rgb.max(axis=0)
# superpixel_rgb = (superpixel_rgb - superpixel_rgb.mean(axis=0)) / superpixel_rgb.std(axis=0)

# model = KMeans(n_init=20, n_clusters=4).fit(superpixel_rgb)
# model = Birch(threshold=0.01, n_clusters=None).fit(superpixel_rgb)
# model = GaussianMixture(n_components=4)
model = DBSCAN(eps=5, min_samples=5).fit(superpixel_rgb)
#model = SpectralClustering(n_clusters=4, n_init=10, gamma=1.0, affinity='nearest_neighbors', eigen_tol=0.0, assign_labels='kmeans').fit(superpixel_rgb)
#labels for GaussianMixture, Birch, KMeans
# label = model.fit_predict(superpixel_rgb)

#labels for DBSCAN
label = model.labels_
u_labels = np.unique(label)

print(u_labels)
#plotting the results:
for i in u_labels:
    ax.scatter(superpixel_rgb[label == i , 0] , superpixel_rgb[label == i , 1], superpixel_rgb[label == i , 2],  label = 1)
plt.legend()
plt.show()

# for index, label in enumerate(superpixel_rgb):



# ax.scatter(superpixel_rgb[:, 0], superpixel_rgb[:, 1], superpixel_rgb[:, 2])
# plt.show()