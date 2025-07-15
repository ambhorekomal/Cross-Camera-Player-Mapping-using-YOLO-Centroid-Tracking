import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_color(image, k=4, image_processing_size=None):
    # Resize image if needed
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size)

    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # Clustering
    clt = KMeans(n_clusters=k, n_init=10)
    clt.fit(image)

    # Most common cluster
    dominant_color = clt.cluster_centers_[np.argmax(np.bincount(clt.labels_))]

    return dominant_color.astype(int)
