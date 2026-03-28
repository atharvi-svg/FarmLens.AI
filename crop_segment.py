import numpy as np
from sklearn.cluster import KMeans

def segment_crop(image , k=3):

    #now we reshape the image
    pixels = image.reshape((-1,3))  # 3 because we have 3 color channels (R,G,B)

    #apply k-means, random state helps in maintaining consistent results
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(pixels)

    #labels tell you which cluster each pixel belongs to
    labels = kmeans.labels_

    centers = np.uint8(kmeans.cluster_centers_)  #image pixels must be integers therefore we use np.uint8 so that range of the integers is from 0-255

    #segmented image
    segmented = centers[labels].reshape(image.shape)  #we're rebuilding the image using cluster colours

    #calculate the distribution
    counts = np.bincount(labels)  #count how many pixels belong to each cluster
    percentages = counts/len(labels)*100  #calculate the percentage of each cluster

    #detect green cluster(healthy crops)
    healthy = 0

    for i, c in enumerate(centers):  # i is index and c is the array of color of the cluster center
        if c[1] > c[0] and c[1] > c[2]:   # green dominant
            healthy += percentages[i]

    unhealthy = 100 - healthy

    return segmented, healthy, unhealthy