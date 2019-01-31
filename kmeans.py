from PIL import Image
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from read import read_files, standardize_data

def project(cov, k):
    w, v = LA.eig(cov)

    idx = np.argsort(w)[::-1][0:k]

    projection_matrix = []
    for i in idx:
        eig_vec = v[:,i]
        projection_matrix.append(eig_vec)

    return np.array(projection_matrix).transpose()

def determine_cluster(obs, ref_vecs):
    min = LA.norm(obs - ref_vecs[0])
    cluster = 0

    for i in range(1, len(ref_vecs)):
        distance = LA.norm(obs - ref_vecs[i])

        if distance < min:
            min = distance
            cluster = i

    return cluster

def get_color(index):
    color = {
        0: 'r',
        1: 'b',
        2: 'g',
        3: 'y',
        4: 'c',
        5: 'm',
        6: 'k'
    }[index]

    return color

def compute_clusters(data, reference_vectors, k):
    clusters = [ [] for _ in range(k)]

    for obs in data:
        index = determine_cluster(obs, reference_vectors)
        clusters[index].append(obs)

    return clusters

def plot_kmeans(clusters, reference_vectors):
    for i, cluster in enumerate(clusters):
        color = get_color(i)

        for obs in cluster:
            #plt.scatter(obs[0], obs[1], c=color, marker='x', linewidths=1, s=10)
            ax.scatter(obs[0], obs[1], obs[2], c=color, marker='x', linewidths=1, s=10)

    for i, vec in enumerate(reference_vectors):
        color = get_color(i)
        #plt.scatter(vec[0], vec[1], c=color, marker="o", s=80, edgecolors='k')
        ax.scatter(vec[0], vec[1], vec[2], c=color, marker="o", s=100, edgecolors='k')

    return

def compute_reference_vectors(clusters):
    reference_vectors = []

    for i, cluster in enumerate(clusters):
        new_ref = np.sum(cluster, axis=0) / len(cluster)
        reference_vectors.append(new_ref)

    return reference_vectors

z = None
def init(*iteratoins):
    ax.set_title("Initial Setup")
    ax.scatter(z[:,0], z[:,1], z[:,2], marker='x', linewidths=1, s=10)
    plt.savefig("kmeans_intial_setup", bbox_inches="tight")
    return

def animate(i, *iterations):
    ax.clear()
    ax.set_title("Iteration " + str(i+1))

    print("I", i)
    print("len", len(iterations))

    if i is 0:
        plt.savefig("kmeans_first_iteration", bbox_inches="tight")

    if i is len(iterations) - 1:
        plt.savefig("kmeans_last_iteration", bbox_inches="tight")

    clusters = iterations[i]["clusters"]
    reference_vectors = iterations[i]["reference_vectors"]

    plot_kmeans(clusters, reference_vectors)

    return

def myKMeans(data, k):
    iterations = []

    num_obs = data.shape[0]
    random.seed(0)
    rand_inds = random.sample(range(0, num_obs), k)

    reference_vectors = []

    for i in rand_inds:
        reference_vectors.append(data[i])

    reference_vectors = np.array(reference_vectors)
    clusters = np.array(compute_clusters(data, reference_vectors, k))
    iterations.append({"clusters": clusters, "reference_vectors": reference_vectors})

    while True:
        new_reference_vectors = np.array(compute_reference_vectors(clusters))
        change = np.sum(np.abs(reference_vectors - new_reference_vectors))

        if change < (2 ** -23):
            break

        clusters = np.array(compute_clusters(data, new_reference_vectors, k))
        reference_vectors = new_reference_vectors
        iterations.append({"clusters": clusters, "reference_vectors": reference_vectors})

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=2000, frames=len(iterations), fargs=(iterations), blit=False, repeat=False)
    plt.show()

    #ani.save("test.avi", writer="ffmpeg", fps=0.5)
    print("done")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
results = np.array(read_files())
results = standardize_data(results)
cov = np.cov(results, rowvar=False)
projection_matrix = project(cov, 3)
z = np.real(np.matmul(results, projection_matrix))

myKMeans(z, 3)
