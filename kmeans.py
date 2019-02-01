import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from read import read_images, standardize_data

def perform_pca(data):
    data = standardize_data(data)
    cov = np.cov(data, rowvar=False)
    projection_matrix = project(cov, 3)
    z = np.real(np.matmul(data, projection_matrix))
    return z

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

def plot_kmeans(clusters, reference_vectors, ax, d):
    for i, cluster in enumerate(clusters):
        color = get_color(i)

        for obs in cluster:
            if d < 3:
                ax.scatter(obs[0], obs[1], c=color, marker='x', linewidths=1, s=10, alpha=0.5)
            else:
                ax.scatter(obs[0], obs[1], obs[2], c=color, marker='x', linewidths=1, s=10, alpha=0.5)

    for i, vec in enumerate(reference_vectors):
        color = get_color(i)
        if d < 3:
            ax.scatter(vec[0], vec[1], c=color, marker="o", s=80, edgecolors='k', alpha=1.0)
        else:
            ax.scatter(vec[0], vec[1], vec[2], c=color, marker="o", s=100, edgecolors='k', alpha=1.0)
    return

def compute_reference_vectors(clusters):
    reference_vectors = []

    for i, cluster in enumerate(clusters):
        new_ref = np.sum(cluster, axis=0) / len(cluster)
        reference_vectors.append(new_ref)

    return reference_vectors

def animate(i, d, *iterations):
    if d < 3:
        ax = plt.subplot(111)
    else:
        ax = plt.subplot(111, projection='3d')

    ax.clear()
    ax.set_title("Iteration " + str(i+1))

    clusters = iterations[0][i]["clusters"]
    reference_vectors = iterations[0][i]["reference_vectors"]

    plot_kmeans(clusters, reference_vectors, ax, d)

    if i is 0:
        #First iteration figure
        plt.savefig("kmeans_first_iteration", bbox_inches="tight")
    if i is len(iterations[0]) - 1:
        #Last iteration figure
        plt.savefig("kmeans_last_iteration", bbox_inches="tight")

    return

def init_closure(reference_vectors, data, d):
    def init():
        if d < 3:
            ax = plt.subplot(111)
            ax.scatter(data[:,0], data[:,1])
        else:
            ax = plt.subplot(111, projection='3d')
            ax.scatter(data[:,0], data[:,1], data[:,2], marker='x', linewidths=1, s=10, alpha=0.5)

        for i, vec in enumerate(reference_vectors):
            c = get_color(i)
            ax.scatter(vec[0], vec[1], vec[2], marker='o', s=100, c=c, edgecolors='k', alpha=1.0)

        ax.set_title("Initial Setup")

        #Initial Setup Visualization Figure
        plt.savefig("kmeans_initial_setup", bbox_inches="tight")

    return init

def myKMeans(data, k):
    iterations = []
    reference_vectors = []

    if data.shape[1] > 3:
        data = perform_pca(data)

    fig = plt.figure()
    num_obs, d = data.shape
    random.seed(0)
    rand_inds = random.sample(range(0, num_obs), k)

    for i in rand_inds:
        reference_vectors.append(data[i])

    reference_vectors = np.array(reference_vectors)
    animate_init = init_closure(reference_vectors, data, d)
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

    ani = animation.FuncAnimation(fig, animate, interval=2000, init_func=animate_init, frames=len(iterations), fargs=(d, iterations), blit=False, repeat=False)

    #To save animation as video, comment out plt.show and uncomment ani.save
    plt.show()
    #ani.save(f"K_{k}.mp4", writer="ffmpeg", fps=0.5)

def main():
    args = sys.argv
    k = 3

    if len(args) > 1 and args[1].isdigit():
        k = int(args[1])

    data = np.array(read_images())
    myKMeans(data, k)

if __name__ == '__main__':
    main()
