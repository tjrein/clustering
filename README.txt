==== CS613 ====

  Author: Tom Rein
  Email: tr557@drexel.edu

==== Dependencies ====

  * Python3
  * pip3
  * numpy
  * pillow
  * matplotlib.pyplot

  In the event that dependencies are not installed, I have provided a "requirements.txt" file.
  To install dependencies, use type "pip3 install -r requirements.txt"


==== Files Included ====

  *read.py
  *yaleface.zip
  *pca.py
  *eigenface.py
  *kmeans.py


==== read.py ====

  This file contains functions that are shared across all the other files to read and standardize the yalefaces dataset.

  It consists of two functions:
    * read_images
    * standardize_data

    NOTE: read_images expects to read files out of the directory "./yalefaces". I have included the yalefaces dataset as a zip file.
          Unzipping should allow this to work by default. If not, please ensure the yalefaces is in the same directory as read.py


==== pca.py ====

  This script corresponds with Part 2 of the homework.

  To execute, type "python3 pca.py"

  It will invoke "read_images" to read in the yalefaces dataset.
  After performing PCA, it will display a visualization of the process.
  It will also save the image as "pca.png".


==== eigenfaces.py ====

  This script corresponds with Part 3 of the homework.

  To execute, type "python3 eigenfaces.py"

  It will invoke "read_images" to read in the yalefaces dataset.

  This script will display two figures:
    * Figure 1 is a visualization of the primary principal component
    * Figure 2 is a visualization of the reconstruction of face 1 using the primary component and k components.

  The script will save both figures as "primary.png" and "reconstruction.png" respectively.


==== kmeans.py ====

  This script corresponds with Part 4 of the homework.

  This script can be initialized with an optional integer argument to set the number of clusters.

  To execute, type "python3 kmeans.py {k}", where k is an integer.
  If no argument is passed, the script will default to using k=3

  The script will invoke "read_images" to read in the yalefaces dataset.
  It will then pass the data and k to the "myKmeans" function.

  The "myKmeans" function will perform PCA on the data when D > 3 and reduce to 3 dimensions.
  The function will instantiate a 2D or 3D plot depending on D.
  The function will display an animated figure of the clustering process.
  The function will also save three images, corresponding to the initial setup visualization, the first clustering iteration, and the final clustering iteration.
  These images will be saved as "kmeans_initial_setup.png", "kmeans_first_iteration.png", and "kmeans_last_iteration.png" respectively.

  NOTE: This function can also save the animation as a video.
        However, 'ffmpeg' is a requirement in order to do so.
        Since 'ffmpeg' is not installed on tux, I have this part commented out.

        To enable this functionality, look for the comment #SAVE TO VIDEO, which will explain how to do so.


==== KMEANS VIDEOS ====

  Although the "mykmeans" function allows for saving animations of videos, I ran into some issues with playback of the videos on laptop.
  When I upload the videos to YouTube however, they work perfectly.

  I spoke to Professor Burlick about this and said I could provide links to the videos.
  I am also including the files themselves in the submission, but these links should be used in the event the video has issues playing


  K_2.mp4:
  https://youtu.be/qG90K5fnh5o

  k_3.MP4:
  https://youtu.be/1jZhzWNHPdc

  K_5.mp4
  https://youtu.be/1jZhzWNHPdc
