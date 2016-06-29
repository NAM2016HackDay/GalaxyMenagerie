Uses Tensorflow to classify galaxies by the imagenet image classifier to hilarious ends.

images/ must contain square jpg images of size 100x100px where each images is rotated four times (each rotation needs appending with _1, _2, _3, or _4), i.e. images/Galaxy_4.jpg, images/Galaxy_4.jpg, images/Galaxy_4.jpg, and images/Galaxy_4.jpg. Note, there can only be ONE underscore in the image name.

Run with python classify_images.py. Output is in out/out.txt
