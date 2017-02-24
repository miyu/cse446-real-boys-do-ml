import numpy as np

FILES = "hands/hands1-3650"
frames = range(100)


images = np.load(FILES + "-images.npy")
labels = np.load(FILES + "-labels.npy")

np.save(FILES + "-images-reduced.npy", images[frames])
np.save(FILES + "-labels-reduced.npy", labels[frames])
