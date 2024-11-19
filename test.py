images, labels = next(data_generator(...))
print(images.shape, images.dtype, np.min(images), np.max(images))
