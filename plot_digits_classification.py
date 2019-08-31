# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
# for index, (image, label) in enumerate(images_and_labels[:4]):
#     plt.subplot(2, 4, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# function for converting RGB data into grayscale one
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
# Now predict the value of the digit on the second half:
#expected = digits.target[n_samples // 2:(n_samples // 2)+ 1]
# predicted = classifier.predict(data[n_samples // 2:(n_samples // 2)+ 1])

# image downloaded and compressed 64 RGB bit matrix
image = plt.imread('image_nine.png', format=None)
gray = rgb2gray(image)
a = (16-gray*16).astype(int)  # really weird here, but try to convert to 0..16

# can see the image in gray scale plot

# plt.imshow(a, cmap=plt.get_cmap('gray_r'))
# plt.show()
# print("source data in 8x8:\n", a)

# just for comparison with data set single image
# sample = data[n_samples // 2: (n_samples // 2)+1]
# print('SAMPLE IMAGE', sample, sample.shape)

array = a.flatten()
predicted = classifier.predict(array.reshape(1,-1))

# print('predicted for our test data',predicted );
print('predicted', predicted)


# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
# for index, (image, prediction) in enumerate(images_and_predictions[:4]):
#     plt.subplot(2, 4, index + 5)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Prediction: %i' % prediction)

# plt.show()

##############################################################

# In this file we can see model is mapped using degits data from the sci-kit dataset and 
# a random digit data is picked from google images and processed.

# initially higher pixel data is converted into 8*8 pixel rgb data, which is 3-dimensional
# it is then converted into gray scale image by the user define function

# then it is passed to classifier for Prediction
# result:

# here the value : 9
# prediction came out to be : 7
