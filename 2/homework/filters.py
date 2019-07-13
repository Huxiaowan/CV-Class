import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from time import time


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # YOUR CODE HERE
    out1 = np.zeros((Hi + Hk - 1, Wi + Wk - 1))

    for i in range(0, Hi + Hk - 1):
        for j in range(0, Wi + Wk - 1):
            temp = 0.0
            for m in range(0, Hi):
                for n in range(0, Wi):
                    if (((i - m) >= 0)and ((i - m) < Hk) and ((j - n) >= 0)and((j - n) < Wk)):
                        temp += image[m][n] * kernel[i - m][j - n]
            out1[i][j] = temp

    for i in range(0, Hi):
        for j in range(0, Wi):
            out[i][j] = out1[i + (Hk - 1) / 2][j + (Wk - 1) / 2]
            pass
        pass
    # END YOUR CODE

    return out




def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    # YOUR CODE HERE
    out = np.pad(image, ((pad_height, pad_height), (pad_width,
                                                    pad_width)), 'constant', constant_values=(0, 0))
    pass
    # END YOUR CODE
    return out

def replica_pad(image, pad_height, pad_width):

    H, W = image.shape
    out = None

    # YOUR CODE HERE
    out = np.pad(image, ((pad_height, pad_height), (pad_width,
                                                    pad_width)), 'constant', constant_values=(image[0][0],image[H-1][W-1]))
    pass
    # END YOUR CODE
    return out



def conv_fast(image, kernel,padding_way):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    if padding_way  = 'zero':
        image = zero_pad(image, Hk - 1, Wk - 1)
    elif padding_way     = 'replica':
        image = replica_pad (image, Hk - 1, Wk - 1)
    
    kernel = np.fliplr(kernel)
    kernel = np.flipud(kernel)
    out1 = np.zeros((Hi + Hk - 1, Wi + Wk - 1))
    for i in range(0, Hi + Hk - 1):
        for j in range(0, Wi + Wk - 1):
            temp = 0.0
            for m in range(0, Hk):
                for n in range(0, Wk):
                    temp += kernel[m][n] * image[m + i][n + j]
            out1[i][j] = temp

    for i in range(0, Hi):
        for j in range(0, Wi):
            out[i][j] = out1[i + (Hk - 1) / 2][j + (Wk - 1) / 2]
            pass
        pass
    pass
    # END YOUR CODE

    return out



kernel = np.array(
    [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
img = io.imread('dog.jpg', as_grey=True)
out = conv_nested(img, kernel)

# Plot original image
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

# Plot your convolved image
plt.subplot(1, 2, 2)
plt.imshow(out)
plt.title('Convolution')
plt.axis('off')


plt.show()