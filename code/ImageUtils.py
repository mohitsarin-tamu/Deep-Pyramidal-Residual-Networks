import numpy as np
from matplotlib import pyplot as plt


"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    ### YOUR CODE HERE
    
    # reshape the image
    depth_major = record.reshape((3, 32, 32))
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)
    
    #converting from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])
    

    ### END CODE HERE


    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [3, 32, 32].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32]. The processed image.
    """
    ### YOUR CODE HERE
    
    if training:

        # resize the image
        image = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode='constant')
 
        # crop a [32, 32] section from image

        crop_x = np.random.randint(0, 9)
        crop_y = np.random.randint(0, 9)
        image = image[crop_x:crop_x+32, crop_y:crop_y+32, :]
        
        #randonly removing some part of the image: cutout
        
        length = 16
        h, w, _ = image.shape
        x = np.random.randint(0, h)
        y = np.random.randint(0, w)

        x1 = np.clip(x - length // 2, 0, h)
        x2 = np.clip(x + length // 2, 0, h)
        y1 = np.clip(y - length // 2, 0, w)
        y2 = np.clip(y + length // 2, 0, w)

        image[x1:x2, y1:y2, :] = 0
        
        # flip the image horizontally.
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
    
    # calculate mean and standard deviation   
    image = (image - np.mean(image, axis=(0, 1))) / np.std(image, axis=(0, 1))
    
    ### END CODE HERE

    return image


def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    # reshape the image to [32, 32, 3]
    image_x = image.reshape(3, 32, 32).transpose(1, 2, 0)
    
    # converting to int to visualize the image properly
    image = (image_x * 255).astype(np.uint8)
    
    ### YOUR CODE HERE
    
    plt.imshow(image)
    plt.savefig(save_name)
    return image

# Other functions
### YOUR CODE HERE

### END CODE HERE

