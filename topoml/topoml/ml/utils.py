# Standard library imports

# Third party imports
import numpy as np

# Local application imports

#from topoml.ml.MSCSample import MSCSample


def gaussian_fit(image, plot=False):
    n, bins_ = np.histogram(image.flatten())
    mids = 0.5 * (bins_[1:] + bins_[:-1])
    mu = np.average(mids, weights=n)
    var = np.average((mids - mu) ** 2, weights=n)
    sigma = np.sqrt(var)
    # right_inflection = mu + sigma
    return mu, sigma, var  # , right_inflection

#split can be 'train', 'val', and 'test'
#this is the function that splits a dataset into training, validation and testing set
#We are using a split of 70%-10%-20%, for train-val-test, respectively
#this function is used internally to the defined dataset classes
def get_split(array_to_split, split):
    np.random.seed(0)
    np.random.shuffle(array_to_split)
    np.random.seed()
    if split == 'train':
        array_to_split = array_to_split[:int(len(array_to_split)*0.7)]
    elif split == 'val':
        array_to_split = array_to_split[int(len(array_to_split)*0.7):int(len(array_to_split)*0.8)]
    elif split == 'test':
        array_to_split = array_to_split[int(len(array_to_split)*0.8):]
    elif split is None:
        return array_to_split
    return array_to_split

