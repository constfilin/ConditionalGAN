import numpy as np
import imageio
import scipy
import scipy.misc

def get_unique_filename( sample_path ):
    # TODO: do this in a more civilized manner
    for i in range(0,10000):
        image_path = "./{}/test{:02d}_{:04d}.png".format(sample_path,0,i)
        if not os.path.isfile(image_path):
            return image_path
    raise Exception("Cannot find unique file name in {:s}".format(sample_path))

def reshape_to_rectangle(images,rectangle_shape):
    h, w = images.shape[1],images.shape[2]
    # The images might me black/white (1 channels) or color (3 channels). We allocate
    # for 3 channels (thereby converting any blck/white images to color)
    result = np.zeros((int(h*rectangle_shape[0]),int(w*rectangle_shape[1]),3))
    for idx, image in enumerate(images):
        i = idx %  rectangle_shape[1]
        j = idx // rectangle_shape[1]
        result[j*h:j*h+h, i*w:i*w+w, :] = image
    return result

def reshape_to_square(images):
    # Here images are in shape (some_size,height,width,channels)
    # I.e. we have some number (most likely batch size) of images all stretched out in
    # one long line. It is more convenient to re-arrange the images into a smallest
    # square that can fit them all
    square_size = int(np.sqrt(int(images.shape[0])))
    if square_size*square_size<images.shape[0]:
        square_size = square_size+1
    return reshape_to_rectangle(images,[square_size,square_size])

def crop_image_center(image,crop_shape):
    crop_h,crop_w = crop_shape
    h, w = image.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(image[j:j+crop_h,i:i+crop_w],[crop_h,crop_w])

def read_image(image_path,crop_shape=None):
    result = imageio.imread(image_path).astype(np.float)
    if crop_shape is not None:
        result = crop_image_center(result,crop_shape)
    return result/255.

def save_image(image_path,image):
    return imageio.imsave(image_path,(255.*image).astype(np.uint8))
