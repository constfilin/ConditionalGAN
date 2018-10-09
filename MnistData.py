import os
import numpy as np

class MnistData(object):

    def __init__(self):
        self.shape = [28,28,1] # This is hardcoded in the downloaded data
        self.data, self.data_y = self.load(os.path.join("./data", "mnist"))

    def load(self,data_dir):

        def load_images_and_labels( images_path, labels_path ):
            loaded = np.fromfile(images_path,dtype=np.uint8)
            images = loaded[16:].reshape((-1,*self.shape)).astype(np.float)
            loaded = np.fromfile(labels_path,dtype=np.uint8)
            labels = loaded[8:].reshape((images.shape[0])).astype(np.float)
            labels = np.asarray(labels)
            return images,labels

        trX,trY = load_images_and_labels(os.path.join(data_dir,'train-images-idx3-ubyte'),os.path.join(data_dir,'train-labels-idx1-ubyte'))
        teX,teY = load_images_and_labels(os.path.join(data_dir,'t10k-images-idx3-ubyte'),os.path.join(data_dir, 't10k-labels-idx1-ubyte'))

        X = np.concatenate((trX,teX),axis=0)
        y = np.concatenate((trY,teY),axis=0)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        # convert label to one-hot
        y_vec = np.zeros((len(y),10),dtype=np.float)
        for i,label in enumerate(y):
            y_vec[i,int(label)] = 1.0

        # Normalize the images vector
        return np.array(X/255.),np.array(y_vec)

    def get_next_batch(self, iter_num, batch_size ):
        ro_num = (len(self.data)//batch_size) - 1
        if iter_num % ro_num == 0:
            # Shuffle the data
            length = len(self.data)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.data   = self.data[perm]
            self.data_y = self.data_y[perm]

        ndx_min = (iter_num%ro_num)*batch_size
        ndx_max = ndx_min+batch_size
        return self.data[ndx_min:ndx_max],self.data_y[ndx_min:ndx_max]
