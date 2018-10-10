import os
import numpy as np

class Data(object):

    def __init__(self,name,shape):
        self.name   = name
        self.shape  = shape

    def load_from_files(self,data_file,labels_file):
        loaded = np.fromfile(data_file,dtype=np.uint8)
        data   = np.asarray(loaded[16:].reshape((-1,*self.shape)).astype(np.float))
        loaded = np.fromfile(labels_file,dtype=np.uint8)
        labels = np.asarray(loaded[8:].reshape((data.shape[0])).astype(np.float))
        return data,labels

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

class MnistData(Data):

    def __init__(self):
        super(MnistData,self).__init__("mnist",[28,28,1])
        self.data,self.data_y = self.load(os.path.join("data","mnist"))

    def load(self,data_dir):

        trX,trY = self.load_from_files(os.path.join(data_dir,'train-images-idx3-ubyte'),os.path.join(data_dir,'train-labels-idx1-ubyte'))
        teX,teY = self.load_from_files(os.path.join(data_dir,'t10k-images-idx3-ubyte'),os.path.join(data_dir, 't10k-labels-idx1-ubyte'))

        X = np.concatenate((trX,teX),axis=0)
        y = np.concatenate((trY,teY),axis=0)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        # Convert label to one-hot. The constant 10 in this case comes from the number of
        # different labels in the data set. In case of MNIST there are 10 digits and 10 labels
        y_vec = np.zeros((len(y),10),dtype=np.float)
        for i,label in enumerate(y):
            y_vec[i,int(label)] = 1.0

        # Normalize the images vector
        return np.array(X/255.),y_vec

class CelebAData(Data):

    def __init__(self):
        super(CelebAData,self).__init__("celebA",[218,178,3])
        self.data,self.data_y = self.load(os.path.join("data",self.name))

    def load(self,data_dir):
        raise Exception("Not ready")
