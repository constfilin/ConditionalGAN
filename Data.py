import os
import re
import numpy as np

from utils import *

#############################################################################################
# classes
#############################################################################################
class Data(object):

    def __init__(self,name,attribs,data,labels,labels_are_mutually_exclusive=False):
        self.name   = name
        self.attribs = attribs
        self.data    = data
        self.labels  = labels
        self.shape   = data[0].shape
        self.labels_are_mutually_exclusive = labels_are_mutually_exclusive

    def get_batch(self,step,batch_size):
        # The case when the length of data is less than the batch size is 
        # so unique that does not make sense to try to gneralize the code
        if batch_size>len(self.data):
            data   = self.data
            labels = self.labels
            while len(data)<batch_size:
                data  = np.concatenate((data,self.data))
                labels= np.concatenate((labels,self.labels))
            data   = data[0:batch_size]
            labels = labels[0:batch_size]
            perm = np.arange(batch_size)
            np.random.shuffle(perm)
            return data[perm],labels[perm]

        ro_num = (len(self.data)//batch_size) - 1
        if step % ro_num == 0:
            # Shuffle the data
            length = len(self.data)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.data   = self.data[perm]
            self.labels = self.labels[perm]
        ndx_min = (step%ro_num)*batch_size
        ndx_max = ndx_min+batch_size
        return self.data[ndx_min:ndx_max],self.labels[ndx_min:ndx_max]

    def get_number_of_labels(self):
        return self.labels.shape[1]

    def get_random_labels(self,shape):
        result = np.zeros(shape,dtype=np.float)
        for i in range(0,shape[0]):
            if self.labels_are_mutually_exclusive:
                ndx = np.random.randint(0,shape[1])
                result[i,ndx] = 1.
            else:
                result[i] = np.random.choice([0.,1.],size=shape[1])
        return result

    def get_labels_by_spec(self,shape,spec):
        result = np.zeros(shape,dtype=np.float)
        labels = spec.split(",")
        for i in range(0,shape[0]):
            for l in labels:
                result[i,self.attribs.index(l)] = 1.
        return result
        
    def describe_labels(self,labels):
        result = []
        for i in range(0,len(labels)):
            tmp = []
            for j in range(0,len(labels[i])):
                if labels[i,j]:
                    tmp.append(self.attribs[j])
            result.append("%i: %s" % (i,",".join(tmp)))
        return result

class Mnist(Data):

    def __init__(self,data_path):
        super(Mnist,self).__init__(
            "mnist",
            ["0","1","2","3","4","5","6","7","8","9"],
            *self.load(data_path,[28,28,1]),
            True)

    @staticmethod
    def load(data_path,shape):

        trX,trY = Mnist.load_from_files(os.path.join(data_path,'train-images-idx3-ubyte'),os.path.join(data_path,'train-labels-idx1-ubyte'),shape)
        teX,teY = Mnist.load_from_files(os.path.join(data_path,'t10k-images-idx3-ubyte'),os.path.join(data_path, 't10k-labels-idx1-ubyte'),shape)

        X = np.concatenate((trX,teX),axis=0)
        y = np.concatenate((trY,teY),axis=0)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        # Convert labels to one-hot. The constant 10 in this case comes from the number of
        # different labels in the data set. In case of MNIST there are 10 digits and 10 labels
        labels = np.zeros((len(y),10),dtype=np.float)
        for i,label in enumerate(y):
            labels[i,int(label)] = 1.0

        # Normalize the images vector
        return np.array(X/255.),labels

    @staticmethod
    def load_from_files(data_file,labels_file,shape):
        loaded = np.fromfile(data_file,dtype=np.uint8)
        data   = np.asarray(loaded[16:].reshape((-1,*shape)).astype(np.float))
        loaded = np.fromfile(labels_file,dtype=np.uint8)
        labels = np.asarray(loaded[8:].reshape((data.shape[0])).astype(np.float))
        return data,labels

class ImagesWithAttributes(Data):

    @staticmethod
    def load_attribs_and_annotations(attribs_file_path,limit=float('+inf')):
        attribs = []
        annotations  = {}
        with open(attribs_file_path) as f:
            count   = f.readline().strip()
            attribs = re.split("[ \t]+",f.readline().strip())
            for line in f:
                parts = re.split("[ \t]+",line.strip())
                if len(parts)==len(attribs)+1:
                    annotations[parts[0]] = list(map(lambda n : 1 if n=="1" else 0,parts[1:]))
                    if len(annotations)>=limit:
                        break
                else:
                    print("Found line '%s' in %s that has strange format" % (line,attribs_file_path))
        return attribs,annotations

    @staticmethod
    def load_attribs_data_and_labels(attribs_file_path,images_path,shape,limit=float('+inf')):
        attribs,annotations = ImagesWithAttributes.load_attribs_and_annotations(attribs_file_path,limit)
        data   = np.zeros((len(annotations), *shape),dtype=np.float)
        labels = np.zeros((len(annotations),len(attribs)),dtype=np.float)
        count  = 0
        images_fns  = os.listdir(images_path)
        images_fns.sort()
        for fn in images_fns:
            if fn in annotations:
                imagedata = read_image(os.path.join(images_path,fn),shape[:2])
                # for black/white images imread will return shape [h,w] instead of [h,w,1]. 
                # for color images imread returns [h,w,channels] and no reshaping is needed
                data[count]   = imagedata if tuple(data[count].shape)==tuple(imagedata.shape) else np.reshape(imagedata,data[count].shape)
                labels[count] = np.array(annotations[fn],dtype=np.float)
                count = count+1
                if count>=limit:
                    break
        return attribs,data,labels

class CelebA(ImagesWithAttributes):

    def __init__(self,data_path,limit):
        # The sizes of celebA imagez are 218x178 but the convolutional layer of ConditionalGAN requires
        # the dimensions to be divisible by 4. So we change the sizes
        super(CelebA,self).__init__(
            "celebA",
            # Layout assumed to be following https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8
            *self.load_attribs_data_and_labels(
                os.path.join(os.path.join(data_path,"Anno"),"list_attr_celeba.txt"),
                os.path.join(data_path,"img_align_celeba"),
                [216,176,3],
                limit))

class Wines(ImagesWithAttributes):

    def __init__(self,data_path):
        super(Wines,self).__init__(
            "wines",
            *self.load_attribs_data_and_labels(
                os.path.join(data_path,"attribs.txt"),
                data_path,
                [108,88,3]))

