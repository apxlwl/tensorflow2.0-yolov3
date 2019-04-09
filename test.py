from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.keras.saving.hdf5_format import *
from tensorflow import summary
import time
trainwriter = summary.create_file_writer(logdir='./summary/{}-{}/train'.format('test',
                                                                                          time.strftime(
                                                                                            "%m%d-%H-%M-%S",
                                                                                            time.localtime(
                                                                                              time.time()))))

root='/home/gwl/datasets/coco2017/images/val2017'
filelist=os.listdir(root)
with trainwriter.as_default():
  for i in range(5):
    print(i)
    img=np.array(Image.open(os.path.join(root,filelist[i])))
    pad=np.zeros(shape=(img.shape[0],20,3))
    print(pad.dtype)
    img=np.concatenate((img,pad),1)
    plt.imshow(img)
    plt.show()
    assert 0
    # tf.summary.image("detections", tf.expand_dims(tf.convert_to_tensor(img), 0),step=i)
