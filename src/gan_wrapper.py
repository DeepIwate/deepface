import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from began.trainer import Trainer
from began.config import get_config
from began.data_loader import get_loader
from began.utils import prepare_dirs_and_logger, save_config

class GANWrapper:
    def __init__(self):
        config, unparsed = get_config()
        prepare_dirs_and_logger(config)

        config.data_path = "../data/capture"
        config.model_dir = "began/logs/celebA"
        config.is_train = False
        data_path = config.data_path
        config.batch_size = 1
        self.__data_loader = get_loader(
                    data_path, config.batch_size, config.input_scale_size,
                    config.data_format, config.split)
        self.trainer = Trainer(config, self.__data_loader)
        self.config = config
        self.data_loader = None

    def autoencode(self, img, imgPath=None):
        if imgPath is None or self.trainer.sess is None:
            #real1_batch = trainer.get_image_from_loader()
            img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_CUBIC)
            img = np.asarray(img).astype('float')
            img = np.expand_dims(img, axis=0)
            #img = img/255.0
            #print(img)
            #img = np.transpose(img, [0, 3, 1, 2])
            #img = tf.expand_dims(img, -1)
            #img = tf.image.resize_nearest_neighbor(img, [64, 64])
            #img = tf.to_float(img)
        else:
            img = mpimg.imread("tempdir/tmp.png")

            img = np.expand_dims(img, axis=0)

            x = tf.Variable(img, name='x')
            model = tf.global_variables_initializer()
            #x.set_shape([1, 64, 64, 3])

            x = tf.image.resize_nearest_neighbor(x, [64, 64])
            x = tf.transpose(x, [0, 3, 1, 2])
            x = tf.to_float(x) * 255

            self.trainer.sess.run(model)
            img = x.eval(session=self.trainer.sess)
            print(img)
            if True:
                img = img.transpose([0, 2, 3, 1])
            # if self.data_loader is None:
            #     self.data_loader = get_loader(
            #                 imgPath, 1, self.config.input_scale_size,
            #                 self.config.data_format, self.config.split)
            #     tf.train.start_queue_runners(sess=self.trainer.sess)
            # img = self.get_image_from_loader(self.data_loader)
            tf.train.start_queue_runners(sess=self.trainer.sess)
            img = self.get_image_from_loader(self.__data_loader)
            #plt.imshow(img[0])
            #plt.show()

        result = self.trainer.autoencode(img)
        result = result[0][0]
        #plt.imshow(result[0][0])
        #plt.show()
        return result

    def get_image_from_loader(self, data_loader):
        x = data_loader.eval(session=self.trainer.sess)
        if True:
            x = x.transpose([0, 2, 3, 1])
        return x
