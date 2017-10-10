import tensorflow as tf
import numpy as np
import cv2

from began.trainer import Trainer
from began.config import get_config
from began.data_loader import get_loader
from began.utils import prepare_dirs_and_logger, save_config

class GANWrapper:
    def __init__(self):
        config, unparsed = get_config()
        prepare_dirs_and_logger(config)

        config.data_path = "../data/capture"
        config.model_dir = "began/logs/faces_model"
        config.is_train = False
        data_path = config.data_path
        config.batch_size = 1
        data_loader = get_loader(
                    data_path, config.batch_size, config.input_scale_size,
                    config.data_format, config.split)
        self.trainer = Trainer(config, data_loader)

    def autoencode(self, img):
        #real1_batch = trainer.get_image_from_loader()
        img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_CUBIC)
        img = np.asarray(img).astype('float')
        img = np.expand_dims(img, axis=0)
        #img = np.transpose(img, [0, 3, 1, 2])
        #img = tf.expand_dims(img, -1)
        #img = tf.image.resize_nearest_neighbor(img, [64, 64])
        #img = tf.to_float(img)
        result = self.trainer.autoencode(img)
        return result[0][0]
