import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from time import gmtime, strftime

from keras.models import load_model

from data_loader.simple_mtl_data_loader import custom_data_loader
from models.mtl_model import build_model
from utils.data_processing import inference_input_processing

import warnings
warnings.filterwarnings("ignore")

LOG_LEVEL = 'info'
# configure logger
if len(logging.getLogger().handlers) > 0:
    # The Lambda environment pre-configures a handler logging to stderr. If a handler is already configured,
    # `.basicConfig` does not execute. Thus we set the level directly.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.getLevelName(LOG_LEVEL.upper()))
else:
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.getLevelName(LOG_LEVEL.upper()),
        datefmt='%Y-%m-%d %H:%M:%S')

log = logging.getLogger(__name__)



class ModelTraining:
    def __init__(self, training_data_loader, validation_data_loader, batch_size=12, epochs=10,
                 trained_model_path=''):
        # self.model = model
        self.training_data_loader = training_data_loader
        self.validation_data_loader = validation_data_loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_timestamp = strftime("%Y-%m-%dT%H:%M:%S", gmtime())
        self.models_folder = "./trained_models/"
        self.trained_model_path = os.path.join(self.models_folder, trained_model_path)
        self.model = build_model()

    def train(self):
        self.history = self.model.fit(
            self.training_data_loader,
            validation_data=self.validation_data_loader,
            epochs=self.epochs,
            batch_size=self.batch_size,
            steps_per_epoch=60//12, # TODO: refactor to dynamic
            validation_steps=1,
            validation_batch_size=self.batch_size,
            shuffle=True,
            verbose=2
        )
        logging.info("Model trained!")
        self.plot_both_task_loss()

    def _sigmoid_to_binary(self, x):
        if x > 0.5:
            return 1
        else:
            return 0

    def inference(self, input: np.array):
        """
        Prediction on trained model
        :param input:
        :return:
        """
        predictions = self.model.predict(input)
        y1 = self._sigmoid_to_binary(predictions[0][0][0])
        y2 = predictions[1][0][0]

        print("Task 1 prediction: %d \nTask 2 prediction: %.3f" % (y1, y2))


    def save_model(self):
        model_path = os.path.join(self.trained_model_path, self.model_timestamp + '.h5')
        self.model.save(model_path)
        logging.info("model saved to: %s" % model_path)

    def load_model(self, load_last_trained_model=False):
        """
        Load a trained model from a specified path or load the last trained model
        by setting load_last_trained_model=True
        :param load_last_trained_model: bool
        :return:
        """
        if load_last_trained_model:
            self.trained_model_path = self.get_newest_file_path()
            logging.info("Loaded latest model %s" % self.trained_model_path)
        try:
            self.model = load_model(self.trained_model_path)
            logging.info("Loaded model from path: %s" % self.trained_model_path)
        except OSError as e:
            logging.error(e)

    def get_newest_file_path(self):
        files = os.listdir(self.models_folder)
        paths = [os.path.join(self.models_folder, basename) for basename in files]
        return max(paths, key=os.path.getctime)

    def _get_model_params(self):
        """
        Extract model weights according to layers
        :return:
        """
        prefixes = ['Xb1', 'dense']
        weights = {layer.name: layer.get_weights() for layer in self.model.layers if layer.name.startswith(tuple(prefixes))}
        B1 = weights['Xb1'][0]
        b4 = weights[list(weights.keys())[1]][0][0]
        b3 = weights[list(weights.keys())[1]][0][1]
        B2 = B1 * b4

        print(" B1:\n {}\n B2:\n {}\n b3:\t {}\n b4:\t {}".format(B1, B2, b3, b4))

    def plot_both_task_loss(self):
        loss_values = self.history.history['loss']
        val_loss_values = self.history.history['val_loss']
        epochs = range(1, len(loss_values) + 1)

        # plot
        plt.plot(epochs, loss_values, '--o', label='Training loss both tasks')
        plt.plot(epochs, val_loss_values, '-X', label='Validation both tasks')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

    def plot_task1_loss(self):
        loss_values = self.history.history['task_1_loss']
        val_loss_values = self.history.history['val_task_1_loss']
        epochs = range(1, len(loss_values) + 1)

        # plot
        plt.plot(epochs, loss_values, '--o', label='Training loss task 1')
        plt.plot(epochs, val_loss_values, '-X', label='Validation task 1')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

    def plot_task1_acc(self):
        loss_values = self.history.history['task_1_accuracy']
        val_loss_values = self.history.history['val_task_1_accuracy']
        epochs = range(1, len(loss_values) + 1)

        # plot
        plt.plot(epochs, loss_values, '--o', label='Training accuracy task 1')
        plt.plot(epochs, val_loss_values, '-X', label='Validation accuracy task 1')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    def plot_task2_loss(self):
        loss_values = self.history.history['task_2_loss']
        val_loss_values = self.history.history['val_task_2_loss']
        epochs = range(1, len(loss_values) + 1)

        # plot
        plt.plot(epochs, loss_values, '--o', label='Training loss task 2')
        plt.plot(epochs, val_loss_values, '-X', label='Validation task 2')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()


if __name__ == '__main__':
    # load dataset
    data = pd.read_csv("./data/data.csv")
    N = 900000
    batch_size = 12 # define model training batch size
    data_loader_batch_size = int(batch_size / 2) # batch size for each task (number of samples from each task must be equal in final batch)

    # Simple manual split to train and validation datasets
    ## TODO: furhter improvments dynamic train and validation split in one generator
    df_train = data.iloc[batch_size:N, :]
    df_val = pd.concat([data.iloc[:batch_size, :], data.iloc[N:N + batch_size, :]], ignore_index=True)

    # Crate gernerator objects for train and validation
    training_data_loader = custom_data_loader(df_train, batchSize=data_loader_batch_size)
    validation_data_loader = custom_data_loader(df_val, batchSize=data_loader_batch_size)

    # Define model class object
    MTL_model = ModelTraining(training_data_loader, validation_data_loader, epochs=60)
    # Train model
    MTL_model.train()
    # Save trained model
    MTL_model.save_model()

test = pd.DataFrame(data=[[5.62, 1.26, -0.49, 6.63, 3.77, -0.13, -0.91]], columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'z'])
X = inference_input_processing(test)

#
# MTL_model.load_model()
# MTL_model.inference(X)
# MTL_model._get_model_params()
# keras.utils.plot_model(MTL_model, show_dtype=True,
#                        show_layer_names=True, show_shapes=True,
#                        to_file='model.png')
