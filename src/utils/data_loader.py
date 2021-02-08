from mlxtend.data import loadlocal_mnist
import numpy as np
import pandas as pd
import os


class GestureLoader:

    cwd = os.getcwd()

    def __init__(self, scaling, features_to_keep):
        self.t_train = pd.read_csv(self.cwd + "/data/gesture_targets.csv")
        self.x_train = pd.read_csv(self.cwd + "/data/demo_video_frames.csv")
        self.scaling = scaling
        self.features_to_keep = features_to_keep

    def data_preprocessing(self, data):
        frame_diff = np.vstack((data[1:] - data[:-1], np.zeros(data.shape[1])))
        frame_diff = np.interp(frame_diff, (frame_diff.min(), frame_diff.max()), (-1, 1))
        rolled_diffs = np.hstack([data - np.roll(data, -i) for i in range(1, data.shape[1])])
        rolled_diffs = np.interp(rolled_diffs, (rolled_diffs.min(), rolled_diffs.max()), (-1, 1))
        return np.hstack((frame_diff, rolled_diffs))


    def get_train_data(self):
        timestamp = self.x_train.iloc[:, 0]
        timestamp = timestamp.iloc[1:timestamp.shape[0]].to_numpy().astype(float).astype(int)

        # drop all confidence columns and the ones we don't want to keep
        data = self.x_train.drop([col_name for col_name in list(self.x_train.columns) if col_name.endswith('.2') or not
                                 col_name.startswith(self.features_to_keep)], axis=1)

        data = data.iloc[1:].to_numpy().astype(float)
        data = self.data_preprocessing(data)

        targets = np.zeros(timestamp.shape[0])
        interv_index = 0
        t_index = 0
        for i in timestamp:
            curr_interv = self.t_train.iloc[interv_index]
            targets[t_index] = curr_interv.target_id
            if i >= curr_interv.end:
                interv_index += 1
            t_index += 1
        targets = targets.astype(int)
        # transform labels into one hot representation
        targets_one_hot = np.identity(targets.max() + 1, dtype=float)[targets]

        # we don't want zeroes and ones in the labels neither:
        targets_one_hot[targets_one_hot == 0] = 0.01
        targets_one_hot[targets_one_hot == 1] = 0.99

        return data * self.scaling, targets_one_hot


class MnistLoader:

    cwd = os.getcwd()

    def __init__(self, scaling):
        self.x_train, self.t_train = loadlocal_mnist(
            images_path=self.cwd + '/data/train-images.idx3-ubyte',
            labels_path=self.cwd + '/data/train-labels.idx1-ubyte')

        self.x_test, self.t_test = loadlocal_mnist(
            images_path=self.cwd + '/data/t10k-images.idx3-ubyte',
            labels_path=self.cwd + '/data/t10k-labels.idx1-ubyte')

        self.scaling = scaling

    def get_train_data(self):
        t_train = np.array(self.t_train)

        # transform labels into one hot representation
        targets_one_hot = np.identity(t_train.max() + 1, dtype=float)[t_train]

        # we don't want zeroes and ones in the labels neither:
        targets_one_hot[targets_one_hot == 0] = 0.01
        targets_one_hot[targets_one_hot == 1] = 0.99

        return np.asfarray(self.x_train) * self.scaling + 0.01, targets_one_hot

    def get_test_data(self):
        t_train = np.array(self.t_train)

        # transform labels into one hot representation
        targets_one_hot = np.identity(t_train.max() + 1, dtype=float)[t_train]

        # we don't want zeroes and ones in the labels neither:
        targets_one_hot[targets_one_hot == 0] = 0.01
        targets_one_hot[targets_one_hot == 1] = 0.99

        return np.asfarray(self.x_train) * self.scaling + 0.01, targets_one_hot
