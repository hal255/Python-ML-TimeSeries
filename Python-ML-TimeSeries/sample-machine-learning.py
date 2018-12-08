
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TimeSeriesNaive:
    def __init__(self, csv_file):
        self.data_file = self.get_csv(csv_file)
        self.train = self.get_training(self.data_file)
        self.training_test = self.get_training_test(self.data_file)
        self.y_hat = self.get_y_hat(self.train, self.training_test)

    # import csv
    def get_csv(self, csv_file):
        return pd.read_csv(csv_file)

    def get_training(self, data_file):
        return data_file[0:10392]

    def get_training_test(self, data_file):
        return data_file[10392:]

    def get_y_hat(self, training, training_test):
        dd = np.asarray(training.Count)
        y_hat = training_test.copy()
        y_hat['naive'] = dd[len(dd) - 1]
        return y_hat


class TimeSeriesML:
    def __init__(self):
        self.alpha = 0
        self.level_0 = 0
        self.seasonal_0 = 0
        self.trend_0 = 0

        self.level = self.update_level(self.alpha)
        self.trend = None
        self.seasonal = None
        self.forecast = None

    def update_level(self, alpha):
        self.level = 1

    def get_level(self):
        return self.level



if __name__ == '__main__':
    naive_test = TimeSeriesNaive('train.csv')
    plt.figure(figsize=(12,8))
    plt.plot(naive_test.train.index, naive_test.train['Count'], label='train')
    plt.plot(naive_test.training_test.index, naive_test.training_test['Count'], label='training_test')
    plt.plot(naive_test.y_hat.index, naive_test.y_hat['naive'], label='naive forecast')
    plt.legend(loc='best')
    plt.title('Naive forecast')
    plt.show()
