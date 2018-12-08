
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
        result = pd.read_csv(csv_file)
        return result

    def get_training(self, data_file):
        result = data_file[0:10392]
        return result

    def get_training_test(self, data_file):
        result = data_file[10392:]
        return result

    def get_y_hat(self, training, training_test):
        dd = np.asarray(training.Count)
        y_hat = training_test.copy()

        # Naive approach: set value of y_hat to previous day of y_hat
        # set all y_hat['naive'] values to last value of training['Count']
        y_hat['naive'] = dd[len(dd) - 1]

        return y_hat


if __name__ == '__main__':
    naive_test = TimeSeriesNaive('train.csv')
    plt.figure(figsize=(12, 8))     # size of view window
    plt.plot(naive_test.train.index, naive_test.train['Count'], label='train')                  # display train values
    plt.plot(naive_test.training_test.index, naive_test.training_test['Count'], label='training_test')  # display test
    plt.plot(naive_test.y_hat.index, naive_test.y_hat['naive'], label='naive forecast')         # display naive values
    plt.legend(loc='best')
    plt.title('Naive forecast')
    plt.show()
