
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

class TimeSeriesML:
    def __init__(self, train_csv, test_csv):
        # self.train = self.get_training(self.train_file)
        # self.training_test = self.get_training_test(self.test_file)
        self.train = self.get_csv(train_csv)
        self.test = self.get_csv(test_csv)
        self.y_hat = self.get_y_hat(self.train, self.test)

    # import csv
    def get_csv(self, csv_file):
        result = pd.read_csv(csv_file)
        return result

    # def get_training(self, data_file):
    #     train = data_file[0:10392]
    #     train.Timestamp = pd.to_datetime(train.Datetime, format='%d-%m-%y %H:%M')
    #     train.index = train.Timestamp
    #     return train
    #
    # def get_training_test(self, data_file):
    #     test = data_file[10392:]
    #     return test

    def get_y_hat(self, train, test):
        dd = np.asarray(train.Count)
        y_hat = test.copy()

        # Naive approach: set value of y_hat to previous day of y_hat
        # set all y_hat['naive'] values to last value of training['Count']
        y_hat['naive'] = dd[len(dd) - 1]

        # avg typically performs better than naive, but not in this case
        y_hat['avg_forecast'] = train['Count'].mean()

        # Computing simple exponential smoothing
        fit2 = SimpleExpSmoothing(np.asarray(train['Count'])).fit(smoothing_level=0.6, optimized=False)
        y_hat['SES'] = fit2.forecast(len(test))

        return y_hat

    def get_hourly(self):
        # Hourly
        return self.train.resample('H').mean()

    def get_daily(self):
        # Daily
        return self.train.resample('D').mean()

    def get_weekly(self):
        # Weekly
        weekly = self.train.resample('W').mean()

    def get_monthly(self):
        # Monthly
        monthly = self.train.resample('M').mean()


if __name__ == '__main__':
    naive_test = TimeSeriesML('train.csv', 'test.csv')
    plt.figure(figsize=(12, 8))     # size of view window
    plt.plot(naive_test.train.index, naive_test.train['Count'], label='train')                  # display train values
    plt.plot(naive_test.test.index, naive_test.test['Count'], label='training_test')  # display test
    # plt.plot(naive_test.y_hat.index, naive_test.y_hat['naive'], label='naive forecast')         # display naive values
    # plt.plot(naive_test.y_hat.index, naive_test.y_hat['avg_forecast'], label='avg_forecast')    # display avg values
    plt.plot(naive_test.y_hat.index, naive_test.y_hat['SES'], label='SES')    # display avg values
    plt.legend(loc='best')
    plt.title('Naive forecast')
    plt.show()
