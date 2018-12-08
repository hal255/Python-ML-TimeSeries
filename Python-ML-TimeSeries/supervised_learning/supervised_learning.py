
"""
    following example: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
"""

from pandas import DataFrame


def main():
    df = DataFrame()
    df['t'] = [x for x in range(10)]
    df['t-1'] = df['t'].shift(1)
    print(df)


if __name__ == '__main__':
    main()
