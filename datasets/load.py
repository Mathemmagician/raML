
import pandas as pd


def Swedish_Auto_Insurance():
    df = pd.read_csv('Swedish_Auto_Insurance.csv', skiprows = 10, header=0, delimiter='\t', decimal=',')

    return df['X'].values, df['Y'].values


if __name__ == '__main__':
    Swedish_Auto_Insurance()