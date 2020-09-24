
import pandas as pd


def Swedish_Auto_Insurance():
    '''The Swedish Auto Insurance Dataset involves predicting the total payment for all 
    claims in thousands of Swedish Kronor, given the total number of claims.

    It is a regression problem. The variable names (X, Y) are as follows:
        1. Number of claims.
        2. Total payment for all claims in thousands of Swedish Kronor.
    
    The baseline performance of predicting the mean value is an RMSE of approximately 81 thousand Kronor.
    '''
    url = "https://www.math.muni.cz/~kolacek/docs/frvs/M7222/data/AutoInsurSweden.txt"
    df = pd.read_csv(url, skiprows = 10, header=0, delimiter='\t', decimal=',')

    return df['X'].values, df['Y'].values


