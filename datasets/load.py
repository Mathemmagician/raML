
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


def Wine_Quality():
    '''The Wine Quality Dataset involves predicting the quality of white wines on 
    a scale given chemical measures of each wine.

    There are 11 feature columns (Density, Acidity, etc.), and a Quality column with a total of 4898 observation.

    The baseline performance of predicting the mean value is an RMSE of approximately 0.148 quality points.
    '''
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    df = pd.read_csv(url, delimiter=';')

    X, Y = df[df.columns[:-1]].values, df[df.columns[-1]].values
    
    return X.transpose(), Y


def Pima_Indians_Diabetes():
    '''The Pima Indians Diabetes Dataset involves predicting the onset of diabetes 
    within 5 years in Pima Indians given medical details.

    It is a binary (2-class) classification problem. The number of observations for 
    each class is not balanced. There are 768 observations with 8 input variables 
    and 1 output variable. Missing values are believed to be encoded with zero values.

    The baseline performance of predicting the most prevalent class is a classification 
    accuracy of approximately 65%. Top results achieve 77% accuracy.

    Columns:
        1. Number of times pregnant.
        2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
        3. Diastolic blood pressure (mm Hg).
        4. Triceps skinfold thickness (mm).
        5. 2-Hour serum insulin (mu U/ml).
        6. Body mass index (weight in kg/(height in m)^2).
        7. Diabetes pedigree function.
        8. Age (years).
        9. Class variable (0 or 1).
    '''
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
    df = pd.read_csv(url)

    X, Y = df[df.columns[:-1]].values, df[df.columns[-1]].values
    return X, Y


if __name__ == '__main__':
    print(Pima_Indians_Diabetes())