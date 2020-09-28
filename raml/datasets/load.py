
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


def Sonar():
    '''The Sonar Dataset involves the prediction of whether or not an object is a mine or 
    a rock given the strength of sonar returns at different angles.

    Imbalanced binary classicifaction problem with 208 observations, 60 features 
    and 1 output variable (M for mine and R for rock).

    The baseline performance of predicting the most prevalent class is a classification accuracy 
    of approximately 53%. Top results achieve a classification accuracy of approximately 88%.
    '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
    raise "Not Finished"

def Boston_House_Price():
    '''The Boston House Price Dataset involves the prediction of a house price in thousands 
    of dollars given details of the house and its neighborhood.

    There are 506 observations with 13 input variables and 1 output variable. 

        CRIM: per capita crime rate by town.
        ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
        INDUS: proportion of nonretail business acres per town.
        CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
        NOX: nitric oxides concentration (parts per 10 million).
        RM: average number of rooms per dwelling.
        AGE: proportion of owner-occupied units built prior to 1940.
        DIS: weighted distances to five Boston employment centers.
        RAD: index of accessibility to radial highways.
        TAX: full-value property-tax rate per $10,000.
        PTRATIO: pupil-teacher ratio by town.
        B: 1000(Bk – 0.63)^2 where Bk is the proportion of blacks by town.
        LSTAT: % lower status of the population.
        MEDV: Median value of owner-occupied homes in $1000s.

    The baseline performance of predicting the mean value is an RMSE of approximately 9.21 thousand dollars.
    '''
    
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data'
    df = pd.read_csv(url, header=None, delim_whitespace=True)

    X, Y = df[df.columns[:-1]].values, df[df.columns[-1]].values
    return X, Y


if __name__ == '__main__':
    print(Pima_Indians_Diabetes())