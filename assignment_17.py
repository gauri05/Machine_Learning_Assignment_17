import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def Advertising_Predic():

    # Load data
    data = pd.read_csv('Advertising.csv')

    print("Size of data set",data.shape)

    X=np.column_stack((data['TV'],data['radio'],data['newspaper']))
    Y=data['sales']
    #Z=data['newspaper'].values

    # divide the data set
    data_train, data_test, target_train, target_test = train_test_split(X, Y, test_size=0.5)

    reg=LinearRegression()

    #train
    reg=reg.fit(data_train,target_train)

    #test data
    y_pred =reg.predict(data_test)

    r2 =reg.score(X,Y)

    print(r2)

def main():
    print("Advertising Agency")
    Advertising_Predic()

if __name__=="__main__":
    main()