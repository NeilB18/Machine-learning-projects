import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
df =pd.read_excel('data.xlsx')
x = df.iloc[:, :-1].values # ':' represents all the rows and ':-1' represents all the columns except the last one
y = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split 
# we make our vairables for training and testing the model.
# test_size = 0.2 means that we are goin to be using 20% of the dataset as our testing data.
# random_state = 0 is a way of making sure that we have the sam test set and trainging set everytime the model is run,
# because the function train_test_split randomly selects data therefore it is important for us to make sure we are using
# the same training and testing set
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
k =  x_test.shape[1]
n =  x_test.shape[0]
adjr2 = 1-(1-r2)*(n-1)/(n-k-1)
