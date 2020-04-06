# Make necessary imports
import quandl
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pickle 
import matplotlib.pyplot as pplt

FETCH_DATA = False

# Get Amazon stock data
if FETCH_DATA:
    amazon = quandl.get("WIKI/AMZN", api_key="ik4uZP8JA6jy4At-yyWH")
    dump_file = open('amazon_stock.dat', 'wb') 
    pickle.dump(amazon, dump_file)
else:
    dump_file = open('amazon_stock.dat', 'rb') 
    amazon = pickle.load(dump_file)


# Get only the data for the Adjusted Close column
print(amazon.head())
amazon = amazon[['Adj. Close']]

# Predict for 30 days; Predicted has the data of Adj. Close shifted up by 30 rows
forecast_len=30
amazon['Predicted'] = amazon[['Adj. Close']].shift(-forecast_len)

# Drop the Predicted column, turn it into a NumPy array to create dataset
x=np.array(amazon.drop(['Predicted'],1))
# Remove last 30 rows
x=x[:-forecast_len]

# Create dependent dataset for predicted values, remove the last 30 rows
y=np.array(amazon['Predicted'])
y=y[:-forecast_len]

# Split datasets into training and test sets (80% and 20%)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# Create SVR model and train it
svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1) 
svr_rbf.fit(x_train,y_train)

# Create Linear Regression model and train it
lr=LinearRegression()
lr.fit(x_train,y_train)

# Get score for SVR model
svr_rbf_confidence=svr_rbf.score(x_test,y_test)
print(f"SVR Confidence: {round(svr_rbf_confidence*100,2)}%")

# Get score for Linear Regression
lr_confidence=lr.score(x_test,y_test)
print(f"Linear Regression Confidence: {round(lr_confidence*100,2)}%")

# Predict data
svr_prediction = svr_rbf.predict(X=x_test)
lr_prediction = lr.predict(X=x_test)

pplt.plot(y_test[-30:])
pplt.plot(svr_prediction[-30:])
pplt.plot(lr_prediction[-30:])
pplt.show()




