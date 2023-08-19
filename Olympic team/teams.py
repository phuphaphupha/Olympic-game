import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression


#load datatset
teams = pd.read_csv('/Users/phupha/Desktop/teams.csv')

#remove column
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals", ]]
#print(teams.head())

#see the correlation between data
#print(teams.corr()["medals"]) (all coulumn has to be in number, cannot be just string)

'''

#plotting to see how the data shows
sns.lmplot(x = "athletes", y="medals", data=teams, fit_reg=True, ci = None)
sns.lmplot(x = "prev_medals", y="medals", data=teams, fit_reg=True, ci= None)
teams.plot.hist(y="medals")
plt.show()

'''

#find rows with missing value
teams[teams.isnull().any(axis=1)]
#drop row without value
teams = teams.dropna()

#split data set
train = teams[teams["year"]<2012].copy()
test = teams[teams["year"]>=2012].copy()
print(train.shape, test.shape)

#train the model
reg = LinearRegression()
predictors = ["athletes", "prev_medals"]
target = "medals"
reg.fit(train[predictors], train[target])
predictions = reg.predict(test[predictors])
#add column of prediction to the test data set
test["predictions"] = predictions

#correct the prediction (as some prediction is invalid, negative number of medals)
test.loc[test["predictions"]<0, "predictions"]=0
test["predictions"] = test["predictions"].round()
#print(test)

#evaluate the prediction by mean square error - within error output
error = mean_absolute_error(test["medals"], test["predictions"])
print(error)

#check whether the error that we achieve is reasonable or not
#first check: error that we obtain should be below standard deviation
print(teams.describe()["medals"])

#look team by team (row that we interested)
print(test[test["team"]== "USA"])
