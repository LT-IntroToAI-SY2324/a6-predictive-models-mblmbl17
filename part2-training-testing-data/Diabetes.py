import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = load_diabetes(as_frame= True)
y = data.target.values
data = data.frame
x = data["bmi"].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)

# reshape the xtrain data into a 2D array
xtrain = xtrain.reshape(-1, 1)

# create the linear regression model using the training data
model = LinearRegression().fit(xtrain, ytrain)

# get the coef_, intercept_ valuesm and r^2 values
# use float() to turn the arrays into a single float value
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(xtrain, ytrain)

# print out the linear equation and r^2 value
print("Model's Linear Equation: y=",coef, "x+", intercept)
print("R Squared value:", r_squared)

'''
**********TEST THE MODEL**********
'''

# reshape the xtest data into a 2D array
xtest = xtest.reshape(-1,1)
# get the predicted y values for the xtest values - returns an array of the results
predict = model.predict(xtest)
# round the value in the np array to 2 decimal places
predict = np.around(predict, 2)

# compare the actual and predicted values
print("\nTesting Linear Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index] # gets the actual y value from the ytest dataset
    predicted_y = predict[index] # gets the predicted y value from the predict variable
    x_coord = xtest[index] # gets the x value from the xtest dataset
    print("x value:", float(x_coord), "Predicted y value:", predicted_y, "Actual y value:", actual)

'''
**********CREATE A VISUAL OF THE RESULTS**********
'''
#sets the size of the graph
plt.figure(figsize=(5,4))

#creates a scatter plot and labels the axes
plt.scatter(xtrain,ytrain, c="purple", label="Training Data")
plt.scatter(xtest, ytest, c="blue", label="Testing Data")

plt.scatter(xtest, predict, c="red", label="Predictions")

plt.xlabel("bmi")
plt.ylabel("Diabetes level")
plt.title("Diabetes level by bmi")
plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

plt.legend()
plt.show()
#avg_percent_error = 0.0
#count=0
#for index in range(len(xtest)):
    #actual = ytest[index]
    #predicted_y = predict[index]
    #x_coord = xtest[index] 
    #count+=1
    #percent_error =abs((predicted_y-actual)/actual)*100
    #avg_percent_error=
    #print("x value:", float(x_coord), "Predicted y value:", predicted_y, "Actual y value:", actual)
