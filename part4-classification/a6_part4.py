import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("part4-classification/iris_data.csv")
data['Species'].replace(['setosa', 'versicolor','virginica'], [0, 1, 2], inplace=True)
x = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm","PetalWidthCm"]].values
y = data["Species"].values

# Step 1: Print the values for x and y
print("Values of x:")
print(x)  
print("\nValues of y:")
print(y)  
# Step 2: Standardize the data using StandardScaler, 
scaler = StandardScaler()
# Step 3: Transform the data
x_scaled = scaler.fit_transform(x)
# Step 4: Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# Step 5: Fit the data
# Step 6: Create a LogsiticRegression object and fit the data
logistic_model = linear_model.LogisticRegression()
logistic_model.fit(x_train, y_train)
# Step 7: Print the score to see the accuracy of the model
accuracy = logistic_model.score(x_test, y_test)
print("\nAccuracy of the model:", accuracy)
# Step 8: Print out the actual ytest values and predicted y values
# based on the xtest data
y_pred = logistic_model.predict(x_test)

print("\nActual ytest values:")
print(y_test)  

print("\nPredicted y values:")
print(y_pred)