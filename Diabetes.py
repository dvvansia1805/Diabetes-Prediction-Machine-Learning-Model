import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Loading the diabetes data to pandas DataFrame
file_path = "D:/College Work/AI/Diabetes Prediction/diabetes.csv"
diabetes_dataset = pd.read_csv(file_path)

# Printing the first 5 rows of the dataset
print(diabetes_dataset.head())

# Number of rows and columns in the dataset
print("Shape of dataset:", diabetes_dataset.shape)

# Getting the statistical measures of the data
print(diabetes_dataset.describe())

# Distribution of the Outcome
print(diabetes_dataset['Outcome'].value_counts())

# Mean value for all the columns grouped by the Outcome
print(diabetes_dataset.groupby('Outcome').mean())

# Separating data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

# Standardizing the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)

X = standardized_data
print(X)
print(Y)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print("Shapes:", X.shape, X_train.shape, X_test.shape)

# Training the Support Vector Machine classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy score of the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of training data is:', training_data_accuracy)

# Accuracy score of the testing data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of testing data is:', testing_data_accuracy)

# Example input data for prediction
input_data = (3, 169, 74, 19, 125, 29.9, 0.268, 31)

# Changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardizing the input
std_data = scaler.transform(input_data_reshaped)

# Prediction using the trained classifier
prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print('The person is non-diabetic')
else:
    print('The person is diabetic')