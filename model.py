# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('your_data.csv')

# Handling missing values
dataset.fillna(dataset.mean(), inplace=True)

X = dataset.iloc[:, :-1]

y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
linearregressor = LinearRegression()

#Fitting model with training data
linearregressor.fit(X, y)

# Saving model to disk
pickle.dump(linearregressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

# Example usage:
squareMeters = int(input("Enter square meters: "))
numberOfRooms = int(input("Enter number of rooms: "))
hasYard = int(input("Enter 1 if has yard, 0 otherwise: "))
hasPool = int(input("Enter 1 if has pool, 0 otherwise: "))
floors = int(input("Enter number of floors: "))
cityCode = int(input("Enter city code: "))
cityPartRange = int(input("Enter city part range: "))
numPrevOwners = int(input("Enter number of previous owners: "))
made = int(input("Enter year made: "))
isNewBuilt = int(input("Enter 1 if new built, 0 otherwise: "))
hasStormProtector = int(input("Enter 1 if has storm protector, 0 otherwise: "))
basement = int(input("Enter 1 if has basement, 0 otherwise: "))
attic = int(input("Enter 1 if has attic, 0 otherwise: "))
garage = int(input("Enter 1 if has garage, 0 otherwise: "))
hasStorageRoom = int(input("Enter 1 if has storage room, 0 otherwise: "))
hasGuestRoom = int(input("Enter 1 if has guest room, 0 otherwise: "))

user_input = [squareMeters, numberOfRooms, hasYard, hasPool, floors, cityCode, cityPartRange, numPrevOwners, made, isNewBuilt, hasStormProtector, basement, attic, garage, hasStorageRoom, hasGuestRoom]
x_user = np.array(user_input).reshape(1, -1)
print(model.predict(x_user))