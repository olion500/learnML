import pandas as pd
from sklearn.tree import DecisionTreeRegressor

iowa_file_path = 'data/melb_data.csv'

data = pd.read_csv(iowa_file_path)

Y = data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[features]

melbourne_model = DecisionTreeRegressor(random_state=1)

melbourne_model.fit(X, Y)

print("Making predictions for the following 5 houses")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))