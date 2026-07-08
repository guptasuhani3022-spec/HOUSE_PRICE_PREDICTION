import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 

# Create Dataset
data = {
    "Area_sqft": [800, 1000, 1200, 1500, 1800, 2000, 2200,
                  2500, 2800, 3000],
    "Bedrooms" : [1, 2, 2, 3, 3, 4, 4, 4, 5, 5],
    "Price": [2000000, 2800000, 3500000, 4500000, 5500000,
              6500000,7000000, 8000000, 9000000, 
              10000000]
              }
 
#Create DataFrame
df = pd.DataFrame(data)

#Display Dataset
print("House price Dataset")
print(df)

#Features and Target
x= df[["Area_sqft", "Bedrooms"]]
y= df["Price"]

#Split the Dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

#Create linear Regression Model
model = LinearRegression()

#Train the Model
model.fit(x_train, y_train)

#Predict Test Data
y_pred = model.predict(x_test)

#Calculate Accuracy
score = r2_score(y_test, y_pred)
print("R2 Score:", score)

#User Input
area = float(input("Enter Area (sqft):"))
bedrooms = int(input("Enter Number of Bedrooms:"))

#Predict House Price
prediction = model.predict([[area,bedrooms]])

print("\nPredicted House Price:$", prediction[0 ])  





#Scatter plot
plt.scatter(df["Area_sqft"],
df["Price"], color="blue")

#Regression Line
plt.plot(df["Area_sqft"],
         model.predict(df[["Area_sqft","Bedrooms"]]),color="red")

plt.title("House Price Prediction")
plt.xlabel("Area (sqft)")
plt.ylabel("Price")

#Save Graph
plt.savefig("images/house_price_graph.png")

#Show Graph
plt.show()