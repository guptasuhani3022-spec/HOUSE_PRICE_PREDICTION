import pandas as pd 
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Loan Approval App",page_icon="🏦") 

#for page title
st.title("🏦 Loan Approval Prediction System")

#load dataset
df=pd.read_csv("loan.csv")

#create two variable(x,y) for input and output
x= df[["Income","CIBIL_Score","Loan_Amount","Employment_Years"]]
y= df["Loan_Status"]

#data devide
x_train, x_test, y_train,  y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
model = DecisionTreeClassifier(max_depth=4,min_samples_split=8,min_samples_leaf=4,random_state=42)
model.fit(x_train,y_train)
prediction = model.predict(x_test)
accuracy = accuracy_score(y_test,prediction)

st.success(f"Model Accuracy:{accuracy*100:.2f} %")
Income = st.sidebar.number_input("💸Income:",min_value=10000,max_value=100000,value=40000,step=1000)
Cibil = st.sidebar.number_input("💰CIBIL Score:",min_value=300,max_value=900,value=700,step=50)
Loan = st.sidebar.number_input("💶Loan Ammount:",min_value=50000,max_value=1000000,value=200000,step=5000)
Experience = st.sidebar.number_input("Employment Years:",min_value=0,max_value=40,value=5,step=5)

if st.button("Predict Loan"):
    result = model.predict([[Income,Cibil,Loan,Experience*12000]])

    if result[0]==1:
        approved=min(Loan,int(Income*8+(Cibil-650)*400+Experience*12000))
        if approved<0:
            approved=0

            st.success("👏 Loan Approved")
            st.info("Approved Loan Ammount:${approved:,}")
        else:
            st.error("❌ Loan Rejected")
            



