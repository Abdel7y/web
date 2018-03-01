import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv(r"C:\Users\EL MAGIC\Desktop\abalonedata.csv")
#columns=pd.read_csv(r"C:\Users\EL MAGIC\Desktop\abalonedomain.csv")
columns=["sex : M","length : continuous","diameter:continuous","height:continuous","whole weight:continuous","shucked weight:continuous","viscera weight:continuous","shell weight:continuous","rings:continuous"]
columns = list(df.columns)
print(df)
for column in columns :
     if df[column].dtype == "object":
         df[column].fillna(df[column].mode()[0], inplace = True )
         le = LabelEncoder()
         le.fit(df[column])
         df[column] = le.transform(df[column])      
     else : 
         df[column].fillna(df[column].mean(), inplace = True)
x=df.iloc[:,0:8]
y=df.iloc[:,8]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=42)
linearReg=GradientBoostingRegressor()
linearReg.fit(x_train,y_train)
res=linearReg.predict(x_test)
print(np.sqrt(mean_squared_error(y_test, res)))