## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
      
```python
import pandas as pd
df=pd.read_csv("/content/Encoding Data (1).csv")
df
```
![alt text](image.png)
```python
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold',]
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![alt text](image-2.png)
```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![alt text](image-3.png)

```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![alt text](image-4.png)
```python
dfc['con_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![alt text](image-5.png)
```python
from  sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![alt text](image-6.png)
```python
pd.get_dummies(df2,columns=["nom_0"])
```
![alt text](image-7.png)
```python
pip install --upgrade category_encoders
```
![alt text](image-8.png)
```python
from category_encoders import BinaryEncoder
import pandas as pd
df=pd.read_csv("/content/data.csv")
df
```
![alt text](image-9.png)
```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![alt text](image-10.png)
```python
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc['City'],y=cc['Target'])
cc=pd.concat([cc,new],axis=1)
cc
```
![alt text](image-11.png)
```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![alt text](image-12.png)
```python
df.skew()
```
![alt text](image-13.png)
```python
df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df["Highly Negative Skew"]=np.sqrt(df["Highly Negative Skew"])
df
```
![alt text](image-14.png)
```python
df['Highly Positive Skew_boxcox'], parameters=stats.boxcox(df['Highly Positive Skew'])
df
```
![alt text](image-15.png)
```python
df['Moderate Positive Skew_yeojohnson'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
df
```
![alt text](image-16.png)
```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![alt text](image-17.png)
```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![alt text](image-18.png)
```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df['Moderate Negative Skew']=qt.fit_transform(df[['Moderate Negative Skew']])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![alt text](image-19.png)

# RESULT:
       Thus Feature encodind and transformation process is performed on the given data.

       
