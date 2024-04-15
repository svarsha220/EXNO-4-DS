# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/863c7d4b-6afe-46ce-8f90-1a335ca2bbbd)

```
data.isnull().sum()
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/29db7509-42c2-49f4-a603-fbc732f2d11a)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/64d94308-321c-4500-bf44-a1449113dda3)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/1a10225b-ad1f-4340-b0f9-ac2c10ec87b8)
```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/1128fc3a-1b04-45e7-9a76-7c6a9402c14c)
```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/ecc8c9eb-2393-4f20-84a6-1fc4432f26e2)
```
data2
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/450087c6-e55c-4e07-be89-c98015f9d699)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/f7b752aa-d11f-4367-a9a1-0a8a3b20281d)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/2bb23f59-f78a-4228-9b42-380de6bc03d4)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/8bdfa286-419d-46ee-8268-fd836dfbfaa1)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/967b3c59-2857-4b58-938b-d37c22ca4137)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/ce17c263-a6f4-4dd9-8f6e-fc42f293b823)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/6fba7d28-0c9e-4e64-ac8a-01a57eada851)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/2e35f25a-0ab1-41d8-a27a-d263acb1112d)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/287907b0-3f66-4984-84c4-30cff447f90b)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/832b431a-457a-41f5-93d6-493b679866c7)
```
data.shape
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/23cba83a-2857-41c2-9af7-fd8125d42f25)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/b1099868-4023-4d86-81c1-642c9292186d)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/a9f06171-688a-48fa-91d2-9089306ec0d3)
```
tips.time.unique()
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/a42adbd0-a4dc-4791-8790-e16e7e59543e)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/dad8a3f5-6390-4599-8579-ddcfa5948096)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/svarsha220/EXNO-4-DS/assets/127709117/62319ba3-6622-4607-87a8-c0405591f98c)
# RESULT:
Thus, Feature selection and Feature scaling has been used and executed in the given dataset.

