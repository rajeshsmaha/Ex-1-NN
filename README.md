<H3>NAME: Rajeshwaran.D</H3>
<H3>REGISTER NO.: 212223040165</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 22/08/2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("Churn_Modelling.csv")
df

df.isnull().sum()

df.fillna(0)
df.isnull().sum()

df.duplicated()

df['EstimatedSalary'].describe()

scaler = StandardScaler()
inc_cols = ['CreditScore', 'Tenure', 'Balance', 'EstimatedSalary']
scaled_values = scaler.fit_transform(df[inc_cols])
df[inc_cols] = pd.DataFrame(scaled_values, columns = inc_cols, index = df.index)
df

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("X Values")
x

print("Y Values")
y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print("X Training data")
x_train

print("X Testing data")
x_test
```
## OUTPUT:
### Read the dataset from drive
<img width="1040" height="332" alt="image" src="https://github.com/user-attachments/assets/413bc536-d937-4b4d-99ea-519faabff7cc" />


### Finding Missing Values
<img width="148" height="413" alt="image" src="https://github.com/user-attachments/assets/6d927448-730b-49ec-bd6b-97e0a92f80a9" />


### Handling Missing values
<img width="131" height="423" alt="image" src="https://github.com/user-attachments/assets/3c2139d5-6311-4f51-b0da-b9adf8c72891" />


### Check for Duplicates
<img width="175" height="365" alt="image" src="https://github.com/user-attachments/assets/76acd51e-2c32-4371-916a-c8a9e7cdab3e" />

### Detect Outliers
<img width="192" height="254" alt="image" src="https://github.com/user-attachments/assets/96cea22c-39ba-47c7-807c-f3312eb784f8" />

### Normalize the dataset
<img width="1039" height="367" alt="image" src="https://github.com/user-attachments/assets/ee596ab3-a714-444d-8762-3b8f83fbfb5a" />


### Split the dataset into input and output
<img width="998" height="347" alt="image" src="https://github.com/user-attachments/assets/45a5a4d7-910f-40ec-af2c-026a3d38ccf6" />
<img width="275" height="379" alt="image" src="https://github.com/user-attachments/assets/45e62a38-aa8b-4ea2-9984-bb19228a58ae" />


### Print the training data and testing data
<img width="1003" height="351" alt="image" src="https://github.com/user-attachments/assets/57f0e36e-42de-41c6-90af-bd1433e27424" />
<img width="967" height="322" alt="image" src="https://github.com/user-attachments/assets/d4b0c9c8-ecb9-4430-8f66-f2378f15c912" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


