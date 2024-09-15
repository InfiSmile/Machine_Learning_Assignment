import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import LinearRegression
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings and set plotting
warnings.filterwarnings('ignore')
%matplotlib inline

# Load the dataset
df = pd.read_csv('train.csv')
df.drop(columns=['ID'],axis=0,inplace=True)
# Visualizations (Boxplots)
# Determine the number of columns in the dataframe
num_columns = len(df.columns)

# Create subplots dynamically based on the number of columns
fig, ax = plt.subplots(ncols=7, nrows=(num_columns // 7 + 1), figsize=(20, 10))
ax = ax.flatten()

# Plot boxplots for each column
for index, (col, value) in enumerate(df.items()):
    sns.boxplot(y=col, data=df, ax=ax[index])

plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

# Create distribution plots
fig, ax = plt.subplots(ncols=7, nrows=(num_columns // 7 + 1), figsize=(20, 10))
ax = ax.flatten()

for index, (col, value) in enumerate(df.items()):
    sns.distplot(value, ax=ax[index])

plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

# Normalization for specific columns
cols = ['crim', 'zn', 'tax', 'black']
for col in cols:
    minimum = min(df[col])
    maximum = max(df[col])
    df[col] = (df[col] - minimum) / (maximum - minimum)

# Visualizations after normalization (Distribution plots)
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.distplot(value, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# Correlation heatmap
corr = df.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Regression plots
sns.regplot(y=df['medv'], x=df['lstat'])
