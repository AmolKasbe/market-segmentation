#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
# Import necessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[3]:


df=pd.read_csv('mcdonalds.csv')


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[8]:


sns.boxplot(data=df)


# In[9]:


df.info()


# In[10]:


df.shape


# # Data Cleaning

# In[11]:


df.isnull().sum()


# In[12]:


df.duplicated().sum()


# In[13]:


df.drop_duplicates()


# In[14]:


df.describe()


# In[15]:


df['yummy'].value_counts()


# In[18]:


# Handle invalid entries in the 'Like' column
df['Like'] = df['Like'].replace({'I love it!+5': +5, 'I hate it!-5': -5}).astype(int)


# In[20]:


df


# In[ ]:


# Handle invalid entries in the 'Like' column
df['VisitFrequency'] = df['VisitFrequency'].replace({'I love it!+5': +5, 'I hate it!-5': -5}).astype(int)


# In[21]:


df['VisitFrequency'].value_counts()


# In[22]:


df['VisitFrequency'].unique()


#  Check if the string contains 'never' and map it to 0 visits
#  Check if the string contains 'once a year' and map it to 1 visit per year
#  Check if the string contains 'every three months' and map it to 4 visits per year
#  Check if the string contains 'once a month' and map it to 12 visits per year
#  Check if the string contains 'once a week' and map it to 52 visits per year
#  Check if the string contains 'more than once a week' and map it to 104 visits per year
#  If none of the above conditions are met, return None for invalid or unrecognized values

# In[23]:


# Standardize and map `VisitFrequency` values to integers
def clean_visit_frequency(value):
    value = str(value).strip().lower()
    if "never" in value:
        return 0
    elif "once a year" in value:
        return 1
    elif "every three months" in value:
        return 4  # Quarterly visits
    elif "once a month" in value:
        return 12  # Monthly visits
    elif "once a week" in value:
        return 52  # Weekly visits
    elif "more than once a week" in value:
        return 104  # Twice a week
    else:
        return None  # Mark invalid or unknown values as None
    
# Apply the `clean_visit_frequency` function to the 'VisitFrequency' column of the DataFrame
# This transforms the column's values based on the mappings defined above
df['VisitFrequency'] = df['VisitFrequency'].apply(clean_visit_frequency)


# In[24]:


df


# Visualizations:
# Bar Plot: Frequency of visits across all respondents.
# Box Plot: Relationship between age and visit frequency.
# Gender Distribution: Proportion of male vs. female respondents.
# Heatmap: Correlation between numeric variables.
# Stacked Bar Chart: Comparison of preferences (yummy, convenient, etc.) by gender.

# # Data Visualization

# In[25]:


# Plot 1: Bar Plot for `VisitFrequency`
plt.figure(figsize=(8, 6))
sns.countplot(x='VisitFrequency', data=df)
plt.title('Distribution of Visit Frequencies')
plt.xlabel('Visit Frequency (per year)')
plt.ylabel('Count')
plt.show()


# In[26]:


# Plot 2: Box Plot for `Age` vs. `VisitFrequency`
plt.figure(figsize=(8, 6))
sns.boxplot(x='VisitFrequency', y='Age', data=df)
plt.title('Age vs. Visit Frequency')
plt.xlabel('Visit Frequency (per year)')
plt.ylabel('Age')
plt.show()


# In[27]:


# Plot 3: Gender Distribution
plt.figure(figsize=(6, 6))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[28]:


# Plot 4: Heatmap for Feature Correlation
plt.figure(figsize=(10, 8))
corr_matrix = df[['Like', 'Age', 'VisitFrequency']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[29]:


# Plot 5: Stacked Bar Chart for Preferences by Gender
preferences = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']
gender_preferences = df.groupby('Gender')[preferences].apply(lambda x: (x == 'Yes').sum())

gender_preferences.T.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Preferences by Gender')
plt.xlabel('Preferences')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.show()


# In[ ]:





# In[33]:


# Create subplots for all pie charts in one line
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Pie Chart 1: Distribution of `VisitFrequency`
visit_freq_counts = df['VisitFrequency'].value_counts()
axes[0].pie(visit_freq_counts, labels=visit_freq_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
axes[0].set_title('Visit Frequency Distribution')

# Pie Chart 2: Gender Distribution
gender_counts = df['Gender'].value_counts()
axes[1].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("muted"))
axes[1].set_title('Gender Distribution')

# Pie Chart 3: Preferences (e.g., "Yummy")
yummy_counts = df['yummy'].value_counts()
axes[2].pie(yummy_counts, labels=yummy_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("bright"))
axes[2].set_title('Yummy Preference Distribution')

# Adjust the layout to prevent overlap
plt.tight_layout()
plt.show()


# Explanation:
# Subplots (plt.subplots(1, 3)): Creates a single row (1) with three columns (3), allowing the pie charts to be arranged horizontally.
# axes[0], axes[1], and axes[2]: These are the individual axes where each pie chart is drawn.
# plt.tight_layout(): Ensures that the subplots fit within the figure area without overlap

# # Feature Scaling

# In[34]:


# Import necessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[37]:


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# List of categorical columns that need to be label encoded
categorical_columns = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting', 'Gender']

# Apply Label Encoding for each categorical column
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])


# In[40]:


df


# In[42]:


# Standard Scaler for all variables
standard_scaler = StandardScaler()

# scale all variables
df_scaled = standard_scaler.fit_transform(df)

# convert scaled data back to dataframe
#df_scaled = pd.DataFrame(df_scaled, columns=df.columns)


# In[44]:


df_scaled


# In[45]:


df_scaled = pd.DataFrame(df_scaled, columns=df.columns)


# In[46]:


df_scaled


# In[47]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Selecting all numerical features for clustering
# Since your data is already standardized, we can use all columns for clustering
X = df_scaled # Drop 'Gender' as it's categorical



# In[58]:


#Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(df_scaled)
PCA_ds = pd.DataFrame(pca.transform(df_scaled), columns=(["col1","col2", "col3"]))
PCA_ds.describe().T


# In[65]:


#A 3D Projection Of Data In The Reduced Dimension
x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]
#To plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, c="maroon", marker="o" )
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()


# In[61]:


df_scaled


# In[59]:


# Quick examination of elbow method to find numbers of clusters to make.
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()


# In[ ]:




