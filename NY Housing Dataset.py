#!/usr/bin/env python
# coding: utf-8

# #  Dataset Description :
# #     This dataset contains prices of New York houses, providing valuable insights into the real estate market in the region. It includes information such as broker titles, house types, prices, number of bedrooms and bathrooms, property square footage, addresses, state, administrative and local areas, street names, and geographical coordinates.
# 

# ## Dataset Key Features: 
# ### 1 BROKERTITLE: Title of the broker
# ### 2 TYPE: Type of the house
# ### 3 PRICE: Price of the house
# ### 4 BEDS: Number of bedrooms
# ### 5 BATH: Number of bathrooms
# ### 6 PROPERTYSQFT: Square footage of the property
# ### 7 ADDRESS: Full address of the house
# ### 8 STATE: State of the house
# ### 9 MAIN_ADDRESS: Main address information
# ### 10 ADMINISTRATIVE_AREA_LEVEL_2: Administrative area level 2 information
# ### 11 LOCALITY: Locality information
# ### 12 SUBLOCALITY: Sublocality information
# ### 13 STREET_NAME: Street name
# ### 14 LONG_NAME: Long name
# ### 15 FORMATTED_ADDRESS: Formatted address
# ### 16 LATITUDE: Latitude coordinate of the house
# ### 17 LONGITUDE: Longitude coordinate of the house

# ### Tasks:
# #### Perform Exploratory Data Analysis (EDA).
# #### Find new features for training.
# #### Build a model, tune parameters for training, and select the best one.
# 

# ## Exploratory Data Analysis

# ### Importing Libraries & Loading Data

# #### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ML Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
# To avoid the warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("NY-House-Dataset.csv")
df


# In[3]:


# Getting shape and size
print(f"This data has {df.shape[0]} rows/enties and {df.shape[1]} columns/features.")


# In[4]:


# This dataset is used to display First few rows
df.head()


# In[5]:


# This dataset is used to display last few rows
df.tail()


# In[6]:


# The info() is used to display the datatype of the dataset
df.info()


# In[7]:


# This function is used to check the null values
df.isnull().sum()


# In[8]:


# To check the duplicates
df.duplicated()


# ## DATA ANAYSIS 

# In[9]:


# Statistical Summary of all columns
df.iloc[:, :-1].describe().T.sort_values(by='std', ascending = False)\
                          .style.background_gradient(cmap="Greens")\
                          .bar(subset=["max"], color='#F8766D')\
                          .bar(subset=["mean"], color='#00BFC4')


# In[10]:


df.describe()


# In[11]:


# Getting and Storing all the Categorical variables
categorical_features = [feature for feature in df.columns if df[feature].dtype=='O']
print(f"Numbers of Categorical Features : {len(categorical_features)}")


# In[12]:


# Getting and Storing all the Numerical variables
numerical_features = [feature for feature in df.columns if df[feature].dtype!='O']
print(f"Numbers of Numerical Features : {len(numerical_features)}")


# In[13]:


#To display stastical values of price column
print("Mean: " + str(df["PRICE"].mean()))
print("Count: " + str(df["PRICE"].count())) 
print("Max: " + str(df["PRICE"].max()))
print("Min: " + str(df["PRICE"].min()))
print("Meadian: " + str(df["PRICE"].median()))
print("Standart: " + str(df["PRICE"].std()))


# ## Data Visualization

# In[14]:


# Categorical Features
categorical_features = [feature for feature in df.columns if df[feature].dtype=='O']
print(f"Numbers of Categorical Features : {len(categorical_features)}")


# In[15]:


# Numerical Features
numerical_features = [feature for feature in df.columns if df[feature].dtype!='O']
print(f"Numbers of Numerical Features : {len(numerical_features)}")


# In[16]:


priceOfBeds = df[['BEDS', 'PRICE']]
priceOfBeds


# In[17]:


bed_prices = priceOfBeds.groupby('BEDS').mean()
bed_prices


# In[18]:


# lets see the numerical and categorical columns:
cat_col = df.select_dtypes(include = object)
num_col = df.select_dtypes([int,float])
display(cat_col.columns,"\n",num_col.columns,"\n\n")


# In[19]:


# we can fetch both categorical and numerical columns by pyhton approach
ca_col = [i for i in df.columns if df[i].dtypes == "O"]

nm_col = [i for i in df.columns if i not in ca_col]
print(ca_col,"\n",nm_col)


# In[20]:


# Call the function and get column lists
print('Continuous numerical features: ', nm_col)
print('No. of Continuous features: ', len(nm_col))
print('\nCategorical or discrete features: ', ca_col)
print('No. of Categorical features: ', len(ca_col))


# In[21]:


type_price_sum = df.groupby('TYPE')['PRICE'].sum().sort_values(ascending=False)
pd.DataFrame(type_price_sum).plot.bar(title='The price sum of each type')


# In[22]:


# Set a threshold for grouping minor categories
threshold = 0.5e9
minor_categories = type_price_sum[type_price_sum < threshold].index


# In[23]:


# Create a new column 'type_grouped' with 'Other' for minor categories
df['type_grouped'] = df['TYPE'].apply(lambda x: x if x not in minor_categories else 'Other')


# In[24]:


# Group by 'type_grouped' and calculate the sum of 'price' for each type
grouped_price_sum = df.groupby('type_grouped')['PRICE'].sum().sort_values(ascending=False)


# In[25]:


# Set the type with the highest sum to explode
explode_values = [0.15 if type == grouped_price_sum.idxmax() else 0 for type in grouped_price_sum.index]


# In[26]:


# Define colors for the pie chart
colors = ['pink' if type == grouped_price_sum.idxmax() else 'skyblue' for type in grouped_price_sum.index]


# In[27]:


# Plot
plt.pie(grouped_price_sum, labels=grouped_price_sum.index, autopct='%1.1f%%', startangle=90, explode=explode_values, colors=colors)


# In[28]:


# Number of BAths
fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.countplot(data=df, x='BATH', log=True)
plt.xticks(rotation=90)
plt.title('Number of Baths')
plt.show()


# In[29]:


df['BATH'] = df['BATH'].apply(np.ceil)
# Convert 'BATH' column to float32 (or float64)
df['BATH'] = df['BATH'].astype('float32')
# Create the count plot
fig, ax = plt.subplots(figsize=(18, 4))
sns.countplot(data=df, x='BATH', log=True)
plt.xticks(rotation=90)
plt.title('Number of Baths')


# In[30]:


fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.countplot(data=df, x='LOCALITY', log=True)
plt.xticks(rotation=90)
plt.title('Localities')
plt.show()


# In[31]:


# Number Of Beds
fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.countplot(data=df, x='BEDS', log=True)
plt.title('Number of Beds')
plt.show()


# In[32]:


sns.barplot(x = "BEDS", y = df.BEDS.index, data= df);


# In[33]:


# Property area in SquareFeet:
# Distribution of various property's areas can be seen in following graph.
fig, ax = plt.subplots(figsize=(25, 10))
fig = sns.histplot(data=df, x='PROPERTYSQFT', bins=50, kde=True)
plt.ylim(0,500)
plt.ticklabel_format(style = 'plain')
fig.set(xlabel='')
plt.suptitle('Distribution of Area')
plt.show()


# In[34]:


fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.histplot(data=df, x='PROPERTYSQFT', bins=50, kde=True)
plt.ylim(0,1000)
plt.ticklabel_format(style = 'plain')
plt.show()


# In[35]:


# Price: Distribution of prices of houses can be seen in following graph.
fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.histplot(data=df, x='PRICE', bins=50, kde=True)
plt.ylim(0,500)
plt.ticklabel_format(style = 'plain')
fig.set(xlabel='')
plt.suptitle('Distribution of Price')
plt.show()


# In[36]:


# Bar Graph related to the beds value count
df["BEDS"].value_counts().plot.barh()


# In[37]:


# Bar Graph related to the type value coun
df["TYPE"].value_counts().plot.barh();


# In[38]:


# Group by 'TYPE' and 'BEDS', count occurrences, and reset index
grouped_df = df.groupby(['TYPE', 'BEDS']).size().reset_index(name='count')
# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='TYPE', y='count', hue='BEDS', data=grouped_df)
plt.title('Count of Beds by Type')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()


# In[39]:


# Plotting
plt.figure(figsize=(10, 6))
# Plot histogram without KDE
sns.distplot(df.BEDS, kde=False, label='Histogram')
# Plot histogram with KDE
sns.distplot(df.BEDS, label='Histogram with KDE')
plt.title('Histogram of BEDS with KDE')
plt.xlabel('BEDS')
plt.ylabel('Density')
plt.legend()
plt.show()


# In[40]:


# Plotting
plt.figure(figsize=(10, 6))
# Plot KDE without shading
sns.kdeplot(df.BEDS, shade=False, label='KDE')
# Plot KDE with shading
sns.kdeplot(df.BEDS, shade=True, label='KDE with Shading')
plt.title('Kernel Density Estimation (KDE) of BEDS')
plt.xlabel('BEDS')
plt.ylabel('Density')
plt.legend()
plt.show()


# In[41]:


# Display Histogram for all columns
df.hist(figsize = (10,10))


# In[42]:


df = df.drop(['SUBLOCALITY'], axis=1)
df.head()


# In[43]:


plt.figure(figsize = (12,8))
sns.boxplot(df)
plt.grid()
plt.show()


# In[44]:


print('Price :')
print(f"Max Price : {df.PRICE.max()}")
print(f"Min Price : {df.PRICE.min()}")
print(f"Price > 2100000000  : {(df.PRICE > 200000).sum()}")
print()
print('Beds :')
print(f"Max Beds : {df.BEDS.max()}")
print(f"Min Beds : {df.BEDS.min()}")
print(f"Beds > 45 : {(df.BEDS > 90).sum()}")


# In[45]:


print('Bath :')
print(f"Max Bath : {df.BATH.max()}")
print(f"Min Bath : {df.BATH.min()}")
print(f"Bath > 2100000000  : {(df.PRICE > 200000).sum()}")
print()
print('PropertySqft :')
print(f"Max PropertySqft : {df.PROPERTYSQFT .max()}")
print(f"Min PropertySqft : {df.PROPERTYSQFT .min()}")
print(f"PropertySqft  > 45 : {(df.PROPERTYSQFT  > 90).sum()}")
print()
print('Longitude :')
print(f"Max Longitude : {df.LONGITUDE .max()}")
print(f"Min Longitude : {df.LONGITUDE .min()}")
print(f"Longitude  > 45 : {(df.LONGITUDE  > 90).sum()}")


# In[46]:


def hist_box_plots(data,col,bins="auto"):
    fig,axis = plt.subplots(ncols=2,figsize=(11,3)) 
    sns.histplot(data=data,x=col,bins=bins,ax=axis[0],kde=True)
    sns.boxplot(data=data,x=col,ax=axis[1])


# In[47]:


for col in num_col.columns:
    hist_box_plots(num_col,col)


# In[48]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='PROPERTYSQFT', y='PRICE', data=df)
plt.title('Price vs. Square Footage')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.show()


# In[49]:


# Draw bar chart
fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.barplot(data=bed_prices, x=bed_prices.index, y=bed_prices['PRICE'], log=True)
# ax.set_yticks(range(0, len(bed_prices['PRICE'])))
plt.xlabel('Number of beds')
plt.ylabel('Average Price')
plt.title('AVERAGE PRICE FOR EACH BEDS')
plt.show()


# In[50]:


# Pairplot for visualizing relationships between numeric features
sns.pairplot(df[nm_col])
plt.suptitle('Pairplot of Numeric Features')
plt.show()


# ## MACHINE LEARNING

# In[51]:


# Heatmap for correlation matrix of numeric features
plt.figure(figsize=(10, 8))
sns.heatmap(df[nm_col].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.show()


# In[52]:


# Replacing Outliers by upper cap values
def handle_outler(df,feature):
    # Calculating IQR
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1
    # Calculating Lower & Upper Bound
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    # If data has Outlier changing it with Upper_Bound
    df[feature] = np.where(df[feature]>upper_bound,upper_bound,df[feature])
    
handle_outler(df,'PRICE')
handle_outler(df,'BATH')
handle_outler(df,'PROPERTYSQFT')
handle_outler(df,'LONGITUDE')
handle_outler(df,'BEDS')


# In[53]:


df.head()


# In[54]:


#Analysing the OUTLIERS
plt.figure(figsize = (12,8))
sns.boxplot(df)
plt.grid()
plt.show()


# In[55]:


X = df[['BEDS','BATH','PROPERTYSQFT']]
y = df['PRICE']


# In[56]:


X.head()


# In[57]:


y.head()


# In[58]:


## Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=42,)


# In[59]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[60]:


# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_pred = linear_reg.predict(X_test)


# In[61]:


# Decision Tree Regression
dt_reg = DecisionTreeRegressor(random_state=10)
dt_reg.fit(X_train, y_train)
dt_pred = dt_reg.predict(X_test)


# In[62]:


# Support Vector Regression
svr_reg = SVR()
svr_reg.fit(X_train, y_train)
svr_pred = svr_reg.predict(X_test)


# In[63]:


# Random Forest Regression
rf_reg = RandomForestRegressor(random_state=10)
rf_reg.fit(X_train, y_train)
rf_pred = rf_reg.predict(X_test)


# In[64]:


# Evaluate models
def evaluate_model(model_name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Model:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    print("")


# In[65]:


# Evaluate Linear Regression Model
evaluate_model("Linear Regression", y_test, linear_pred)


# In[66]:


# Evaluate Decision Tree Regression Model
evaluate_model("Decision Tree Regression", y_test, dt_pred)


# In[67]:


# Evaluate Random Forest Regression Model
evaluate_model("Random Forest Regression", y_test, rf_pred)


# In[68]:


# Evaluate Support Vector Regression Model
evaluate_model("Support Vector Regression", y_test, svr_pred)


# In[69]:


#k-nearest neighbors
knn_model=KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train,y_train)
knn_predictions=knn_model.predict(X_test)
#mean squared error
mse=mean_squared_error(y_test,knn_predictions)
print("means squared error:",mse)


# In[70]:


#Y = β0 + β1X1 + β2X2 + β3X3 + … + βnXn + e
print("intercept i.e b0",linear_reg.intercept_)
print("coefficients i.e b1,b2,b3")
list(zip(X,linear_reg.coef_))


# In[71]:


# Initialize and train the Gaussian Naive Bayes classifier  
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
# Make predictions on the test set
nb_predictions = nb_classifier.predict(X_test)


# In[72]:


# Calculate the accuracy score
accuracy = accuracy_score(y_test, nb_predictions)
print("Naive Bayes Accuracy:", accuracy) 


# In[73]:


# Extract relevant columns
x = df[['BEDS','BATH','PROPERTYSQFT']]
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
# Choose the number of clusters (you can experiment with different values)
num_clusters = 3


# In[74]:


# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("Cluster Centers:")
print(scaler.inverse_transform(kmeans.cluster_centers_))


# In[75]:


mean_aberror = metrics.mean_absolute_error(y_test,linear_pred)
mean_sqerror = metrics.mean_squared_error(y_test,linear_pred)
rmsqurrerror = np.sqrt(metrics.mean_squared_error(y_test,linear_pred))
print(linear_reg.score(x,y)*100)
print(mean_aberror) #0.00000000000004298186828178065
print(mean_sqerror)
print(rmsqurrerror) 


# In[76]:


# Visualize the clusters (you can modify this based on your needs)
plt.scatter(df['BEDS'], df['PROPERTYSQFT'], c=df['Cluster'], cmap='viridis')
plt.xlabel('BEDS')
plt.ylabel('PROPERTYSQFT')
plt.title('K-Means Clustering')
plt.show()


# In[77]:


# Scatter plot for predicted vs actual prices
plt.figure(figsize=(12, 8))
# Linear Regression
plt.scatter(y_test, linear_pred, color='blue', label='Linear Regression')
# Decision Tree Regression
plt.scatter(y_test, dt_pred, color='red', label='Decision Tree Regression')
# Random Forest Regression
plt.scatter(y_test, rf_pred, color='green', label='Random Forest Regression')
# Support Vector Regression
plt.scatter(y_test, svr_pred, color='skyblue', label='Support Vector Regression')
# knn_predictions
plt.scatter(y_test, knn_predictions, color='blue', label='k-nearest neighbors')
# Add diagonal line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.xlabel('Actual Prices (USD)')
plt.ylabel('Predicted Prices (USD)')
plt.title('Actual vs Predicted Prices for NY-House-Dataset')
plt.legend()
plt.show()


# In[78]:


# Sample data
models = ['Linear Regression', 'Decision Tree Regression', 'Random Forest Regression', 'Support Vector Regression','KNeighborsRegressor']
actual_prices = np.random.rand(len(models)) * 100  # Example actual prices
predicted_prices = np.random.rand(len(models)) * 100  # Example predicted prices
plt.figure(figsize=(12, 8))
# Create bar chart
plt.bar(models, actual_prices, color='blue', alpha=0.5, label='Actual Prices')
plt.bar(models, predicted_prices, color='red', alpha=0.5, label='Predicted Prices')
plt.xlabel('Regression Models')
plt.ylabel('Prices (USD)')
plt.title('Actual vs Predicted Prices for NY-House_Dataset')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# In[79]:


# Sample data
models = ['Linear Regression', 'Decision Tree Regression', 'Random Forest Regression', 
          'Support Vector Regression', 'KNeighborsRegressor']
actual_prices = np.random.rand(len(models)) * 100  # Example actual prices
predicted_prices = np.random.rand(len(models)) * 100  # Example predicted prices
# Calculate total actual and predicted prices
total_actual = np.sum(actual_prices)
total_predicted = np.sum(predicted_prices)
# Create pie chart
labels = ['Actual Prices', 'Predicted Prices']
sizes = [total_actual, total_predicted]
colors = ['pink', 'purple']
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Actual vs Predicted Prices for NY-House_Dataset')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()


# In[80]:


# Sample data
models = ['Linear Regression', 'Decision Tree Regression', 'Random Forest Regression', 
          'Support Vector Regression', 'KNeighborsRegressor']
actual_prices = np.random.rand(len(models)) * 100  # Example actual prices
predicted_prices = np.random.rand(len(models)) * 100  # Example predicted prices

# Create pie chart for each model
for i in range(len(models)):
    total_actual = actual_prices[i]
    total_predicted = predicted_prices[i]
    labels = ['Actual Prices', 'Predicted Prices']
    sizes = [total_actual, total_predicted]
    colors = ['pink', 'purple']
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title(f'Actual vs Predicted Prices for {models[i]}')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.show()


# In[81]:


# Scatter plot function
def scatter_plot(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.title(f"Actual vs Predicted Prices - {model_name}")
    plt.xlabel("Actual Prices (USD)")
    plt.ylabel("Predicted Prices (USD)")
    plt.show()


# In[82]:


# Scatter plot for Linear Regression
scatter_plot(y_test, linear_pred, "Linear Regression")


# In[83]:


# Scatter plot for Decision Tree Regression
scatter_plot(y_test, dt_pred, "Decision Tree Regression")


# In[84]:


# Scatter plot for Random Forest Regression
scatter_plot(y_test, rf_pred, "Random Forest Regression")


# In[85]:


# Scatter plot for Support Vector Regression
scatter_plot(y_test, svr_pred, "Support Vector Regression")


# In[86]:


# knn_predictions
scatter_plot(y_test, knn_predictions, 'k-nearest neighbors')


# In[ ]:




