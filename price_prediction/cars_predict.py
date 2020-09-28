import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pickle


# %matplotlib inline
df=pd.read_csv('car_data.csv')
# print(df.head())
# print(df.shape) # print(df['Seller_Type'].unique()) # print(df['Transmission'].unique()) # print(df['Fuel_Type'].unique())
# print(df['Owner'].unique())

# check missing or null values
# print(df.isnull().sum())#to view null value is thaire or not
# print(df.describe())
# print(df.columns)
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
# print(final_dataset.head())
final_dataset['Current_Year'] = 2020
# print(final_dataset.head())
final_dataset['no_year'] = final_dataset['Current_Year']-final_dataset['Year']
# print(final_dataset.head())
final_dataset.drop(['Year'],axis=1,inplace=True)#inplace will delete the data permenently like permenent operation
final_dataset.drop(['Current_Year'],axis=1,inplace=True)
# print(final_dataset.head())

final_dataset = pd.get_dummies(final_dataset,drop_first=True)
# print(final_dataset.head())
# print(final_dataset.corr())
# print(sns.pairplot(final_dataset))
corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# print(g)
# plt.show()
# print(final_dataset.head())
# independent and dependent features
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]
print(X.head())
print(y.head())

model = ExtraTreesRegressor()
print(model.fit(X,y))
print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
# plt.show()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
# print(X_train.shape)

rf_random = RandomForestRegressor( )

#Hyperparameters
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
print(n_estimators)

#Randomized Search CV

#Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5,30, num=6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split=[2,5,10,15,100]
# Minimum number of samples required at each leaf node
min_samples_leaf=[1,2,5,10]

random_grid = {'n_estimators':n_estimators,
				'max_features':max_features,
				'max_depth':max_depth,
				'min_samples_split':min_samples_split,
				'min_samples_leaf':min_samples_leaf
				}
print(random_grid)

# Use the random grid to search for best heperparameters
# First create the base model to tune
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs=1)
rf_random.fit(X_train,y_train)
predictions=rf_random.predict(X_test)
print(predictions)
sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)
plt.show()

#open a file, where you want to store the data
file = open('random_forest_regression_model.pkl', 'wb')
pickle.dump(rf_random,file)















