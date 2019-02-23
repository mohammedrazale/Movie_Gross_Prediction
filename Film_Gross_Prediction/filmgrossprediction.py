
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

movies = pd.read_csv("movie_metadata.csv")

movies.head()


# In[2]:

movies.info()


# In[3]:

correlat_s = movies.corr()
correlat_s.sort_values("gross",ascending=False,inplace=True)
correlat_s["gross"]


# In[4]:

movies = movies[movies["gross"].notnull() & movies["num_voted_users"].notnull() 
                & movies["num_user_for_reviews"].notnull() & movies["num_critic_for_reviews"].notnull() 
                & movies["imdb_score"].notnull() & movies["budget"].notnull()]
movies.drop(["color","director_name","duration","director_facebook_likes","actor_3_facebook_likes","actor_2_facebook_likes","duration","cast_total_facebook_likes","actor_1_facebook_likes","director_facebook_likes","aspect_ratio","title_year","facenumber_in_poster"],axis=1,inplace=True)
movies.drop(["actor_2_name","genres","actor_1_name","movie_title","actor_3_name","plot_keywords","movie_imdb_link","language","country","content_rating","movie_facebook_likes"],axis=1,inplace=True)
movies.info()


# In[5]:

movies.columns.tolist()


# In[8]:

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

train_x = movies.columns.tolist()
train_x.remove('gross')
X = movies[train_x]
y = movies['gross']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, random_state=0)
"""
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
score = regr.score(X_test, y_test)
print(score)
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print(y_test.shape)
# Plot outputs
plt.scatter(X_test.iloc[:,0], y_test,  color='black')
plt.plot(X_test.iloc[:,0], y_pred, color='blue', linewidth=1)


plt.show()


# In[7]:

from sklearn.linear_model import Ridge

train_x = movies.columns.tolist()
train_x.remove('gross')
X = movies[train_x]
y = movies['gross']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, random_state=0)
ridge = Ridge(alpha=0.1).fit(X_train,y_train)
print(ridge.score(X_train,y_train))
print(ridge.score(X_test,y_test))


# In[8]:

from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train,y_train)
print(lasso.score(X_train,y_train))
print(lasso.score(X_test,y_test))


# In[9]:

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor().fit(X_train,y_train)
pred_tree = tree.predict(X_test)
score = tree.score(X_test, y_test)
print(score)
print('Variance score: %.2f' % r2_score(y_test, pred_tree))

"""
# In[11]:

n_folds = 3

from sklearn.model_selection import KFold
kf = KFold(n_splits=n_folds, random_state=1)
kf = kf.get_n_splits(X_train)


# In[38]:
#Random Forest
print ('Training Random Forest...')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


rf = RandomForestRegressor(n_estimators=1000,max_depth=10) 
rf = rf.fit( X_train, y_train )
rf_score = rf.score(X_test, y_test)
print ('The regressor accuracy score is {:.2f}'.format(rf_score))

score = cross_val_score(rf, X_test, y_test, cv=kf)
print ('The {}-fold cross-validation accuracy score for this classifier is {:.2f}'.format(n_folds, score.mean()))



# In[14]:

from pprint import pprint
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[15]:

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth}

pprint(random_grid)


# In[16]:

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)


# In[17]:

rf_random.best_params_


# In[46]:

from sklearn.metrics import mean_squared_error
from math import sqrt

base_model = RandomForestRegressor(n_estimators = 1000, max_depth = 10)
base_model.fit(X_train, y_train)
print('Base Model')
predictions = base_model.predict(X_test)
b_mse = mean_squared_error(y_test, predictions)
b_rmse = sqrt(b_mse)
b_score = base_model.score(X_test, y_test)
print('Model Performance')
print('Score = {:0.2f}'.format(b_score))
print('Mean Squared Error = {:0.2f}'.format(b_mse))
print('Root Mean Squared Error = {:0.2f}'.format(b_rmse))
plt.figure()
plt.scatter(X_test.iloc[:,0], y_test, c="orange", label="Actual")
plt.scatter(X_test.iloc[:,0], predictions, color="blue", label="Predictions")
plt.xlabel("data")
plt.ylabel("target")
plt.title("Random Forest Regression")
plt.legend()
plt.show()
plt.savefig('before.png')


best_random = rf_random.best_estimator_
print("After Randomized Search")
predictions = best_random.predict(X_test)
a_mse = mean_squared_error(y_test, predictions)
a_rmse = sqrt(a_mse)
a_score = best_random.score(X_test, y_test)
print('Model Performance')
print('Score = {:0.2f}'.format(a_score))
print('Mean Squared Error = {:0.2f}'.format(a_mse))
print('Root Mean Squared Error = {:0.2f}'.format(a_rmse))
plt.figure()
plt.scatter(X_test.iloc[:,0], y_test, c="orange", label="Actual")
plt.scatter(X_test.iloc[:,0], predictions, color="blue", label="Predictions")
plt.xlabel("data")
plt.ylabel("target")
plt.title("Random Forest Regression")
plt.legend()
plt.show()
plt.savefig('after.png')


print('Improvement of {:0.2f}%.'.format( 100 * (a_score - b_score) / b_score))
score_series = [b_score, a_score]
mse_series = [b_mse, a_mse]
rmse_series = [b_mse, a_mse]


# In[47]:

Result_df = pd.DataFrame(index=['Base Model','After RandomSearch'])


# In[49]:

Result_df["Score"] = score_series
Result_df["Mean Squared Error"] = mse_series
Result_df["Root Mean Squared Error"] = rmse_series


# In[50]:

Result_df


# In[52]:

Result_df.to_excel("Results.xls")





