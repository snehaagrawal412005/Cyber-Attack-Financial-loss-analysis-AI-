#!/usr/bin/env python
# coding: utf-8

# In[1389]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[1391]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import PolynomialFeatures


# In[1393]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


# In[1395]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[1397]:


import warnings
warnings.filterwarnings('ignore')


# In[1399]:


df = pd.read_csv("Global_Cybersecurity_Threats_2015-2024.csv")


# In[1401]:


df.head()


# In[1403]:


df.info()


# In[1405]:


df.describe()


# In[1407]:


missing_values = df.isnull().sum()
print(missing_values)


# In[1409]:


q_high = df["Financial Loss (in Million $)"].quantile(0.95)
df = df[df["Financial Loss (in Million $)"] < q_high]


# In[1411]:


if 'Country' in df.columns:
    df = df.drop(columns=['Country'])


# In[1413]:


df['Time_per_User'] = df['Incident Resolution Time (in Hours)'] * df['Number of Affected Users']


# In[1415]:


df['Financial Loss (in Million $)'] = np.log1p(df['Financial Loss (in Million $)'])


# In[1417]:


df = pd.get_dummies(df, columns=['Attack Type', 'Target Industry'], drop_first=True)
le = LabelEncoder()
df['Attack Source'] = le.fit_transform(df['Attack Source'])
df['Security Vulnerability Type'] = le.fit_transform(df['Security Vulnerability Type'])
df['Defense Mechanism Used'] = le.fit_transform(df['Defense Mechanism Used'])


# In[1419]:


plt.figure(figsize=(8, 4))
sns.histplot(df['Financial Loss (in Million $)'], kde=True, color='blue')
plt.title('Distribution of Financial Loss')
plt.show()


# In[1421]:


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[1422]:


plt.figure(figsize=(8, 4))
sns.scatterplot(x='Incident Resolution Time (in Hours)', y='Financial Loss (in Million $)', data=df)
plt.title('Incident Duration vs. Financial Loss')
plt.xlabel('Duration (Hours)')
plt.ylabel('Loss (USD)')
plt.show()


# In[1423]:


plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Financial Loss (in Million $)'])
plt.title('Boxplot of Financial Loss (Outlier Detection)')
plt.show()


# In[1425]:


numeric_cols = ['Incident Resolution Time (in Hours)', 'Number of Affected Users', 'Time_per_User']


# In[1429]:


poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)


# In[1431]:


scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# In[1433]:


X = df.drop(columns=['Financial Loss (in Million $)'])
y = df['Financial Loss (in Million $)']


# In[1435]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[1437]:


models = {
    "Linear Regression": LinearRegression(),
    
    "Random Forest": RandomForestRegressor(
        n_estimators=500,      
        max_depth=25,           
        min_samples_split=2,   
        min_samples_leaf=1,  
        bootstrap=True,
        n_jobs=-1,              
        random_state=42
    ),
    
    "SVR": SVR(
        kernel='rbf', 
        C=100,                 
        gamma='scale',       
        epsilon=0.01           
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42
    )
}


# In[1439]:


results = {'Model': [], 'MAE': [], 'MSE': [], 'R2 Score': []}


# In[1441]:


print("Model Evaluation Results")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results['Model'].append(name)
    results['MAE'].append(mae)
    results['MSE'].append(mse)
    results['R2 Score'].append(r2)
    
    print(f"{name} = R2: {r2:.4f}, MAE: {mae:.2f}")


# In[1442]:


rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)


# In[1443]:


plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
plt.title('Which Factors Drive Financial Loss? (Feature Importance)')
plt.show()


# In[1444]:


print("Detailed Feature Importance")
print(feature_importance_df)


# In[1445]:


y_pred_rf = rf_model.predict(X_test)
residuals = y_test - y_pred_rf

plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color='purple')
plt.axvline(0, color='red', linestyle='--')
plt.title('Residual Analysis (Prediction Errors)')
plt.xlabel('Error Amount (Predicted - Actual)')
plt.show()


# In[1446]:


results_df = pd.DataFrame(results)


# In[1447]:


plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='R2 Score', data=results_df, palette='viridis')
plt.title('Model Accuracy Comparison (R2 Score)')
plt.axhline(0, color='black', lw=1)
plt.show()

print("\nFinal Results Table:")
print(results_df)


# In[ ]:





# In[ ]:





# In[ ]:




