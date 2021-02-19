import shap
import numpy as np
import pandas as pd
from sklearn.utils import resample
from deepforest import CascadeForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import IsolationForest

# load JS visualization code to notebook
shap.initjs()

# # loading the data
# covid_df   = pd.read_csv("/Users/maryamaljame/ML+covid-19/final_model/reported_results/LSTM_data.csv")
# covid_cols = ['Hematocrit', 'Hemoglobin',
# 'Platelets', 'Red blood Cells', 'Lymphocytes', 'Leukocytes',
# 'Basophils', 'Eosinophils', 'Monocytes', 'Serum Glucose', 'Neutrophils',
# 'Urea', 'Proteina C reativa mg/dL', 'Creatinine', 'Potassium', 'Sodium',
# 'Alanine transaminase', 'Aspartate transaminase', 'Label']
#
# covid_df = covid_df[covid_cols]
#
# imputer   = KNNImputer(n_neighbors=11)
# Ximputer  = imputer.fit_transform(covid_df)
# dataframe = pd.DataFrame(Ximputer, columns=covid_cols)
#
# outlier_detect = IsolationForest(n_estimators=150, contamination=float(0.03), max_features=covid_df.shape[1])
#
# outlier_detect.fit(dataframe)
# outliers_predicted = outlier_detect.predict(dataframe)
#
# covid_check = dataframe[outlier_detect.predict(dataframe) == -1]
#
# print("covid_check outlier", len(covid_check))
#
# dataframe = dataframe[outlier_detect.predict(dataframe) != -1]
#
# values = dataframe.values
#
# n_size = int(len(dataframe) * 0.80)
# # prepare train and test sets
# data_sample = resample(values, n_samples=n_size)
#
# dataframe = pd.DataFrame(data_sample, columns=covid_cols)
#
# y = dataframe.Label.to_numpy() # Target variable
# # X_Features = dataframe.drop(['Label'], axis = 1)
# X = dataframe.drop(['Label'], axis = 1).to_numpy() # Features
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# sm = SVMSMOTE(k_neighbors=11)
# X_train, y_train = sm.fit_resample(X_train, y_train)
#
# model = CascadeForestClassifier(backend="sklearn")
#
# model.fit(X_train, y_train)
#
# forest = model.get_forest(0, 0, "rf")
# explainer = shap.TreeExplainer(forest)
# shap_values = np.array(explainer.shap_values(X_train))
# # shap.summary_plot(shap_values[0],X_train)
# # shap.summary_plot(shap_values[1], X_test)
#
# shap.summary_plot(shap_values[1], feature_names=X_train.columns)

# feature_names = feature_names

# feature_names


X, y = shap.datasets.iris()

# print("X", X.head())
# print("y", y)

feature_names = X.columns
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = CascadeForestClassifier(backend="sklearn")
model.fit(X_train, y_train)
forest = model.get_forest(0, 0, "rf")
explainer = shap.TreeExplainer(forest)
shap_values = explainer.shap_values(X_test)

# shap_values = np.array(explainer.shap_values(X_train))
# print(shap_values.shape)
# print(shap_values[2].shape)

# shap.summary_plot(shap_values[2], X_train)
#
#
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X_train, y_train)
# explainer = shap.TreeExplainer(clf)
# shap_values = np.array(explainer.shap_values(X_test))
# shap.summary_plot(shap_values[0],X_train)
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)

# # shap.summary_plot(shap_values, X_test)
# # shap.summary_plot(shap_values, X) #, plot_type="dots"
# shap.summary_plot(shap_values, X, plot_type='dots')
# # shap.summary_plot(shap_values[0], X, plot_type='layered_violin')
#
# # shap.dependence_plot("RM", shap_values, X)

# shap.summary_plot(shap_values, X)
