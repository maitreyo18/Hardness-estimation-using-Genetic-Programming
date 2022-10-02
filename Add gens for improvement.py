import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from gplearn.genetic import SymbolicRegressor
from gplearn import functions
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

est_gp = pickle.load(open("D:\Projects\Project-Dr. Amlan Dutta\gp_model(1).pkl", 'rb'))
#Data processing
total_file = pd.read_excel("D:\Projects\Project-Dr. Amlan Dutta\Training-8th_in Xls form.xls")
total_file = total_file.drop(['S', 'Si', 'P', 'Al', 'Nb', 'Ti', 'Reference'], axis=1)

X = np.array(total_file.iloc[:, :-1])
y = np.array(total_file.iloc[:, -1])
y = y.reshape(-1, 1)

scalar_1 = MinMaxScaler(feature_range=(0.0001, 1))
scalar_2 = MinMaxScaler(feature_range=(0.0001, 1))

scalar_1.fit(X)
X = scalar_1.transform(X)

scalar_2.fit(y)
y = scalar_2.transform(y)

#Test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_train = y_train.reshape(-1,)

est_gp.set_params(generations=400, warm_start=True)
est_gp.fit(X_train, y_train)
print("\nBest fit function is : ")
print(est_gp._program)

print("\nThe RMSE fitness for the training set is: ", est_gp._program.raw_fitness_)

#Studying the test set
y_pred = est_gp.predict(X_test)
test_fitness = mean_squared_error(y_test, y_pred)
print("The RMSE fitness for the test set is: ", np.sqrt(test_fitness))

score_2 = est_gp.score(X_train, y_train)
print("\nThe r2_score for the training set is : ", score_2)

score_1 = est_gp.score(X_test, y_test)
print("The r2_score for the test set is : ", score_1)

# Visualizing the program tree
dot_data = est_gp._program.export_graphviz()
graph = graphviz.Source(dot_data)
graph.view()



#Exporting data
total_output = est_gp.predict(X)
total_output = pd.DataFrame(total_output)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
export_data = pd.concat([X, y, total_output], axis=1)
export_data.to_excel("D:\Projects\Project-Dr. Amlan Dutta\Project data(2).xls")

#Exporting the function
filename = "D:\Projects\Project-Dr. Amlan Dutta\gp_model(2).pkl"
pickle.dump(est_gp, open(filename, 'wb'))
