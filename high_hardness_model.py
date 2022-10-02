import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from gplearn.genetic import SymbolicRegressor
from gplearn import functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

#Data processing
total_file = pd.read_excel("C:/Data/Studies/Projects/Amlan_sir/High_hardness.xlsx")

X = np.array(total_file.iloc[:, :-1])
y = np.array(total_file.iloc[:, -1])
y = np.reshape(y, (-1,1))

scalar_1 = MinMaxScaler(feature_range=(0.0001, 1))
scalar_2 = MinMaxScaler(feature_range=(0.0001, 1))

scalar_1.fit(X)
X = scalar_1.transform(X)

scalar_2.fit(y)
y = scalar_2.transform(y)

#Test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
y_train = y_train.reshape(-1,)

#Creating and including new functions
def _protectdiv(a,b):
    if (np.logical_and(a==0, b==0).any()):
        return 1
    elif (np.logical_and(a!=0, b==0).any()):
        return np.power(10, 20)
    else:
        return np.divide(a, b)

def _inverse(x):
    return _protectdiv(1, x)

protectdiv = functions.make_function(function=_protectdiv, name="protectdiv", arity=2)
inverse = functions.make_function(function=_inverse, name="inverse", arity=1)


#Genetic Programming
est_gp = SymbolicRegressor(population_size=5000, generations=200, tournament_size=40, const_range=(-1, 1), stopping_criteria=0.001,
                           init_depth=(2, 6), metric='rmse', function_set=['add', 'sub', 'mul', protectdiv, inverse],
                           p_crossover=0.7, p_subtree_mutation=0.05, p_hoist_mutation=0.05, parsimony_coefficient=0.00004,
                           p_point_mutation=0.06, p_point_replace=0.05, max_samples=0.9, n_jobs=6,
                           feature_names=['C', 'Mn', 'N', 'A_temp', 'A_Time', 'Prestr', 'PreStr_Temp', 'Ag_Temp', 'Ag_Time'], verbose=1)
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

#Additional 100 generations

if est_gp._program.raw_fitness_ > 0.09:
    print("\nAdditional 100 generations using warm start\n")
    est_gp.set_params(generations=300, warm_start=True)
    est_gp.fit(X_train, y_train)
    print("\nBest fit function is : ")
    print(est_gp._program)

    print("\nThe RMSE fitness for the training set is: ", est_gp._program.raw_fitness_)

    # Studying the test set
    y_pred = est_gp.predict(X_test)
    test_fitness = mean_squared_error(y_test, y_pred)
    print("The RMSE fitness for the test set is: ", np.sqrt(test_fitness))

    score_2 = est_gp.score(X_train, y_train)
    print("\nThe r2_score for the training set is : ", score_2)

    score_1 = est_gp.score(X_test, y_test)
    print("The r2_score for the test set is : ", score_1)
    
    if est_gp._program.raw_fitness_ > 0.08:
        print("\nAdditional 100 generations using warm start\n")
        est_gp.set_params(generations=400, warm_start=True)
        est_gp.fit(X_train, y_train)
        print("\nBest fit function is : ")
        print(est_gp._program)

        print("\nThe RMSE fitness for the training set is: ", est_gp._program.raw_fitness_)

        # Studying the test set
        y_pred = est_gp.predict(X_test)
        test_fitness = mean_squared_error(y_test, y_pred)
        print("The RMSE fitness for the test set is: ", np.sqrt(test_fitness))

        score_2 = est_gp.score(X_train, y_train)
        print("\nThe r2_score for the training set is : ", score_2)

        score_1 = est_gp.score(X_test, y_test)
        print("The r2_score for the test set is : ", score_1)



#Exporting the function
filename = "C:/Data/Studies/Projects/Amlan_sir/gp_model_high_hardness_400.pkl"
pickle.dump(est_gp, open(filename, 'wb'))