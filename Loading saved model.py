import pickle
filename = "filepath/gp_model(2).pkl"
est_gp = pickle.load(open(filename, 'rb'))
print(est_gp._program.raw_fitness_)