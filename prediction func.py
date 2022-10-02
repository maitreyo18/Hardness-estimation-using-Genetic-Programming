#how the function should be used for input and output
#You need to input all the input variables as an array mantioned as #x_array
def hardness_predict(x_array):
    #x_array = ['C', 'Mn', 'N', 'A_temp', 'A_Time', 'Prestr', 'PreStr_Temp', 'Ag_Temp', 'Ag_Time']
    import pickle
    shape = x_array.shape
    print("Shape of input array", shape)
    if shape[0] == 9:
        x_array = x_array.reshape(1, -1)
    filename = "filepath/gp_model(2).pkl"
    est_gp = pickle.load(open(filename, 'rb'))
    pred_hard = est_gp.predict(x_array)
    return pred_hard