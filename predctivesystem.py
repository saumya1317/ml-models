# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle 
loaded_model = pickle.load(open(r'C:\Users\Saumya\Desktop\mlmodel\heart_disease_model.sav', 'rb'))
input= (51,1,0,140,298,0,1,122,1,4.2,1,3,3)
input_data = np.asarray(input)
reshaped = input_data.reshape(1,-1)
prediction =loaded_model.predict(reshaped)
if(prediction == 1 ):
   print("defective heart")
else :
  print("healthy heart")