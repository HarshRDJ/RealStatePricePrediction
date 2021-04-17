from joblib import load, dump
model = load('RealStatePricePrediction.joblib')

import numpy as np
input = np.array([-0.40024809, -0.48677005, 0.07379607, -0.27288841, 0.11759244, -0.21782702, 0.27998081,
                  -0.70700509, -0.98674509, -0.81320785, 1.17302027, 0.4394630, 1-0.53354544])
print(model.predict([input]))