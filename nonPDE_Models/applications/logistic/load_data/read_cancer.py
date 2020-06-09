import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ## testing data
df = pd.read_csv("arcene_test.data", sep=',', header=None)
# print("df = ", df[0])

data = []
for j in range(100):
    data.append([float(i) for i in df[0][j].split()])
inputs_testing = np.array(data)
print("inputs_testing shape", inputs_testing.shape)

# ## training data
df = pd.read_csv("arcene_train.data", sep=',', header=None)
# print("df = ", df[0])

data = []
for j in range(100):
    data.append([float(i) for i in df[0][j].split()])
inputs_training = np.array(data)
print("inputs_training shape", inputs_training.shape)

df = pd.read_csv("arcene_train.labels", sep=',', header=None)
# print("df = ", df[0])
outputs_training = np.array(df[0][:])

# print("outputs_training = ", outputs_training)

# ## validation data
df = pd.read_csv("arcene_valid.data", sep=',', header=None)
# print("df = ", df[0])

data = []
for j in range(100):
    data.append([float(i) for i in df[0][j].split()])
inputs_validation = np.array(data)
print("inputs_validation shape", inputs_validation.shape)

df = pd.read_csv("arcene_valid.labels", sep=',', header=None)
# print("df = ", df[0])
outputs_validation = np.array(df[0][:])

# print("outputs_validation", outputs_validation)

inputs = np.append(inputs_training, inputs_validation, axis=0)
outputs = np.append(outputs_training, outputs_validation)

print("inputs shape = ", inputs.shape, "outputs shape = ", outputs.shape)

np.savez("cancer", inputs=inputs, outputs=outputs)
# data = pickle.load(open("data/yacht_hydrodynamics.data", 'rb'))

# data = np.load("housing.npz")
# print(data["inputs"], data["outputs"])

# plt.figure()
# plt.plot(outputs, '.')
# plt.show()
