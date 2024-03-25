import matplotlib.pyplot as plt
import numpy as np

# ckavalue = [0.767, 0.792, 0.75, 0.427, 1]
# arousalf1 = [0.57, 0.57, 0.55, 0.46, 0.57]
# apneaf1 = [0.5, 0.5, 0.54, 0.47, 0.49]
# crossentropyLoss = [1.159, 1.133, 1.061, 1.333, 1.182]
# model = ["Unet+DPRNN1D", "TimesNet(avg period)+DPRNN1D", "TimesNet+DPRNN1D", "TimesNet(avg period)+Conv1x1", "DPRNN1D"]
def linear_regression(x, y):
  x = np.array(x)
  y = np.array(y)
  N = len(x)
  sumx = sum(x)
  sumy = sum(y)
  sumx2 = sum(x ** 2)
  sumxy = sum(x * y)
  A = np.mat([[N, sumx], [sumx, sumx2]])
  b = np.array([sumy, sumxy])
  return np.linalg.solve(A, b)


ckavalue = [0.767, 0.792, 0.75, 0.427, 0.619]
arousalf1 = [0.57, 0.57, 0.55, 0.46, 0.56]
apneaf1 = [0.5, 0.5, 0.54, 0.47, 0.51]
crossentropyLoss = [1.159, 1.133, 1.061, 1.333, 1.205]


model = ["Unet+DPRNN1D", "TimesNet(avg period)+DPRNN1D", "TimesNet+DPRNN1D", "TimesNet(avg period)+Conv1x1", "Unet+DPRNN2D"]


plt.figure(1, figsize=(8, 6))
for i in range(0, len(ckavalue)):
    plt.scatter(ckavalue[i], arousalf1[i], label=model[i])

""" fit line """
a0, a1 = linear_regression(ckavalue, arousalf1)
_X2 = np.array(range(0, 10))*0.1
_Y2 = [a0 + a1 * x for x in _X2]
plt.plot( _X2, _Y2)

plt.xlabel("CKA Similarity")
plt.ylabel("Arousal F1 Score")
plt.legend()


plt.figure(2, figsize=(8, 6))
for i in range(0, len(ckavalue)):
    plt.scatter(ckavalue[i], apneaf1[i], label=model[i])

""" fit line """
a0, a1 = linear_regression(ckavalue, apneaf1)
_X2 = np.array(range(0, 10))*0.1
_Y2 = [a0 + a1 * x for x in _X2]
plt.plot( _X2, _Y2)

plt.xlabel("CKA Similarity")
plt.ylabel("Apnea F1 Score")
plt.legend()

plt.figure(3, figsize=(8, 6))
for i in range(0, len(ckavalue)):
    plt.scatter(ckavalue[i], crossentropyLoss[i], label=model[i])

""" fit line """
a0, a1 = linear_regression(ckavalue, crossentropyLoss)
_X2 = np.array(range(0, 10))*0.1
_Y2 = [a0 + a1 * x for x in _X2]
plt.plot( _X2, _Y2)

plt.xlabel("CKA Similarity")
plt.ylabel("Crossentropy Loss")
plt.legend()

plt.show()