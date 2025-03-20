import numpy as np

f_c_1 = 3.5
f_c_2 = 2.8e3

Gain = 10

C_1 = 10e-6

R_1 = 1/(2*np.pi*f_c_1*C_1)

R_2 = R_1 * Gain

C_2 = 1/(2*np.pi*f_c_2*R_2)


print(R_1)
print(R_2)
print(C_1)
print(C_2)