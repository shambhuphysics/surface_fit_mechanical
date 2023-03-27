import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
data = np.loadtxt('out.dat')
x1 = data[:, 0]
x2 = data[:, 1]
y = data[:, 2]
def func(x, a1, a2, a3):
    return a1*x[:,0]**2 + a2*x[:,1]**2 + a3*x[:,0]*x[:,1]
p0 = [1, 1, 1]
coeffs, _ = curve_fit(func, np.column_stack((x1,x2)), y, p0=p0)
x1_vals = np.linspace(np.min(x1), np.max(x1), 50)
x2_vals = np.linspace(np.min(x2), np.max(x2), 50)
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
y_pred = coeffs[0]*x1_grid**2 + coeffs[1]*x2_grid**2 + coeffs[2]*x1_grid*x2_grid
A0 = float(input("What is the value of Area (A)=? "))
C11 = 2*coeffs[0]/A0*16
C22 = 2*coeffs[1]/A0*16
C12 = coeffs[2]/A0*16
C66 = (C11 - C12)/2
print('')
print('\033[91m' + 'Elastic constants:' + '\033[0m')
print('')
print('C11 = ', C11, 'N/m')
print('')
print('C22 =', C22, 'N/m')
print('')
print('C12 =', C12, 'N/m')
print('')
print('C66 =', C66, 'N/m')
print('')
print('\033[32m' + 'Have a nice day buddy !!' + '\033[0m')
print('')
print('\033[91m' + 'Hey any confusion!! check tutorial on my channel??' + '\033[0m')
print('\033[91m' + 'Please VISIT: https://www.youtube.com/channel/UCv5riEshGsYyonaSr0Ur2lA' + '\033[0m')
print('')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, y_pred, cmap='coolwarm')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()
