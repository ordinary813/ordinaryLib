import numpy as np
import matplotlib.pyplot as plt

arr = [5.51, 20.82, -0.77, 19.3, 14.24, 9.74, 11.59, -6.08]
brr = [5.35, 24.03, -0.57, 19.38, 12.77, 9.68, 12.06, -5.22]
print(np.mean(arr))

print(np.mean(brr))


for i, (x, y) in enumerate(zip(arr, brr)):
    plt.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.scatter(arr, brr)
plt.grid(True)
plt.axhline(y=0, color='black', linewidth=2)
plt.axvline(x=0, color='black', linewidth=2)
plt.show()