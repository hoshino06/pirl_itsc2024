import os
import numpy as np
import matplotlib.pyplot as plt

# data 

# figure
data = np.load('safe_prob.npz')    

    
x_list = data['beta']
y_list = data['yawr']
z      = data['prob']

x, y = np.meshgrid(x_list, y_list)
    
contour = plt.contourf(x, y, z, cmap='viridis')

"""
    # Creating the contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x, y, z, cmap='viridis')
    plt.colorbar(contour)  # Adding color bar to show z values
    
    # Adding labels and title
    plt.xlabel(key[0])
    plt.ylabel(key[1])
    plt.title('Safety Probability')
    
    # Displaying the plot
    plt.show()
"""