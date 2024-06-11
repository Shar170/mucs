import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the corrected function D(t) as described
def D_corrected(t, alpha):
    return (1 - alpha) * np.exp(-t/3600) + alpha * (1 + np.log(1 + t/(3600)))

# Generate values for t
t = np.linspace(0, 1.5*60 * 60, 400)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(t, D_corrected(t, 0), color='orange')

# Adding labels and title
ax.set_xlabel('t')
ax.set_ylabel('D(t)')
ax.set_title('Graph of the corrected D(t) with varying α')
ax.set_ylim(0, 3)  # Adjust the y-axis limit to accommodate all values
ax.grid(True)

# Animation function
def animate(frame):
    alpha = (np.sin(frame / 20) + 1) / 2  # Vary alpha cyclically from 0 to 1
    y = D_corrected(t, alpha)
    line.set_ydata(y)
    ax.legend([f'α={alpha}'])
    return line,

# Create the animation
ani = FuncAnimation(fig, animate, frames=400, interval=50, blit=True)

# Display the animation
plt.show()
