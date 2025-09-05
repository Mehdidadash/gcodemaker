import numpy as np
import matplotlib.pyplot as plt

def feed_function(z, feed_max):
    """
    Calculate feed speed based on z values and predefined thresholds and formulas.

    Parameters:
    - z: Array of z values.
    - feed_max: Maximum feed speed.

    Returns:
    - feed_func_values: Array of feed speeds corresponding to z values.
    """
    # Predefined variables
    a = feed_max / 4
    b = 1
    c = 95
    d = feed_max / 4
    e = feed_max / 10

    feed_func_values = np.zeros_like(z)
    for i in range(len(z)):
        i_percentage = (i / len(z)) * 100
        if i_percentage < b:
            # First segment: i_percentage = 0 -> feed = a, i_percentage = b -> feed = feed_max
            feed_func_values[i] = a + (feed_max - a) * (i_percentage / b)
        elif b <= i_percentage <= c:
            # Second segment: i_percentage = b -> feed = feed_max, i_percentage = c -> feed = d
            feed_func_values[i] = feed_max + (d - feed_max) * ((i_percentage - b) / (c - b))
        elif i_percentage > c:
            # Third segment: i_percentage = c -> feed = d, i_percentage = 100 -> feed = e
            feed_func_values[i] = d + (e - d) * ((i_percentage - c) / (100 - c))
    return feed_func_values

def metal_feed_function(z, feed_max):
    """
    Calculate feed speed based on z values using a linear relationship.

    Parameters:
    - z: Array of z values.
    - feed_max: Maximum feed speed.

    Returns:
    - feed_func_values: Array of feed speeds corresponding to z values.
    """
    # Predefined variables
    m = 10  # Denominator for feed_max at i_percentage=0
    a = 5  # i_percentage where feed = feed_max / n
    n = 7  # Denominator for feed_max at i_percentage=a
    b = 12  # i_percentage where feed = feed_max / h
    h = 1  # Denominator for feed_max at i_percentage=b and c
    c = 85  # i_percentage where feed = feed_max / h
    o = 8  # Denominator for feed_max at i_percentage=100

    feed_func_values = np.zeros_like(z)
    for i in range(len(z)):
        i_percentage = (i / len(z)) * 100
        if i_percentage < a:
            # Linear interpolation from i_percentage=0 to i_percentage=a
            feed_func_values[i] = (feed_max / m) + ((feed_max / n) - (feed_max / m)) * (i_percentage / a)
        elif a <= i_percentage < b:
            # Linear interpolation from i_percentage=a to i_percentage=b
            feed_func_values[i] = (feed_max / n) + ((feed_max / h) - (feed_max / n)) * ((i_percentage - a) / (b - a))
        elif b <= i_percentage < c:
            # Constant feed between i_percentage=b and i_percentage=c
            feed_func_values[i] = feed_max / h
        else:
            # Linear interpolation from i_percentage=c to i_percentage=100
            feed_func_values[i] = (feed_max / h) + ((feed_max / o) - (feed_max / h)) * ((i_percentage - c) / (100 - c))
    return feed_func_values

# Parameters
z_min = 0
z_max = 16.2  # Example range for z
num_points = 320
feed_max = 9000

# Generate z values and calculate feed
z = np.linspace(z_min, z_max, num_points)

# Generate feed for metal_feed_function
metal_feed = metal_feed_function(z, feed_max)

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(z, metal_feed, label="Metal Feed Speed", color="red", linestyle="--")
plt.title("Relationship Between z and Metal Feed Speed")
plt.xlabel("z (Position along the axis)")
plt.ylabel("Feed Speed")
plt.grid(True)
plt.legend()
plt.show()