import numpy as np
import matplotlib.pyplot as plt
import time
import playsound

start_time = time.time()
mu_0 = 4 * np.pi * 1e-7


# Example parameters (adjust as needed)
n = 8  # number of wires
I = 10.0  # current in each wire, amps
L = 0.6096  # wire length along z
R = 0.08  # radius at which wires are placed
theta_w = np.linspace(0, 2 * np.pi, n, endpoint=False)

# Define your field grid
x_vals = np.linspace(-0.1, 0.1, 200, dtype=np.float32)
y_vals = np.linspace(-0.1, 0.1, 200, dtype=np.float32)
z_vals = np.linspace(0, L, 500, dtype=np.float32)
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

# Initialize total field arrays
Bx_total = np.zeros_like(X, dtype=np.float32)
By_total = np.zeros_like(X, dtype=np.float32)
Bz_total = np.zeros_like(X, dtype=np.float32)
print("Total field arrays initialized")
# Wire discretization settings
N_seg = 500  # number of segments per wire
batch_size = 50  # process segments in batches to save memory
print("Wire loop parameterizations begins now")
for j in range(n):
    # Parameterize the j-th wire
    x_wire_pos = R * np.cos(theta_w[j])
    y_wire_pos = R * np.sin(theta_w[j])
    z_start = 0.0
    z_end = L

    z_wire = np.linspace(z_start, z_end, N_seg + 1, dtype=np.float32)
    x_wire = np.full_like(z_wire, x_wire_pos, dtype=np.float32)
    y_wire = np.full_like(z_wire, y_wire_pos, dtype=np.float32)

    # Compute segment midpoints
    x_mid = 0.5 * (x_wire[:-1] + x_wire[1:])
    y_mid = 0.5 * (y_wire[:-1] + y_wire[1:])
    z_mid = 0.5 * (z_wire[:-1] + z_wire[1:])

    # dl components for each segment
    dx = np.diff(x_wire)
    dy = np.diff(y_wire)
    dz = np.diff(z_wire)
    dl = np.stack((dx, dy, dz), axis=-1)  # shape: (N_seg, 3)

    dl_x = dl[:, 0]
    dl_y = dl[:, 1]
    dl_z = dl[:, 2]
print("Main for loop begins now")
# Process wire segments in batches to save memory
for seg_start in range(0, N_seg, batch_size):
    seg_end = min(seg_start + batch_size, N_seg)

    # Extract batch of segments
    dl_x_batch = dl_x[seg_start:seg_end]
    dl_y_batch = dl_y[seg_start:seg_end]
    dl_z_batch = dl_z[seg_start:seg_end]

    x_mid_batch = x_mid[seg_start:seg_end]
    y_mid_batch = y_mid[seg_start:seg_end]
    z_mid_batch = z_mid[seg_start:seg_end]

    # Compute R = r - r' for this batch of segments
    # Dimensions: (Nx, Ny, Nz, batch_size)
    R_x = X[..., np.newaxis] - x_mid_batch
    R_y = Y[..., np.newaxis] - y_mid_batch
    R_z = Z[..., np.newaxis] - z_mid_batch

    R_mag = np.sqrt(R_x**2 + R_y**2 + R_z**2)
    R_mag_cubed = R_mag**3

    # Compute dl x R for this batch
    # dl x R = (dl_y*R_z - dl_z*R_y, dl_z*R_x - dl_x*R_z, dl_x*R_y - dl_y*R_x)
    dBx = (
        dl_y_batch[np.newaxis, np.newaxis, np.newaxis, :] * R_z
        - dl_z_batch[np.newaxis, np.newaxis, np.newaxis, :] * R_y
    ) / R_mag_cubed
    dBy = (
        dl_z_batch[np.newaxis, np.newaxis, np.newaxis, :] * R_x
        - dl_x_batch[np.newaxis, np.newaxis, np.newaxis, :] * R_z
    ) / R_mag_cubed
    dBz = (
        dl_x_batch[np.newaxis, np.newaxis, np.newaxis, :] * R_y
        - dl_y_batch[np.newaxis, np.newaxis, np.newaxis, :] * R_x
    ) / R_mag_cubed

    # Sum contributions from this batch of segments
    Bx_wire_batch = (mu_0 * I / (4 * np.pi)) * np.sum(dBx, axis=-1)
    By_wire_batch = (mu_0 * I / (4 * np.pi)) * np.sum(dBy, axis=-1)
    Bz_wire_batch = (mu_0 * I / (4 * np.pi)) * np.sum(dBz, axis=-1)

    # Accumulate into total field
    Bx_total += Bx_wire_batch
    By_total += By_wire_batch
    Bz_total += Bz_wire_batch

# After processing all wires, we have the total B field:
B_magnitude = np.sqrt(Bx_total**2 + By_total**2 + Bz_total**2)
print("Main for loop complete")
"""
#test data for plotting

x_vals = np.linspace(-0.1, 0.1, 100, dtype=np.float32)
y_vals = np.linspace(-0.1, 0.1, 100, dtype=np.float32)
L = 0.6096
R=.08
z_vals = np.linspace(0, L, 100, dtype=np.float32)
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

Bx_total = -Y
By_total = X
Bz_total = np.zeros_like(X)

B_magnitude = np.sqrt(Bx_total**2 + By_total**2 + Bz_total**2)

"""

# Bx_total, By_total, Bz_total, and B_magnitude are now computed without needing >32GB of RAM.
print(B_magnitude[0, 0, 0])
valid_mask = B_magnitude > 1e-25
Bx_plot = np.zeros_like(Bx_total)  # initialize Bx_plot
By_plot = np.zeros_like(By_total)  # initialize By_plot
Bx_plot[valid_mask] = (
    Bx_total[valid_mask] / B_magnitude[valid_mask]
)  # normalize for plotting and prevent div by 0
By_plot[valid_mask] = (
    By_total[valid_mask] / B_magnitude[valid_mask]
)  # normalize for plotting and prevent div by 0
mask_outside = X**2 + Y**2 > R**2
Bx_plot[mask_outside] = 0
By_plot[mask_outside] = 0
z_index = np.argmin(np.abs(z_vals - L / 2))
plt.figure(figsize=(8, 6))
stride = 3
plt.quiver(
    x_vals[::stride],
    y_vals[::stride],
    Bx_plot[::stride, ::stride, z_index],
    By_plot[::stride, ::stride, z_index],
    scale=200,
    scale_units="xy",
    linewidths=0.5,
)
plt.title(f"Magnetic Field at z = {z_vals[z_index]:.2f} m")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.savefig("B_field.png")
circle = plt.Circle((0, 0), R, color="red", fill=False, linestyle="--", linewidth=1.5)
plt.gca().add_artist(circle)
plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
plt.axline((0, 0), slope=1, color="red", linestyle=":")
plt.axline((0, 0), slope=-1, color="red", linestyle=":")
end_time = time.time()
print(f"Time taken: {(end_time - start_time)/60:.2f} minutes")
# playsound.playsound("C:/Users/enzoa/Music/calm alarm.wav")
plt.show()
