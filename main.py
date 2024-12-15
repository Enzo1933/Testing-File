import numpy as np
import time
import scipy.constants as const

start_time = time.time()
mu_0 = 4 * np.pi * 1e-7

# Example parameters (adjust as needed)
n = 8         # number of wires
I = 10.0       # current in each wire, amps
L = 0.6096     # wire length along z
R = 0.08       # radius at which wires are placed
R_domain = 0.076 # accelerator radius (not used directly here)
theta_w = np.linspace(0, 2*np.pi, n, endpoint=False)
N_seg = 500    # number of segments per wire

# Define your field grid
x_vals = np.linspace(-0.1, 0.1, 200, dtype=np.float32)
y_vals = np.linspace(-0.1, 0.1, 200, dtype=np.float32)
z_vals = np.linspace(0, L, N_seg, dtype=np.float32)

# Create memory-mapped arrays to store the entire 3D result without loading into RAM
Bx_total = np.memmap('Bx_total.dat', dtype='float32', mode='w+', shape=(x_vals.size, y_vals.size, z_vals.size))
By_total = np.memmap('By_total.dat', dtype='float32', mode='w+', shape=(x_vals.size, y_vals.size, z_vals.size))
Bz_total = np.memmap('Bz_total.dat', dtype='float32', mode='w+', shape=(x_vals.size, y_vals.size, z_vals.size))

# Precompute wire segment geometry (to avoid recomputing for each z-plane)
z_wire = np.linspace(0.0, L, N_seg+1, dtype=np.float32)
dx_all = []
dy_all = []
dz_all = []
x_mid_all = []
y_mid_all = []
z_mid_all = []

for j in range(n):
    x_wire_pos = R * np.cos(theta_w[j])
    y_wire_pos = R * np.sin(theta_w[j])
    
    x_wire = np.full_like(z_wire, x_wire_pos, dtype=np.float32)
    y_wire = np.full_like(z_wire, y_wire_pos, dtype=np.float32)
    
    # Midpoints and differences
    x_mid = 0.5 * (x_wire[:-1] + x_wire[1:])
    y_mid = 0.5 * (y_wire[:-1] + y_wire[1:])
    z_mid = 0.5 * (z_wire[:-1] + z_wire[1:])
    
    dx = np.diff(x_wire)
    dy = np.diff(y_wire)
    dz = np.diff(z_wire)
    
    dx_all.append(dx)
    dy_all.append(dy)
    dz_all.append(dz)
    x_mid_all.append(x_mid)
    y_mid_all.append(y_mid)
    z_mid_all.append(z_mid)

# Convert to arrays for convenience
dx_all = np.array(dx_all)
dy_all = np.array(dy_all)
dz_all = np.array(dz_all)
x_mid_all = np.array(x_mid_all)
y_mid_all = np.array(y_mid_all)
z_mid_all = np.array(z_mid_all)

batch_size = 50
print("Initialization done, starting computation...", "Time:", time.time() - start_time)

# Process the field one z-plane at a time
for z_idx, z_plane in enumerate(z_vals):
    # Create X-Y mesh for this particular z-plane
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Initialize fields for this slice
    Bx_slice = np.zeros((x_vals.size, y_vals.size), dtype=np.float32)
    By_slice = np.zeros((x_vals.size, y_vals.size), dtype=np.float32)
    Bz_slice = np.zeros((x_vals.size, y_vals.size), dtype=np.float32)
    
    # Sum contributions from each wire
    for j in range(n):
        dx_wire = dx_all[j]
        dy_wire = dy_all[j]
        dz_wire = dz_all[j]
        x_mid_wire = x_mid_all[j]
        y_mid_wire = y_mid_all[j]
        z_mid_wire = z_mid_all[j]
        
        # Process wire segments in batches
        for seg_start in range(0, N_seg, batch_size):
            seg_end = min(seg_start + batch_size, N_seg)
            
            dl_x_batch = dx_wire[seg_start:seg_end]
            dl_y_batch = dy_wire[seg_start:seg_end]
            dl_z_batch = dz_wire[seg_start:seg_end]
            x_mid_batch = x_mid_wire[seg_start:seg_end]
            y_mid_batch = y_mid_wire[seg_start:seg_end]
            z_mid_batch = z_mid_wire[seg_start:seg_end]
            
            # Compute R = r - r' only for this slice
            # Shape: (X, Y, number_of_segments_in_batch)
            R_x = X[..., np.newaxis] - x_mid_batch
            R_y = Y[..., np.newaxis] - y_mid_batch
            R_z = z_plane - z_mid_batch
            
            R_mag = np.sqrt(R_x**2 + R_y**2 + R_z**2)
            R_mag_cubed = R_mag**3
            
            # Compute dl x R for each segment
            # dl x R = (dl_y*R_z - dl_z*R_y, dl_z*R_x - dl_x*R_z, dl_x*R_y - dl_y*R_x)
            dBx = (dl_y_batch * R_z - dl_z_batch * R_y) / R_mag_cubed
            dBy = (dl_z_batch * R_x - dl_x_batch * R_z) / R_mag_cubed
            dBz = (dl_x_batch * R_y - dl_y_batch * R_x) / R_mag_cubed
            
            # Sum over segments in this batch
            Bx_wire_batch = (mu_0*I/(4*np.pi)) * np.sum(dBx, axis=-1)
            By_wire_batch = (mu_0*I/(4*np.pi)) * np.sum(dBy, axis=-1)
            Bz_wire_batch = (mu_0*I/(4*np.pi)) * np.sum(dBz, axis=-1)
            
            # Accumulate into slice
            Bx_slice += Bx_wire_batch
            By_slice += By_wire_batch
            Bz_slice += Bz_wire_batch
    
    # Write this z-plane's results into the memory-mapped arrays
    Bx_total[:, :, z_idx] = Bx_slice
    By_total[:, :, z_idx] = By_slice
    Bz_total[:, :, z_idx] = Bz_slice

print("Computation complete!", "Total time:", time.time() - start_time)
