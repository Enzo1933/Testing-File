import numpy as np
import time
import scipy.constants as const
import matplotlib as plt
import playsound

start_time = time.time()
mu_0 = 4 * np.pi * 1e-7

# Example parameters
n = 8        # number of wires
I = 10.0     # current in each wire, amps
L = 0.6096   # wire length along z
R = 0.08     # radius at which wires are placed
R_domain = .076 #Accelerator Radius
N_seg = 500  # number of segments per wire

theta_w = np.linspace(0, 2*np.pi, n, endpoint=False)

# Define your field grid
x_vals = np.linspace(-0.1, 0.1, 200, dtype=np.float32)
y_vals = np.linspace(-0.1, 0.1, 200, dtype=np.float32)
z_vals = np.linspace(0, L, N_seg, dtype=np.float32)

# Create memory-mapped arrays to store the entire 3D field
Bx_total = np.memmap('Bx_total.dat', dtype='float32', mode='w+', shape=(x_vals.size, y_vals.size, z_vals.size))
By_total = np.memmap('By_total.dat', dtype='float32', mode='w+', shape=(x_vals.size, y_vals.size, z_vals.size))
Bz_total = np.memmap('Bz_total.dat', dtype='float32', mode='w+', shape=(x_vals.size, y_vals.size, z_vals.size))

# Precompute wire geometry
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

dx_all = np.array(dx_all)
dy_all = np.array(dy_all)
dz_all = np.array(dz_all)
x_mid_all = np.array(x_mid_all)
y_mid_all = np.array(y_mid_all)
z_mid_all = np.array(z_mid_all)

batch_size = 50

# Define chunk size for z dimension only
chunk_size_z = 200
x_size = x_vals.size
y_size = y_vals.size
z_size = z_vals.size

print("Starting computation by z-chunks.", "Time:", time.time() - start_time)

# Create the full XY grid once (x_size, y_size)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

for z_start in range(0, z_size, chunk_size_z):
    z_end = min(z_start + chunk_size_z, z_size)
    z_chunk = z_vals[z_start:z_end]
    z_chunk_size = z_end - z_start
    
    # Create Z dimension for this chunk
    # Z_chunk_grid: (x_size, y_size, z_chunk_size)
    Z_chunk_grid = np.empty((x_size, y_size, z_chunk_size), dtype=np.float32)
    for i_z, z_val in enumerate(z_chunk):
        Z_chunk_grid[:, :, i_z] = z_val

    # Broadcast X and Y to have the z dimension
    # X_broadcast, Y_broadcast: (x_size, y_size, z_chunk_size)
    X_broadcast = np.broadcast_to(X[:, :, None], (x_size, y_size, z_chunk_size))
    Y_broadcast = np.broadcast_to(Y[:, :, None], (x_size, y_size, z_chunk_size))
    
    # Initialize fields for this chunk
    Bx_chunk = np.zeros((x_size, y_size, z_chunk_size), dtype=np.float32)
    By_chunk = np.zeros((x_size, y_size, z_chunk_size), dtype=np.float32)
    Bz_chunk = np.zeros((x_size, y_size, z_chunk_size), dtype=np.float32)
    
    # Loop over wires
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

            # Shape adjustments for broadcasting:
            # x_mid_batch, y_mid_batch, z_mid_batch: (batch_segments,)
            # Add dimensions to align them to (1,1,1,batch_segments)
            x_mid_batch_4D = x_mid_batch[None, None, None, :]
            y_mid_batch_4D = y_mid_batch[None, None, None, :]
            z_mid_batch_4D = z_mid_batch[None, None, None, :]

            # Now X_broadcast, Y_broadcast, Z_chunk_grid have shape (x_size, y_size, z_chunk_size)
            # Add a new dimension for batch_segments:
            # (x_size, y_size, z_chunk_size, 1) - (1,1,1,batch_segments) -> (x_size,y_size,z_chunk_size,batch_segments)
            R_x = X_broadcast[:, :, :, None] - x_mid_batch_4D
            R_y = Y_broadcast[:, :, :, None] - y_mid_batch_4D
            R_z = Z_chunk_grid[:, :, :, None] - z_mid_batch_4D
            
            R_mag = np.sqrt(R_x**2 + R_y**2 + R_z**2)
            R_mag_cubed = R_mag**3
            
            # dl x R calculations
            # dl_x_batch, dl_y_batch, dl_z_batch: (batch_segments,)
            # Broadcasting them to (1,1,1,batch_segments)
            dl_x_4D = dl_x_batch[None, None, None, :]
            dl_y_4D = dl_y_batch[None, None, None, :]
            dl_z_4D = dl_z_batch[None, None, None, :]

            dBx = (dl_y_4D * R_z - dl_z_4D * R_y) / R_mag_cubed
            dBy = (dl_z_4D * R_x - dl_x_4D * R_z) / R_mag_cubed
            dBz = (dl_x_4D * R_y - dl_y_4D * R_x) / R_mag_cubed
            
            # Sum over segments in this batch
            Bx_wire_batch = (mu_0*I/(4*np.pi)) * np.sum(dBx, axis=-1)
            By_wire_batch = (mu_0*I/(4*np.pi)) * np.sum(dBy, axis=-1)
            Bz_wire_batch = (mu_0*I/(4*np.pi)) * np.sum(dBz, axis=-1)
            
            Bx_chunk += Bx_wire_batch
            By_chunk += By_wire_batch
            Bz_chunk += Bz_wire_batch
    
    # Write this chunk back into the memory-mapped arrays
    Bx_total[:, :, z_start:z_end] = Bx_chunk
    By_total[:, :, z_start:z_end] = By_chunk
    Bz_total[:, :, z_start:z_end] = Bz_chunk
    
print("Computation complete!", "Total time:", (time.time() - start_time)/60, "minutes")
B_magnitude= np.sqrt(Bx_total**2+By_total**2+Bz_total**2)
valid_mask = B_magnitude > 1e-25
Bx_plot=np.zeros_like(Bx_total) #initialize Bx_plot
By_plot=np.zeros_like(By_total) #initialize By_plot
Bx_plot[valid_mask] = Bx_total[valid_mask] / B_magnitude[valid_mask] #normalize for plotting and prevent div by 0
By_plot[valid_mask] = By_total[valid_mask] / B_magnitude[valid_mask] #normalize for plotting and prevent div by 0
mask_outside = (X**2+Y**2 > R_domain**2)
Bx_plot[mask_outside] = 0
By_plot[mask_outside] = 0
z_index = np.argmin(np.abs(z_vals - L/2))
plt.figure(figsize=(8, 6))
stride = 5
plt.quiver(x_vals[::stride], y_vals[::stride], Bx_plot[::stride, ::stride, z_index],
           By_plot[::stride, ::stride, z_index], 
           scale=200, scale_units='xy', linewidths=0.25)
plt.title(f'Magnetic Field at z = {z_vals[z_index]:.2f} m')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.savefig('B_field.png')
circle = plt.Circle((0, 0), R_domain, color='red', fill=False, linestyle='--', linewidth=1.5)
plt.gca().add_artist(circle)
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
plt.axline((0, 0), slope=1, color='red', linestyle=':')
plt.axline((0, 0), slope=-1, color='red', linestyle=':')
end_time = time.time()
print("Simulation Complete",f"Total Time taken: {(end_time - start_time)/60:.2f} minutes")
playsound.playsound("C:/Users/enzoa/Music/calm alarm.wav")
plt.show()

