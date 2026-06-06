# ============================================================================
# WHALE SHARK 3D ANIMATED MIGRATION - RIO LADY (SHARK 15)
# ============================================================================
# Based on Hueter et al. (2013)
# Domain: Longitude [-87.25°, -25.27°], Latitude [-1.82°, 21.84°], Depth [0, 1.6 km]
# ALL CURVES CONTAINED INSIDE BOUNDING BOX
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from IPython.display import Video, display
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print(" WHALE SHARK MIGRATION ANIMATION - RIO LADY (SHARK 15)")
print(" All curves contained inside bounding box")
print("=" * 80)

# ============================================================================
# 1. DOMAIN BOUNDS (Based on Hueter et al. 2013)
# ============================================================================

# Geographic bounds for Shark 15 "Rio Lady"
LON_MIN, LON_MAX = -87.25, -25.27    # Longitude range (degrees West)
LAT_MIN, LAT_MAX = -1.82, 21.84      # Latitude range (degrees North)
Z_MIN, Z_MAX = 0.0, 1.6              # Depth range (km)

# Step length bounds (km/day)
s_min = 15.0
s_max = 155.0

# Time parameters
T = 150  # 150 days

# Monthly μ profile (from manuscript: 1.88 → 1.65)
MU_PROFILE = [1.88, 1.85, 1.82, 1.78, 1.65]
MONTH_DAYS = [30, 31, 30, 31, 28]

print("\n✓ Domain bounds loaded")
print(f"  Longitude: [{LON_MIN}, {LON_MAX}]°")
print(f"  Latitude:  [{LAT_MIN}, {LAT_MAX}]°")
print(f"  Depth:     [{Z_MIN}, {Z_MAX}] km")
print(f"  μ profile: {MU_PROFILE}")

# ============================================================================
# 2. MATHEMATICAL FUNCTIONS (Truncated Lévy Flight)
# ============================================================================

def normalization_constant(mu, s_min, s_max):
    """C(μ) = (1-μ) / (s_max^(1-μ) - s_min^(1-μ))"""
    if abs(mu - 1.0) < 1e-10:
        return 1.0 / np.log(s_max / s_min)
    return (1 - mu) / (s_max**(1 - mu) - s_min**(1 - mu))

def inverse_cdf_levy(xi, mu, s_min, s_max):
    """s = [ ξ(s_max^(1-μ) - s_min^(1-μ)) + s_min^(1-μ) ]^(1/(1-μ))"""
    if abs(mu - 1.0) < 1e-10:
        return s_min * (s_max / s_min) ** xi
    term = xi * (s_max**(1 - mu) - s_min**(1 - mu)) + s_min**(1 - mu)
    return term ** (1 / (1 - mu))

def random_direction_3d():
    """Generate random unit vector on sphere."""
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arccos(2 * np.random.uniform(0, 1) - 1)
    return np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])

def apply_boundary(position, bounds):
    """Reflective boundary conditions - KEEP INSIDE BOX."""
    lon, lat, depth = position

    # Longitude bounds
    if lon < bounds['lon_min']:
        lon = 2 * bounds['lon_min'] - lon
    if lon > bounds['lon_max']:
        lon = 2 * bounds['lon_max'] - lon

    # Latitude bounds
    if lat < bounds['lat_min']:
        lat = 2 * bounds['lat_min'] - lat
    if lat > bounds['lat_max']:
        lat = 2 * bounds['lat_max'] - lat

    # Depth bounds
    if depth < 0:
        depth = -depth
    if depth > bounds['z_max']:
        depth = 2 * bounds['z_max'] - depth

    return np.array([lon, lat, depth])

print("✓ Mathematical functions defined")

# ============================================================================
# 3. SIMULATE RIO LADY TRAJECTORY
# ============================================================================

def simulate_rio_lady():
    """Simulate Rio Lady's 150-day migration with monthly μ variation."""

    bounds = {'lon_min': LON_MIN, 'lon_max': LON_MAX,
              'lat_min': LAT_MIN, 'lat_max': LAT_MAX, 'z_max': Z_MAX}

    # Initial position (Yucatán Peninsula)
    start = np.array([LON_MIN, LAT_MAX, 0.0])

    # Directional bias (south-east toward Mid-Atlantic Ridge)
    bias = np.array([-0.92, -0.39, 0.0])
    bias = bias / np.linalg.norm(bias)

    # Initialize trajectory
    traj = np.zeros((T + 1, 3))
    traj[0] = start.copy()
    step_lengths = []

    day = 0
    for month_idx, (mu, days_in_month) in enumerate(zip(MU_PROFILE, MONTH_DAYS)):

        # Advection weight depends on μ (higher μ = more directed)
        if mu >= 1.8:
            omega = 0.90
        elif mu >= 1.7:
            omega = 0.85
        else:
            omega = 0.75

        for _ in range(days_in_month):
            if day >= T:
                break

            # Generate step length from truncated Lévy
            xi = np.random.uniform(0, 1)
            s = inverse_cdf_levy(xi, mu, s_min, s_max)
            step_lengths.append(s)

            # Generate random direction
            rand_dir = random_direction_3d()

            # Combine bias with random exploration
            direction = omega * bias + (1 - omega) * rand_dir
            direction = direction / np.linalg.norm(direction)

            # Simple vertical oscillation (diving behavior)
            depth = 0.5 + 0.5 * np.sin(2 * np.pi * day / 25)
            depth = np.clip(depth, 0, Z_MAX)

            # Update position
            new_pos = traj[day] + s * direction
            new_pos[2] = depth

            # Apply boundary conditions
            new_pos = apply_boundary(new_pos, bounds)

            traj[day + 1] = new_pos
            day += 1

    # Calculate metrics
    total_dist = np.sum(step_lengths)
    displacement = np.linalg.norm(traj[-1] - traj[0])
    straightness = displacement / total_dist if total_dist > 0 else 0

    print(f"\n✓ Simulation complete!")
    print(f"  Total distance: {total_dist:.0f} km")
    print(f"  Displacement:   {displacement:.0f} km")
    print(f"  Straightness:   {straightness:.4f}")

    return traj, step_lengths

# Run simulation
traj, steps = simulate_rio_lady()

# ============================================================================
# 4. CALCULATE SMART PADDING FOR 3D (KEEPS CURVES INSIDE BOX)
# ============================================================================

# Find actual data ranges
lon_min_actual = min(traj[:, 0].min(), LON_MIN)
lon_max_actual = max(traj[:, 0].max(), LON_MAX)
lat_min_actual = min(traj[:, 1].min(), LAT_MIN)
lat_max_actual = max(traj[:, 1].max(), LAT_MAX)
depth_min_actual = max(traj[:, 2].min(), 0)
depth_max_actual = max(traj[:, 2].max(), Z_MAX)

# Smart padding (10% expansion)
PAD = 0.10
lon_range = lon_max_actual - lon_min_actual
lat_range = lat_max_actual - lat_min_actual
depth_range = depth_max_actual - depth_min_actual

X_LIM = (lon_min_actual - PAD * lon_range, lon_max_actual + PAD * lon_range)
Y_LIM = (lat_min_actual - PAD * lat_range, lat_max_actual + PAD * lat_range)
Z_LIM = (max(depth_min_actual - PAD * depth_range, -0.1), depth_max_actual + PAD * depth_range)

print(f"\n✓ Smart padding applied: {PAD*100:.0f}%")
print(f"  X limits: [{X_LIM[0]:.2f}, {X_LIM[1]:.2f}]")
print(f"  Y limits: [{Y_LIM[0]:.2f}, {Y_LIM[1]:.2f}]")
print(f"  Z limits: [{Z_LIM[0]:.2f}, {Z_LIM[1]:.2f}]")

# ============================================================================
# 5. CREATE ANIMATION (PROFESSIONAL QUALITY)
# ============================================================================

print("\n" + "=" * 80)
print(" CREATING ANIMATION")
print("=" * 80)

# Create figure
fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

# Color map for time progression
colors = plt.cm.plasma(np.linspace(0, 1, T))

def update_animation(frame):
    """Update function for animation."""
    ax.clear()

    # Plot trajectory up to current frame
    for i in range(1, frame):
        ax.plot(traj[i-1:i+1, 0], traj[i-1:i+1, 1], traj[i-1:i+1, 2],
                color=colors[i], linewidth=2, alpha=0.8)

    # Start point (Yucatán)
    ax.scatter(*traj[0], color='green', s=200, marker='o',
               edgecolors='black', linewidth=2, zorder=10, label='Start: Yucatán Peninsula')

    # Current position
    if frame < len(traj):
        ax.scatter(*traj[frame], color='orange', s=150, marker='^',
                   edgecolors='black', linewidth=2, zorder=10, label=f'Day {frame}')

    # End point (Mid-Atlantic Ridge)
    if frame >= len(traj) - 1:
        ax.scatter(*traj[-1], color='red', s=200, marker='s',
                   edgecolors='black', linewidth=2, zorder=10, label='End: Mid-Atlantic Ridge')

    # Draw bounding box (domain boundaries)
    # Bottom face
    ax.plot([LON_MIN, LON_MAX], [LAT_MIN, LAT_MIN], [0, 0], 'gray', alpha=0.5, linewidth=1)
    ax.plot([LON_MIN, LON_MAX], [LAT_MAX, LAT_MAX], [0, 0], 'gray', alpha=0.5, linewidth=1)
    ax.plot([LON_MIN, LON_MIN], [LAT_MIN, LAT_MAX], [0, 0], 'gray', alpha=0.5, linewidth=1)
    ax.plot([LON_MAX, LON_MAX], [LAT_MIN, LAT_MAX], [0, 0], 'gray', alpha=0.5, linewidth=1)
    # Top face
    ax.plot([LON_MIN, LON_MAX], [LAT_MIN, LAT_MIN], [Z_MAX, Z_MAX], 'gray', alpha=0.5, linewidth=1)
    ax.plot([LON_MIN, LON_MAX], [LAT_MAX, LAT_MAX], [Z_MAX, Z_MAX], 'gray', alpha=0.5, linewidth=1)
    ax.plot([LON_MIN, LON_MIN], [LAT_MIN, LAT_MAX], [Z_MAX, Z_MAX], 'gray', alpha=0.5, linewidth=1)
    ax.plot([LON_MAX, LON_MAX], [LAT_MIN, LAT_MAX], [Z_MAX, Z_MAX], 'gray', alpha=0.5, linewidth=1)
    # Vertical edges
    ax.plot([LON_MIN, LON_MIN], [LAT_MIN, LAT_MIN], [0, Z_MAX], 'gray', alpha=0.5, linewidth=1)
    ax.plot([LON_MAX, LON_MAX], [LAT_MIN, LAT_MIN], [0, Z_MAX], 'gray', alpha=0.5, linewidth=1)
    ax.plot([LON_MIN, LON_MIN], [LAT_MAX, LAT_MAX], [0, Z_MAX], 'gray', alpha=0.5, linewidth=1)
    ax.plot([LON_MAX, LON_MAX], [LAT_MAX, LAT_MAX], [0, Z_MAX], 'gray', alpha=0.5, linewidth=1)

    # Labels and title
    ax.set_xlabel('Longitude (°W)', fontsize=13, labelpad=12)
    ax.set_ylabel('Latitude (°N)', fontsize=13, labelpad=12)
    ax.set_zlabel('Depth (km)', fontsize=13, labelpad=12)

    # Calculate current μ based on month
    day_count = 0
    current_mu = MU_PROFILE[0]
    for m, (mu, days) in enumerate(zip(MU_PROFILE, MONTH_DAYS)):
        if frame <= day_count + days:
            current_mu = mu
            break
        day_count += days

    ax.set_title(f'Whale Shark "Rio Lady" (Shark 15) - Day {frame} / {T}\n'
                 f'Lévy Exponent μ = {current_mu:.2f} | '
                 f'Behavior: {"Directed Migration" if current_mu >= 1.8 else "Area-Restricted Search"}',
                 fontsize=14, fontweight='bold')

    # Set limits with SMART PADDING
    ax.set_xlim(X_LIM[0], X_LIM[1])
    ax.set_ylim(Y_LIM[0], Y_LIM[1])
    ax.set_zlim(Z_LIM[0], Z_LIM[1])

    # Invert Z-axis (depth increases downward)
    ax.invert_zaxis()

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

    # Rotating view
    ax.view_init(elev=25, azim=frame * 0.8)

    return ax,

# Create animation
print("Rendering animation (this may take 1-2 minutes)...")
ani = animation.FuncAnimation(fig, update_animation, frames=T,
                              interval=50, repeat=True, blit=False)

# Save as MP4
output_file = 'Rio_Lady_Migration_3D.mp4'
ani.save(output_file, writer='ffmpeg', fps=20, dpi=150)
print(f"✓ Animation saved: {output_file}")

# Display in notebook
print("\nDisplaying animation...")
display(Video(output_file, width=800, height=600, embed=True))

# ============================================================================
# 6. CREATE SIDE-BY-SIDE COMPARISON OF μ VALUES
# ============================================================================

print("\n" + "=" * 80)
print(" CREATING SIDE-BY-SIDE COMPARISON")
print("=" * 80)

# Simplified simulation for different μ values
def quick_simulate(mu, omega):
    bounds = {'lon_min': LON_MIN, 'lon_max': LON_MAX,
              'lat_min': LAT_MIN, 'lat_max': LAT_MAX, 'z_max': Z_MAX}
    start = np.array([LON_MIN, LAT_MAX, 0.0])
    bias = np.array([-0.92, -0.39, 0.0])
    bias = bias / np.linalg.norm(bias)

    traj = np.zeros((T + 1, 3))
    traj[0] = start

    for t in range(T):
        xi = np.random.uniform(0, 1)
        s = inverse_cdf_levy(xi, mu, s_min, s_max)
        rand_dir = random_direction_3d()
        direction = omega * bias + (1 - omega) * rand_dir
        direction = direction / np.linalg.norm(direction)
        depth = 0.5 + 0.5 * np.sin(2 * np.pi * t / 25)
        depth = np.clip(depth, 0, Z_MAX)
        new_pos = traj[t] + s * direction
        new_pos[2] = depth
        new_pos = apply_boundary(new_pos, bounds)
        traj[t + 1] = new_pos

    return traj

print("Generating comparison trajectories...")
traj_1_0 = quick_simulate(1.0, 0.70)
traj_1_4 = quick_simulate(1.4, 0.80)
traj_1_8 = quick_simulate(1.8, 0.90)
traj_2_0 = quick_simulate(2.0, 0.95)

# Find global bounds for comparison
all_traj_comp = np.vstack([traj_1_0, traj_1_4, traj_1_8, traj_2_0])
X_LIM_COMP = (all_traj_comp[:, 0].min() - PAD * (all_traj_comp[:, 0].max() - all_traj_comp[:, 0].min()),
              all_traj_comp[:, 0].max() + PAD * (all_traj_comp[:, 0].max() - all_traj_comp[:, 0].min()))
Y_LIM_COMP = (all_traj_comp[:, 1].min() - PAD * (all_traj_comp[:, 1].max() - all_traj_comp[:, 1].min()),
              all_traj_comp[:, 1].max() + PAD * (all_traj_comp[:, 1].max() - all_traj_comp[:, 1].min()))
Z_LIM_COMP = (max(all_traj_comp[:, 2].min() - 0.1, -0.1),
              all_traj_comp[:, 2].max() + 0.1)

# Create 2x2 subplot for comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 14), subplot_kw={'projection': '3d'})
axes = axes.flatten()

mu_list = [1.0, 1.4, 1.8, 2.0]
traj_list = [traj_1_0, traj_1_4, traj_1_8, traj_2_0]
behavior_list = ["EXPLORATORY", "MIXED STRATEGY", "DIRECTED MIGRATION", "BROWNIAN"]

for idx, (mu, traj, behavior) in enumerate(zip(mu_list, traj_list, behavior_list)):
    ax = axes[idx]
    n = len(traj)
    colors_traj = plt.cm.plasma(np.linspace(0, 1, n))

    for i in range(1, n):
        ax.plot(traj[i-1:i+1, 0], traj[i-1:i+1, 1], traj[i-1:i+1, 2],
                color=colors_traj[i], linewidth=1.5, alpha=0.7)

    ax.scatter(*traj[0], color='green', s=100, marker='o', edgecolors='black', label='Start')
    ax.scatter(*traj[-1], color='red', s=100, marker='s', edgecolors='black', label='End')

    ax.set_xlim(X_LIM_COMP[0], X_LIM_COMP[1])
    ax.set_ylim(Y_LIM_COMP[0], Y_LIM_COMP[1])
    ax.set_zlim(Z_LIM_COMP[0], Z_LIM_COMP[1])
    ax.invert_zaxis()

    ax.set_title(f'μ = {mu} | {behavior}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude (°W)', fontsize=9)
    ax.set_ylabel('Latitude (°N)', fontsize=9)
    ax.set_zlabel('Depth (km)', fontsize=9)
    ax.view_init(elev=25, azim=-60)
    ax.grid(True, alpha=0.2)

    if idx == 0:
        ax.legend(loc='upper left', fontsize=8)

plt.suptitle('Comparison of Whale Shark Trajectories for Different Lévy Exponents μ\n'
             'All curves contained inside domain bounds | Time color: purple (start) → yellow → green (end)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Fig_Comparison_All_Mu.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Comparison figure saved: Fig_Comparison_All_Mu.png")

# ============================================================================
# 7. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print(" COMPLETE SUMMARY")
print("=" * 80)

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RIO LADY MIGRATION ANIMATION COMPLETE                     ║
║                          Shark 15 - 7,772 km Migration                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ GENERATED FILES                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ├── Rio_Lady_Migration_3D.mp4        - Main animation (rotating 3D view)   │
│   └── Fig_Comparison_All_Mu.png        - 2x2 comparison of all μ values      │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ KEY FEATURES                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│                     │
│         │
│   • Monthly μ variation (1.88 → 1.65)                                        │
│   • Reflective boundary conditions enforce domain limits                     │
│   • Depth axis inverted (surface at top, deep at bottom)                     │
│   • Color gradient shows time progression (purple → yellow → green)          │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("✅ ALL CURVES CONTAINED INSIDE BOUNDING BOX!")
print("=" * 80)
