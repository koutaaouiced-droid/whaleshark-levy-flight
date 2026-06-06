# ============================================================================
# WHALE SHARK 3D LÉVY FLIGHT SIMULATION (ALL CURVES INSIDE BOX)
# ============================================================================
# Based on Hueter et al. (2013) - Shark 15 "Rio Lady"
# FIXED: All 3D curves are forced inside the bounding box using smart padding
# ============================================================================

# %% [markdown]
# # Whale Shark 3D Telemetry Framework
# ## High-Resolution Numerical Modeling

# %% [code]
# --- 1. LIBRARIES & CONFIG ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.4
plt.rcParams['figure.dpi'] = 100

print("=" * 80)
print(" WHALE SHARK LÉVY FLIGHT SIMULATION")
print(" All 3D curves CONTAINED inside bounding box")
print("=" * 80)

# %% [code]
# --- 2. MODEL PARAMETERS ---

LAT_MIN, LAT_MAX = -1.82, 21.84      # Latitude range
LON_MIN, LON_MAX = -87.25, -25.27    # Longitude range
Z_MIN = 0.0                          # Ocean Surface (km)
Z_MAX = 1.6                          # Max depth bound (km)

s_min = 15.0                         # Min flight scale (km/day)
s_max = 155.0                        # Max flight scale (km/day)
T = 150                              # Tracking duration (days)

MU_VALUES = [1.0, 1.4, 1.8, 2.0]
COLORS = {1.0: '#1f77b4', 1.4: '#ff7f0e', 1.8: '#2ca02c', 2.0: '#d62728'}

BEHAVIOR = {
    1.0: "EXPLORATORY (Superdiffusive)",
    1.4: "MIXED STRATEGY",
    1.8: "DIRECTED MIGRATION",
    2.0: "BROWNIAN (Localized)"
}

print("\n✓ Core ecological parameters integrated successfully.")

# %% [code]
# --- 3. DYNAMICS ENGINE ---

def normalization_constant(mu, s_min, s_max):
    if abs(mu - 1.0) < 1e-10:
        return 1.0 / np.log(s_max / s_min)
    return (1 - mu) / (s_max**(1 - mu) - s_min**(1 - mu))

def inverse_cdf_levy(xi, mu, s_min, s_max):
    if abs(mu - 1.0) < 1e-10:
        return s_min * (s_max / s_min) ** xi
    term = xi * (s_max**(1 - mu) - s_min**(1 - mu)) + s_min**(1 - mu)
    return term ** (1 / (1 - mu))

def apply_boundary(position, bounds):
    lon, lat, depth = position

    if lon < bounds['lon_min']: lon = 2 * bounds['lon_min'] - lon
    if lon > bounds['lon_max']: lon = 2 * bounds['lon_max'] - lon
    if lat < bounds['lat_min']: lat = 2 * bounds['lat_min'] - lat
    if lat > bounds['lat_max']: lat = 2 * bounds['lat_max'] - lat

    if depth < 0: depth = -depth
    if depth > bounds['z_max']: depth = 2 * bounds['z_max'] - depth

    return np.array([lon, lat, depth])

def simulate_shark(mu, days=T):
    bounds = {'lon_min': LON_MIN, 'lon_max': LON_MAX,
              'lat_min': LAT_MIN, 'lat_max': LAT_MAX, 'z_max': Z_MAX}
    start = np.array([LON_MIN, LAT_MAX, 0.0])

    bias = np.array([0.92, -0.39, 0.0])
    bias /= np.linalg.norm(bias)

    if abs(mu - 1.8) < 1e-5:
        omega = 0.88
    elif abs(mu - 1.4) < 1e-5:
        omega = 0.55
    elif abs(mu - 1.0) < 1e-5:
        omega = 0.25
    else:
        omega = 0.05

    traj = np.zeros((days + 1, 3))
    traj[0] = start
    step_lengths = []
    vertical_states = []

    current_depth = 0.0
    diving_phase = "surface"
    dive_target = 0.0

    for t in range(days):
        xi = np.random.uniform(0, 1)
        s = inverse_cdf_levy(xi, mu, s_min, s_max)
        s_deg = s / 111.0
        step_lengths.append(s)

        theta = np.random.uniform(0, 2 * np.pi)
        rand_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
        h_dir = omega * bias + (1 - omega) * rand_dir
        h_dir /= np.linalg.norm(h_dir)

        if diving_phase == "surface":
            if np.random.uniform(0, 1) < 0.25:
                diving_phase = "down"
                dive_target = np.random.uniform(0.6, Z_MAX) if s > 80 else np.random.uniform(0.1, 0.6)
                current_depth += np.random.uniform(0.1, 0.3)
            else:
                current_depth = max(0.0, current_depth + np.random.uniform(-0.05, 0.05))

        elif diving_phase == "down":
            step_down = np.random.uniform(0.15, 0.4)
            current_depth += step_down
            if current_depth >= dive_target:
                current_depth = min(current_depth, Z_MAX)
                diving_phase = "up"

        elif diving_phase == "up":
            step_up = np.random.uniform(0.1, 0.3)
            current_depth -= step_up
            if current_depth <= 0.05:
                current_depth = max(0.0, current_depth)
                diving_phase = "surface"

        vertical_states.append(diving_phase)

        next_pos = traj[t] + s_deg * h_dir
        next_pos[2] = current_depth

        traj[t + 1] = apply_boundary(next_pos, bounds)

    steps = np.array(step_lengths)
    total_dist = np.sum(steps)
    displacement = np.linalg.norm(traj[-1, :2] - traj[0, :2]) * 111.0
    straightness = displacement / total_dist if total_dist > 0 else 0

    msd = [np.mean(np.sum((traj[lag:] - traj[:-lag])[:, :2]**2, axis=1)) for lag in range(1, min(60, days))]

    return {
        'trajectory': traj, 'steps': steps, 'total_distance_km': total_dist,
        'displacement_km': displacement, 'straightness': straightness,
        'msd': np.array(msd), 'mu': mu, 'omega': omega, 'vertical_states': vertical_states,
        'C': normalization_constant(mu, s_min, s_max)
    }

print("✓ Mathematical framework compiled.")

# %% [code]
# --- 4. EXECUTE SIMULATIONS ---

print("\nRunning Simulations...")
results = {}
for mu in MU_VALUES:
    results[mu] = simulate_shark(mu)
    print(f"μ = {mu:<3} | Dist: {results[mu]['total_distance_km']:4.0f} km | SI: {results[mu]['straightness']:.4f}")

# ============================================================================
# COLLECT GLOBAL BOUNDS FOR ALL FIGURES (SMART PADDING)
# ============================================================================
all_lon = np.concatenate([results[m]['trajectory'][:, 0] for m in MU_VALUES])
all_lat = np.concatenate([results[m]['trajectory'][:, 1] for m in MU_VALUES])
all_depth = np.concatenate([results[m]['trajectory'][:, 2] for m in MU_VALUES])

# SMART PADDING: 10% for 2D, 8% for 3D (keeps curves INSIDE box)
PAD_2D = 0.10
PAD_3D = 0.08

x_range = all_lon.max() - all_lon.min()
y_range = all_lat.max() - all_lat.min()
z_range = all_depth.max() - all_depth.min()

X_LIM_2D = (all_lon.min() - PAD_2D * x_range, all_lon.max() + PAD_2D * x_range)
Y_LIM_2D = (all_lat.min() - PAD_2D * y_range, all_lat.max() + PAD_2D * y_range)
X_LIM_3D = (all_lon.min() - PAD_3D * x_range, all_lon.max() + PAD_3D * x_range)
Y_LIM_3D = (all_lat.min() - PAD_3D * y_range, all_lat.max() + PAD_3D * y_range)
Z_LIM_3D = (max(all_depth.min() - PAD_3D * z_range, -0.1), all_depth.max() + PAD_3D * z_range)

print(f"\n✓ Smart padding applied: 2D={PAD_2D*100:.0f}%, 3D={PAD_3D*100:.0f}%")

# %% [code]
# --- 5. DATA TABLES ---

# TABLE 1
t1_data = []
for mu in MU_VALUES:
    r = results[mu]
    z = r['trajectory'][:, 2]
    t1_data.append({
        'μ': mu, 'C(μ)': f"{r['C']:.4e}", 'Total Distance (km)': f"{r['total_distance_km']:.1f}",
        'Displacement (km)': f"{r['displacement_km']:.1f}", 'Straightness': f"{r['straightness']:.4f}",
        'Max Depth (km)': f"{np.max(z):.2f}", 'Mean Depth (km)': f"{np.mean(z):.2f}"
    })
print("\n--- TABLE 1: SIMULATION RESULTS ---")
display(pd.DataFrame(t1_data))

# TABLE 2
t2_data = []
for mu in MU_VALUES:
    states = results[mu]['vertical_states']
    total = len(states)
    t2_data.append({
        'μ': mu, 'Regime': BEHAVIOR[mu],
        'Surface %': f"{(states.count('surface')/total)*100:.1f}%",
        'Descent %': f"{(states.count('down')/total)*100:.1f}%",
        'Ascent %': f"{(states.count('up')/total)*100:.1f}%"
    })
print("\n--- TABLE 2: VERTICAL STATE ALLOCATION ---")
display(pd.DataFrame(t2_data))

# %% [code]
# --- 6. FIGURE 1: 2D Trajectories Map View ---

plt.figure(figsize=(12, 9))
for mu in MU_VALUES:
    t = results[mu]['trajectory']
    plt.plot(t[:, 0], t[:, 1], color=COLORS[mu], linewidth=2.5, label=f'μ = {mu} ({BEHAVIOR[mu]})')

plt.scatter(LON_MIN, LAT_MAX, color='green', s=180, edgecolors='black', zorder=5, label='Start Position')
plt.scatter(LON_MAX, LAT_MIN, color='red', s=180, marker='s', edgecolors='black', zorder=5, label='End Position')

# SMART PADDING for 2D
plt.xlim(X_LIM_2D[0], X_LIM_2D[1])
plt.ylim(Y_LIM_2D[0], Y_LIM_2D[1])

plt.xlabel('Longitude (°W)')
plt.ylabel('Latitude (°N)')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='lower left', fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.savefig('Fig1_2D_Trajectories.png', bbox_inches='tight', dpi=300)
plt.show()
print("✓ Figure 1 saved: Fig1_2D_Trajectories.png")

# %% [code]
# --- 7. FIGURE 2: 3D ALL TRAJECTORIES (FIXED - SMART PADDING) ---

fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

for mu in MU_VALUES:
    t = results[mu]['trajectory']
    ax.plot(t[:, 0], t[:, 1], t[:, 2], color=COLORS[mu], linewidth=2, label=f'μ = {mu}')

ax.scatter(LON_MIN, LAT_MAX, 0, color='green', s=120, edgecolors='black', label='Start')
ax.scatter(LON_MAX, LAT_MIN, 0, color='red', marker='s', s=120, edgecolors='black', label='End')

# SMART PADDING for 3D
ax.set_xlim(X_LIM_3D[0], X_LIM_3D[1])
ax.set_ylim(Y_LIM_3D[0], Y_LIM_3D[1])
ax.set_zlim(Z_LIM_3D[0], Z_LIM_3D[1])

ax.set_xlabel('Longitude (°W)', labelpad=10)
ax.set_ylabel('Latitude (°N)', labelpad=10)
ax.set_zlabel('Depth (km)', labelpad=10)
ax.set_title('3D Trajectories for Different Lévy Exponents μ\nALL CURVES CONTAINED INSIDE BOX',
             fontweight='bold', fontsize=14)
ax.view_init(elev=22, azim=-55)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('Fig2_3D_All_Trajectories.png', bbox_inches='tight', dpi=300)
plt.show()
print("✓ Figure 2 saved: Fig2_3D_All_Trajectories.png")

# %% [code]
# --- 8. FIGURE 3: Individual 3D Trajectories (FIXED - SMART PADDING) ---

fig, axes = plt.subplots(2, 2, figsize=(15, 13), subplot_kw={'projection': '3d'})
axes = axes.flatten()

for idx, mu in enumerate(MU_VALUES):
    ax = axes[idx]
    t = results[mu]['trajectory']
    n_points = len(t)

    for i in range(n_points - 1):
        ax.plot(t[i:i+2, 0], t[i:i+2, 1], t[i:i+2, 2],
                color=plt.cm.viridis(i/n_points), linewidth=2.2, alpha=0.8)

    ax.scatter(*t[0], color='green', s=80, edgecolors='black', label='Start')
    ax.scatter(*t[-1], color='red', marker='s', s=80, edgecolors='black', label='End')

    # INDIVIDUAL SMART PADDING for each subplot
    x_rng = t[:, 0].max() - t[:, 0].min()
    y_rng = t[:, 1].max() - t[:, 1].min()
    z_rng = t[:, 2].max() - t[:, 2].min()

    ax.set_xlim(t[:, 0].min() - PAD_3D * x_rng, t[:, 0].max() + PAD_3D * x_rng)
    ax.set_ylim(t[:, 1].min() - PAD_3D * y_rng, t[:, 1].max() + PAD_3D * y_rng)
    ax.set_zlim(max(t[:, 2].min() - PAD_3D * z_rng, -0.1), t[:, 2].max() + PAD_3D * z_rng)

    ax.set_title(f'μ = {mu} | {BEHAVIOR[mu]}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude (°W)', fontsize=9)
    ax.set_ylabel('Latitude (°N)', fontsize=9)
    ax.set_zlabel('Depth (km)', fontsize=9)
    ax.view_init(elev=25, azim=-60)

mappable = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, T))
mappable.set_array([])
cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.6, pad=0.05)
cbar.set_label('Time (days)', fontsize=12)

plt.suptitle('Individual 3D Trajectories (ALL CURVES INSIDE BOX)',
             fontweight='bold', fontsize=14, y=0.96)
plt.savefig('Fig3_Individual_3D_Tracks.png', bbox_inches='tight', dpi=300)
plt.show()
print("✓ Figure 3 saved: Fig3_Individual_3D_Tracks.png")

# %% [code]
# --- 9. FIGURE 4: Depth Profiles ---

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axes = axes.flatten()

for idx, mu in enumerate(MU_VALUES):
    ax = axes[idx]
    z = results[mu]['trajectory'][:, 2]
    days = np.arange(len(z))

    ax.plot(days, z, color=COLORS[mu], linewidth=1.8, label='Depth profile')
    ax.fill_between(days, 0, z, color=COLORS[mu], alpha=0.15)
    ax.axhline(0.0, color='blue', linestyle=':', label='Surface', linewidth=1.2)
    ax.axhline(Z_MAX, color='black', linestyle='--', label='Max depth', alpha=0.5)

    ax.set_title(f'μ = {mu} | Mean Depth: {np.mean(z):.2f} km | Max: {np.max(z):.2f} km',
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Depth (km)')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.1, Z_MAX + 0.1)
    ax.invert_yaxis()

plt.suptitle('Depth Profiles', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('Fig4_Depth_Profiles.png', bbox_inches='tight', dpi=300)
plt.show()
print("✓ Figure 4 saved: Fig4_Depth_Profiles.png")

# %% [code]
# --- 10. FIGURE 5: Step-Length Distributions ---

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axes = axes.flatten()

for idx, mu in enumerate(MU_VALUES):
    ax = axes[idx]
    steps = results[mu]['steps']
    C_val = results[mu]['C']

    ax.hist(steps, bins=35, density=True, color=COLORS[mu], alpha=0.6,
            edgecolor='black', label='Simulated')
    dom = np.linspace(s_min, s_max, 300)
    theory_pdf = C_val * (dom ** (-mu))
    ax.plot(dom, theory_pdf, color='black', linewidth=2.5,
            label=r'$p(s) = C(\mu) \cdot s^{-\mu}$')

    ax.set_title(f'μ = {mu} | C = {C_val:.4e}', fontweight='bold', fontsize=11)
    ax.set_xlabel('Step Length (km/day)')
    ax.set_ylabel('Probability Density')
    ax.set_xlim(0, s_max + 15)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=9)

plt.suptitle('Step-Length Distributions', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('Fig5_Step_Distributions.png', bbox_inches='tight', dpi=300)
plt.show()
print("✓ Figure 5 saved: Fig5_Step_Distributions.png")

# %% [code]
# --- 11. FIGURE 6: MSD ---

plt.figure(figsize=(11, 7))
all_msd = []
for mu in MU_VALUES:
    m_val = results[mu]['msd']
    all_msd.extend(m_val)
    lags = np.arange(1, len(m_val) + 1)
    plt.loglog(lags, m_val, color=COLORS[mu], marker='o', markersize=4,
               linewidth=2, label=f'μ = {mu}')

plt.xlabel('Lag Time (days)')
plt.ylabel('MSD (deg²)')
plt.title('Mean Squared Displacement (Log-Log Scale)', fontweight='bold', fontsize=14)
plt.xlim(0.8, 65)
plt.grid(True, which="both", alpha=0.25, linestyle=':')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('Fig6_MSD.png', bbox_inches='tight', dpi=300)
plt.show()
print("✓ Figure 6 saved: Fig6_MSD.png")

# %% [code]
# --- 12. FIGURE 7: KDE Density ---

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, mu in enumerate(MU_VALUES):
    ax = axes[idx]
    t = results[mu]['trajectory']

    x, y = t[:, 0], t[:, 1]
    kernel = gaussian_kde(np.vstack([x, y]))

    x_grid = np.linspace(X_LIM_2D[0], X_LIM_2D[1], 80)
    y_grid = np.linspace(Y_LIM_2D[0], Y_LIM_2D[1], 80)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.reshape(kernel(np.vstack([X.flatten(), Y.flatten()])), X.shape)

    cf = ax.contourf(X, Y, Z, levels=14, cmap='YlGnBu', alpha=0.85)
    ax.plot(x, y, color='black', linewidth=0.7, alpha=0.3, label='Path')
    ax.scatter(x[0], y[0], color='green', marker='o', s=60, edgecolors='black', label='Start')
    ax.scatter(x[-1], y[-1], color='red', marker='s', s=60, edgecolors='black', label='End')

    ax.set_xlim(X_LIM_2D[0], X_LIM_2D[1])
    ax.set_ylim(Y_LIM_2D[0], Y_LIM_2D[1])
    ax.set_title(f'μ = {mu}', fontweight='bold', fontsize=11)
    ax.set_xlabel('Longitude (°W)')
    ax.set_ylabel('Latitude (°N)')
    fig.colorbar(cf, ax=ax, shrink=0.8).set_label('Density')

plt.suptitle('Kernel Density Estimation - Spatial Utilization', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('Fig7_KDE_Density.png', bbox_inches='tight', dpi=300)
plt.show()
print("✓ Figure 7 saved: Fig7_KDE_Density.png")

# %% [code]
# --- 13. FIGURE 8: Sensitivity Analysis ---

mu_sweep = np.linspace(1.0, 2.0, 12)
sweep_dist, sweep_straight, sweep_step, sweep_depth = [], [], [], []

for m in mu_sweep:
    out = simulate_shark(m)
    sweep_dist.append(out['total_distance_km'])
    sweep_straight.append(out['straightness'])
    sweep_step.append(np.mean(out['steps']))
    sweep_depth.append(np.max(out['trajectory'][:, 2]))

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

axes[0,0].plot(mu_sweep, sweep_dist, 'o-', color='navy', linewidth=2)
axes[0,0].set_title('(a) Total Distance vs μ', fontweight='bold')
axes[0,0].set_ylabel('Distance (km)')
axes[0,0].grid(True, alpha=0.25)

axes[0,1].plot(mu_sweep, sweep_straight, 's-', color='darkorange', linewidth=2)
axes[0,1].set_title('(b) Straightness vs μ', fontweight='bold')
axes[0,1].set_ylabel('Straightness')
axes[0,1].grid(True, alpha=0.25)

axes[1,0].plot(mu_sweep, sweep_step, '^-', color='forestgreen', linewidth=2)
axes[1,0].set_title('(c) Mean Step vs μ', fontweight='bold')
axes[1,0].set_ylabel('Step (km/day)')
axes[1,0].grid(True, alpha=0.25)

axes[1,1].plot(mu_sweep, sweep_depth, 'v-', color='purple', linewidth=2)
axes[1,1].set_title('(d) Max Depth vs μ', fontweight='bold')
axes[1,1].set_ylabel('Depth (km)')
axes[1,1].axhline(y=Z_MAX, color='red', linestyle='--', alpha=0.5)
axes[1,1].grid(True, alpha=0.25)

for ax in axes.flatten():
    ax.set_xlabel('Lévy Exponent μ')
    ax.set_xlim(0.95, 2.05)

plt.suptitle('Sensitivity Analysis', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('Fig8_Sensitivity_Analysis.png', bbox_inches='tight', dpi=300)
plt.show()
print("✓ Figure 8 saved: Fig8_Sensitivity_Analysis.png")

# %% [code]
# --- 14. FIGURE 9: Velocity Profiles ---

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, mu in enumerate(MU_VALUES):
    ax = axes[idx]
    steps = results[mu]['steps']
    days = np.arange(1, len(steps) + 1)
    ax.plot(days, steps, color=COLORS[mu], linewidth=1.5, alpha=0.85)
    ax.axhline(np.mean(steps), color='black', linestyle=':',
               label=f'Mean: {np.mean(steps):.1f} km/day')
    ax.set_title(f'μ = {mu}', fontweight='bold', fontsize=11)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Step Length (km/day)')
    ax.set_ylim(0, s_max + 20)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=8)

plt.suptitle('Daily Step Length Profiles', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('Fig9_Velocity_Profiles.png', bbox_inches='tight', dpi=300)
plt.show()
print("✓ Figure 9 saved: Fig9_Velocity_Profiles.png")

# %% [code]
# --- 15. EXPORT DATA ---

print("\n" + "=" * 80)
print(" TABLE 3: CONSERVATION SUMMARY")
print("=" * 80)

t3_data = []
for mu in MU_VALUES:
    r = results[mu]
    risk = "CRITICAL DANGER" if mu <= 1.4 else "LOCAL MANAGEMENT"
    t3_data.append({
        'μ': mu,
        'Regime': BEHAVIOR[mu],
        'Straightness': f"{r['straightness']:.4f}",
        'Risk Profile': risk
    })

summary_df = pd.DataFrame(t3_data)
display(summary_df)

pd.DataFrame(t1_data).to_csv('Table1_Results.csv', index=False)
pd.DataFrame(t2_data).to_csv('Table2_Vertical_States.csv', index=False)
summary_df.to_csv('Table3_Conservation.csv', index=False)

print("\n" + "=" * 80)
print("✅ ALL SIMULATIONS COMPLETED SUCCESSFULLY!")
print("✅ ALL 3D CURVES CONTAINED INSIDE BOUNDING BOX")
print("✅ 9 FIGURES + 3 TABLES GENERATED")
print("=" * 80)
