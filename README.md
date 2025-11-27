# FFT_gravity_solver
# FFT Gravity Solver — Particle-Mesh N-Body Galaxy Simulation

This repository implements a full **particle-mesh (PM) N-body simulation** designed to model the **gravitational collapse of a galaxy** containing up to **100 million particles**.  
The framework includes:

- Plummer-sphere initial conditions (stars, dark matter, black holes)
- Cloud-in-Cell (CIC) mass assignment
- 3D FFT-based Poisson solver using FFTW3
- Zero-padding for isolated boundary conditions
- Finite-difference force computation
- CIC force interpolation to particles
- Symplectic time integration (Leapfrog & 4th-order Yoshida)
- Monte-Carlo black hole collision + merger model
- Full OpenMP parallelization

This project was developed for **PHYD57 — Computational Astrophysics (University of Toronto)**.

---

# 🎥 Simulation Demo

Below is a visualization of the **XY-plane projection** of the galaxy collapse.

> After uploading the video to GitHub (e.g., `/assets/galaxy_xy.mp4`),  
> **replace the link below** with the correct GitHub file path.

[![Galaxy Simulation](https://img.shields.io/badge/Click_to_Play-Galaxy_Simulation-blue?style=for-the-badge)](path/to/galaxy_xy.mp4)

---

# ✨ Features

### **1. Initial Conditions**
- Multi-component Plummer spheres  
- 20M stars, 80M dark matter particles, 10 seed black holes  
- Cold-collapse velocity initialization  
- Center-of-mass correction

### **2. Cloud-in-Cell (CIC) Mass Assignment**
Each particle deposits mass onto the 3D grid using trilinear interpolation.  
Guarantees:
- Exact mass conservation  
- Smooth density representation  
- Momentum-conserving force interpolation  

### **3. FFTW3 Poisson Solver**
Solves:

\[
\nabla^2 \Phi = 4 \pi G \rho
\]

via:

- 3D complex-to-complex FFT  
- Discrete Laplacian eigenvalues
- Zero-padding to remove periodic image forces  

### **4. Finite-Difference Forces**
Central differences compute:

\[
F = -\nabla\Phi
\]

Boundary-corrected stencils used at edges.

### **5. Symplectic Integrators**
- Leapfrog (Velocity Verlet)
- Yoshida 4th-order composition scheme  
Preserves Hamiltonian structure & minimizes long-term energy drift.

### **6. Black Hole Mergers**
Includes:
- Stellar capture radius \( R_{\text{star}} = 1.0h \)
- BH–BH merger radius \( R_{\text{BH}} = 1.5h \)
- Momentum-conserving merging  
- Spatial hashing via cell-linked lists for efficient detection  

### **7. OpenMP Parallelization**
Parallelized modules:
- Initialization  
- Mass assignment  
- Force gathering  
- FFTW3  
- Collision physics  
- Grid operations  


# 🧪 Validation Summary

### ✔ CIC Mass Conservation  
\[
|M_{\text{grid}} - M_{\text{total}}| < 10^{-14}
\]

### ✔ Plummer Profile  
Matches analytic cumulative mass curves.

### ✔ Center-of-Mass Corrections  
- COM velocity < 10⁻¹⁵  
- Centered in the simulation box  

### ✔ FFT Round-Trip  
Error < 10⁻¹³

### ✔ Poisson Solver  
Produces correct potential wells and radial forces.

### ✔ Symplectic Integrators  
Leapfrog and Yoshida tested on:
- Harmonic potentials  
- Kepler orbits  
Energy drift behaves as expected.

### ✔ Black Hole Collisions  
Tested:
- Star capture  
- BH–BH merging  
- Conservation of mass & momentum  
- No collisions for dark matter particles  

---

# 🚧 Current Status

### **Completed:**
- Initial conditions  
- CIC scatter/gather  
- Poisson solver  
- Force calculation  
- Symplectic integrators  
- BH collision physics  
- OpenMP parallelization  
- Full validation suite  

### **In Progress:**
- Production-scale timestep loop  
- Simulation I/O and snapshot system  
- Visualization tools  
- Final scientific analysis (virialization, BH growth, etc.)

---

# 🔧 Build & Run Instructions

### **Dependencies**
- FFTW3 (with threads)
- gcc or icpc
- OpenMP
- Linux/macOS recommended

### **Build**
```bash
make

./gravity_sim <n_steps> <output_every>

📚 References

Hockney & Eastwood — Computer Simulation Using Particles

Binney & Tremaine — Galactic Dynamics

FFTW3 documentation

Yoshida (1990) — Symplectic integrators

Plummer (1911) — Plummer sphere model


👥 Authors

Kaixin, David, Aryan, Aakash
PHYD57 — Computational Astrophysics
University of Toronto (2025)
