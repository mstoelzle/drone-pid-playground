# 🚁 Drone PID Tuning Playground

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive, real-time quadrotor drone simulator with a web-based GUI for tuning cascaded PID controllers. Built with [Viser](https://github.com/nerfstudio-project/viser) for 3D visualization.

<p align="center">
  <img src="assets/screenshot.png" alt="Drone PID Playground Screenshot" width="800">
</p>

## ✨ Features

- **3D Visualization** — Real-time quadrotor visualization in your browser
- **Cascaded PID Control** — Tune attitude rate, attitude, velocity, and position controllers
- **Interactive Sliders** — Adjust all PID gains (Kp, Ki, Kd) in real-time
- **Live Plots** — Monitor orientation, velocity, and position response
- **Disturbance Testing** — Apply impulse torques to test controller robustness
- **Position & Velocity Modes** — Switch between position hold and velocity setpoint control

## 🚀 Quick Start

### Installation

#### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/mstoelzle/drone-pid-playground.git
cd drone-pid-playground

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/mstoelzle/drone-pid-playground.git
cd drone-pid-playground

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Simulator

```bash
python run.py
```

Then open your browser to **http://localhost:8080** to access the GUI.

## 🎮 GUI Controls

| Control | Description |
|---------|-------------|
| **Pause/Resume** | Freeze or continue simulation |
| **Reset** | Return drone to initial hover position |
| **Clear Plots** | Clear plot history |
| **Apply Disturbance** | Apply an impulse torque to test recovery |

### Setpoints

- **Linear Velocity** — Set target velocities (Vx, Vy, Vz)
- **Angular Velocity** — Set target angular rates (roll, pitch, yaw rates)
- **Position Hold Mode** — Enable to track position setpoints instead of velocities
- **Gravity Compensation** — Toggle feedforward gravity compensation

### PID Tuning

Organized by control loop hierarchy:

1. **Rate PID (Inner Loop)** — Controls angular rates → outputs torques
2. **Attitude PID** — Controls orientation → outputs rate setpoints
3. **Velocity PID** — Controls velocity → outputs attitude setpoints
4. **Position PID (Outer Loop)** — Controls position → outputs velocity setpoints

## 🏗️ Control Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Cascaded PID Control                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Position    Velocity    Attitude    Rate                       │
│  Setpoint    Setpoint    Setpoint    Setpoint                   │
│     │           │           │           │                       │
│     ▼           ▼           ▼           ▼                       │
│  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐                     │
│  │ Pos  │──▶│ Vel  │──▶│ Att  │──▶│ Rate │──▶ Thrust & Torques │
│  │ PID  │   │ PID  │   │ PID  │   │ PID  │                     │
│  └──────┘   └──────┘   └──────┘   └──────┘                     │
│   (outer)                          (inner)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Drone Model

The simulator implements a 6DOF rigid body quadrotor with:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Mass | 1.0 kg | Total mass |
| Ixx, Iyy | 0.01 kg·m² | Roll/pitch inertia |
| Izz | 0.02 kg·m² | Yaw inertia |
| Arm Length | 0.25 m | Motor arm length |
| Linear Drag | 0.1 | Velocity damping |
| Angular Drag | 0.01 | Angular velocity damping |

**Physics features:**
- Quaternion-based orientation (no gimbal lock)
- RK4 integration with configurable substeps
- Ground collision constraint

## 📁 Project Structure

```
drone-pid-playground/
├── assets/
│   └── screenshot.png # GUI screenshot
├── drone/
│   ├── state.py       # Drone state representation
│   ├── dynamics.py    # 6DOF physics simulation
│   └── controller.py  # Cascaded PID controller
├── gui/
│   ├── visualizer.py  # Viser 3D visualization
│   └── plots.py       # Real-time plotting
├── run.py             # Main entry point
├── requirements.txt   # Python dependencies
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Feel free to:

- 🐛 Report bugs
- 💡 Suggest new features
- 🔧 Submit pull requests

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Viser](https://github.com/nerfstudio-project/viser) — Excellent 3D visualization library
- [SciPy](https://scipy.org/) — Rotation utilities
- [Plotly](https://plotly.com/) — Interactive plotting
