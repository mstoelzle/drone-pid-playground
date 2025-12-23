#!/usr/bin/env python3
"""
Drone PID Controller Tuning Playground

A real-time drone simulator with interactive PID tuning via a web-based GUI.

Usage:
    python run.py [--port PORT]

Then open http://localhost:PORT in your browser.
"""

import argparse
import time
import numpy as np

from drone.state import DroneState
from drone.dynamics import DroneDynamics, DroneParams
from drone.controller import CascadedPIDController
from gui.visualizer import DroneVisualizer
from gui.plots import PlotManager


def main():
    """Main entry point for the drone simulator."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Drone PID Controller Tuning Playground")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number for the web server (default: 8080)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Drone PID Controller Tuning Playground")
    print("=" * 60)
    print()
    
    # Initialize drone parameters
    # Using default values which are based on a typical ~1kg, 250mm class quadrotor
    # These can be customized for different drone configurations
    params = DroneParams(
        mass=1.0,           # kg - typical for small drone with battery
        Ixx=0.0082,         # kg*m^2 - roll inertia
        Iyy=0.0082,         # kg*m^2 - pitch inertia
        Izz=0.0140,         # kg*m^2 - yaw inertia
        linear_drag=0.1,    # N/(m/s) - aerodynamic drag
        angular_drag=0.02,  # N*m/(rad/s) - rotational damping
        arm_length=0.17,    # m - motor to center distance
    )
    
    # Initialize components
    print("[1/4] Initializing drone dynamics...")
    dynamics = DroneDynamics(params)
    
    print("[2/4] Initializing controller...")
    controller = CascadedPIDController(params=params)
    
    print("[3/4] Starting Viser server...")
    visualizer = DroneVisualizer(params=params, port=args.port)
    
    print("[4/4] Setting up plots...")
    plot_manager = PlotManager(visualizer.server, visualizer.plot_data)
    
    print()
    print("-" * 60)
    print(f"  Server running at: http://localhost:{args.port}")
    print("-" * 60)
    print()
    print("Controls:")
    print("  - Use sliders to adjust PID gains in real-time")
    print("  - Set velocity/position setpoints to test response")
    print("  - Click 'Apply Disturbance' to test controller recovery")
    print("  - Click 'Reset' to return drone to origin")
    print("  - Click 'Pause' to freeze the simulation")
    print()
    print("Direction Control:")
    print("  - Position Mode: Direction buttons adjust position setpoint")
    print("  - Velocity Mode: Enable 'Keyboard Mode' for button-controlled velocity")
    print("  - When Keyboard Mode is on, buttons toggle velocity in each direction")
    print("  - Click 'STOP' to reset all velocities to zero")
    print()
    print("Press Ctrl+C to stop the server.")
    print()
    
    # Initialize drone state at hover
    state = DroneState(
        position=np.array([0.0, 0.0, 1.0]),  # Start 1m above ground
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    
    # Set initial position setpoint
    visualizer.setpoints.position = np.array([0.0, 0.0, 1.0])
    
    # Reset callback
    def on_reset():
        nonlocal state
        state = DroneState(
            position=np.array([0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        )
        controller.reset()
        print("Drone reset to initial state")
    
    visualizer.on_reset = on_reset
    
    # Simulation parameters
    target_dt = 1.0 / 60.0  # 60 Hz visualization update
    sim_substeps = 10  # Physics substeps for stability
    plot_update_interval = 5  # Update plots every N frames
    
    last_time = time.time()
    frame_count = 0
    
    # Control output for logging
    thrust = params.hover_thrust
    torques = np.zeros(3)
    
    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            
            # Rate limiting - wait if we're running too fast
            if dt < target_dt:
                time.sleep(target_dt - dt)
                current_time = time.time()
                dt = current_time - last_time
            
            last_time = current_time
            
            # Skip if paused
            if visualizer.is_paused():
                time.sleep(0.01)
                continue
            
            # Clamp dt to avoid instability
            dt = min(dt, 0.1)
            
            # Get current gains and setpoints from GUI
            gains = visualizer.get_gains()
            setpoints = visualizer.get_setpoints()
            
            # Update controller gains and anti-windup setting
            controller.update_gains(gains)
            controller.set_anti_windup(visualizer.get_anti_windup_enabled())
            
            # Compute control outputs
            thrust, torques = controller.compute(state, setpoints, dt)
            
            # Add disturbance torque and force if active
            disturbance_torque = visualizer.get_disturbance_torque()
            disturbance_force = visualizer.get_disturbance_force()
            total_torques = torques + disturbance_torque
            
            # Step dynamics with external force
            state = dynamics.step(state, thrust, total_torques, dt, substeps=sim_substeps,
                                  external_force=disturbance_force)
            
            # Ground collision (simple constraint)
            if state.position[2] < 0:
                state.position[2] = 0
                state.velocity[2] = max(0, state.velocity[2])
            
            # Update visualization
            visualizer.update(state, thrust, torques)
            
            # Update plots periodically (not every frame for performance)
            frame_count += 1
            if frame_count % plot_update_interval == 0:
                plot_manager.update()
            
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    print("Goodbye!")


if __name__ == "__main__":
    main()

