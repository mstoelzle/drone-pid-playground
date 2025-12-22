"""6DOF Quadrotor dynamics with RK4 integration."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .state import DroneState


@dataclass
class DroneParams:
    """
    Physical parameters of the quadrotor.
    
    Default values are based on a typical 250mm class quadrotor (~1kg),
    similar to the Crazyflie (scaled up) or small racing/research drones.
    
    References:
    - PX4 SITL Iris quadrotor parameters
    - Crazyflie 2.0 parameters (scaled)
    - Standard quadrotor dynamics literature
    """
    mass: float = 1.0  # kg - typical for 250-300mm class drone with battery
    
    # Inertia tensor (diagonal, body frame) [kg*m^2]
    # For a ~1kg drone with 170mm arm length (250mm diagonal):
    # Ixx ≈ Iyy ≈ m * L^2 / 12 where L is characteristic length
    # Values below are typical for a 1kg, 250mm class quadrotor
    Ixx: float = 0.0082  # kg*m^2 - roll inertia
    Iyy: float = 0.0082  # kg*m^2 - pitch inertia  
    Izz: float = 0.0140  # kg*m^2 - yaw inertia (typically ~1.7x roll/pitch)
    
    # Drag coefficients
    # Based on standard quadrotor aerodynamics at low speeds
    linear_drag: float = 0.1  # Linear drag coefficient [N/(m/s)]
    angular_drag: float = 0.02  # Angular drag coefficient [N*m/(rad/s)]
    
    # Arm length (motor to center distance)
    arm_length: float = 0.17  # m - for 250mm diagonal (~340mm motor-to-motor)
    
    # Gravity
    gravity: float = 9.81  # m/s^2
    
    @property
    def inertia(self) -> np.ndarray:
        """Get the inertia tensor as a diagonal matrix."""
        return np.diag([self.Ixx, self.Iyy, self.Izz])
    
    @property
    def inertia_inv(self) -> np.ndarray:
        """Get the inverse inertia tensor."""
        return np.diag([1.0/self.Ixx, 1.0/self.Iyy, 1.0/self.Izz])
    
    @property
    def hover_thrust(self) -> float:
        """Thrust required to hover."""
        return self.mass * self.gravity


class DroneDynamics:
    """
    6DOF Quadrotor dynamics simulator.
    
    Uses quaternion-based orientation and RK4 integration.
    """
    
    def __init__(self, params: DroneParams = None):
        self.params = params or DroneParams()
    
    def compute_derivatives(
        self,
        state: DroneState,
        thrust: float,
        torques: np.ndarray,
    ) -> np.ndarray:
        """
        Compute state derivatives given current state and control inputs.
        
        Args:
            state: Current drone state
            thrust: Total thrust along body z-axis [N]
            torques: (tau_x, tau_y, tau_z) torques in body frame [N*m]
        
        Returns:
            State derivative as a flat array
        """
        p = self.params
        
        # Position derivative = velocity
        pos_dot = state.velocity
        
        # Get rotation matrix (body to world)
        R = state.get_rotation_matrix()
        
        # Thrust in world frame (along body z-axis)
        thrust_world = R @ np.array([0, 0, thrust])
        
        # Gravity in world frame
        gravity_world = np.array([0, 0, -p.mass * p.gravity])
        
        # Linear drag in world frame
        drag_world = -p.linear_drag * state.velocity
        
        # Acceleration
        vel_dot = (thrust_world + gravity_world + drag_world) / p.mass
        
        # Quaternion derivative
        # q_dot = 0.5 * q ⊗ [0, omega]
        omega = state.angular_velocity
        q = state.quaternion
        
        # Quaternion multiplication: q ⊗ [0, omega]
        # Using scalar-first convention (w, x, y, z)
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        quat_dot = 0.5 * self._quaternion_multiply(q, omega_quat)
        
        # Angular acceleration
        # I * omega_dot = torques - omega x (I * omega) - drag
        I = p.inertia
        I_inv = p.inertia_inv
        
        # Gyroscopic term
        gyro = np.cross(omega, I @ omega)
        
        # Angular drag
        angular_drag = -p.angular_drag * omega
        
        # Angular acceleration
        omega_dot = I_inv @ (torques - gyro + angular_drag)
        
        return np.concatenate([pos_dot, vel_dot, quat_dot, omega_dot])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions (scalar-first convention)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    
    def step_rk4(
        self,
        state: DroneState,
        thrust: float,
        torques: np.ndarray,
        dt: float,
    ) -> DroneState:
        """
        Integrate state forward using RK4.
        
        Args:
            state: Current drone state
            thrust: Total thrust [N]
            torques: Body torques [N*m]
            dt: Time step [s]
        
        Returns:
            New drone state after integration
        """
        y = state.to_array()
        
        # RK4 stages
        k1 = self.compute_derivatives(DroneState.from_array(y), thrust, torques)
        k2 = self.compute_derivatives(DroneState.from_array(y + 0.5*dt*k1), thrust, torques)
        k3 = self.compute_derivatives(DroneState.from_array(y + 0.5*dt*k2), thrust, torques)
        k4 = self.compute_derivatives(DroneState.from_array(y + dt*k3), thrust, torques)
        
        # Combine
        y_new = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        new_state = DroneState.from_array(y_new)
        new_state.normalize_quaternion()
        
        return new_state
    
    def step(
        self,
        state: DroneState,
        thrust: float,
        torques: np.ndarray,
        dt: float,
        substeps: int = 10,
    ) -> DroneState:
        """
        Step simulation with substeps for stability.
        
        Args:
            state: Current drone state
            thrust: Total thrust [N]
            torques: Body torques [N*m]
            dt: Total time step [s]
            substeps: Number of integration substeps
        
        Returns:
            New drone state
        """
        dt_sub = dt / substeps
        current = state.copy()
        
        for _ in range(substeps):
            current = self.step_rk4(current, thrust, torques, dt_sub)
        
        return current

