"""Drone state representation with quaternion orientation."""

import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DroneState:
    """
    Complete state representation for a 6DOF quadrotor drone.
    
    Attributes:
        position: (x, y, z) position in world frame [m]
        velocity: (vx, vy, vz) linear velocity in world frame [m/s]
        quaternion: (w, x, y, z) orientation quaternion (scalar-first)
        angular_velocity: (p, q, r) angular velocity in body frame [rad/s]
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def __post_init__(self):
        """Ensure all arrays are numpy arrays with correct shape."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.quaternion = np.asarray(self.quaternion, dtype=np.float64)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)
        self.normalize_quaternion()
    
    def normalize_quaternion(self):
        """Normalize the quaternion to unit length."""
        norm = np.linalg.norm(self.quaternion)
        if norm > 1e-10:
            self.quaternion = self.quaternion / norm
        else:
            self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Get the rotation matrix from body to world frame."""
        # scipy uses scalar-last (x, y, z, w), we use scalar-first (w, x, y, z)
        quat_scipy = np.array([
            self.quaternion[1],  # x
            self.quaternion[2],  # y
            self.quaternion[3],  # z
            self.quaternion[0],  # w
        ])
        return Rotation.from_quat(quat_scipy).as_matrix()
    
    def get_euler_angles(self) -> np.ndarray:
        """
        Get Euler angles (roll, pitch, yaw) in radians.
        Uses ZYX convention (yaw-pitch-roll).
        """
        quat_scipy = np.array([
            self.quaternion[1],  # x
            self.quaternion[2],  # y
            self.quaternion[3],  # z
            self.quaternion[0],  # w
        ])
        # Returns [roll, pitch, yaw]
        return Rotation.from_quat(quat_scipy).as_euler('xyz')
    
    def set_euler_angles(self, roll: float, pitch: float, yaw: float):
        """Set orientation from Euler angles (roll, pitch, yaw) in radians."""
        rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
        quat_scipy = rot.as_quat()  # (x, y, z, w)
        self.quaternion = np.array([
            quat_scipy[3],  # w
            quat_scipy[0],  # x
            quat_scipy[1],  # y
            quat_scipy[2],  # z
        ])
    
    def get_body_z_axis(self) -> np.ndarray:
        """Get the body z-axis (thrust direction) in world frame."""
        R = self.get_rotation_matrix()
        return R[:, 2]
    
    def transform_to_body(self, world_vec: np.ndarray) -> np.ndarray:
        """Transform a vector from world frame to body frame."""
        R = self.get_rotation_matrix()
        return R.T @ world_vec
    
    def transform_to_world(self, body_vec: np.ndarray) -> np.ndarray:
        """Transform a vector from body frame to world frame."""
        R = self.get_rotation_matrix()
        return R @ body_vec
    
    def copy(self) -> 'DroneState':
        """Create a deep copy of the state."""
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            angular_velocity=self.angular_velocity.copy(),
        )
    
    def reset(self):
        """Reset state to initial hover condition."""
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.angular_velocity = np.zeros(3)
    
    def to_array(self) -> np.ndarray:
        """Convert state to a flat array for integration."""
        return np.concatenate([
            self.position,
            self.velocity,
            self.quaternion,
            self.angular_velocity,
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'DroneState':
        """Create state from a flat array."""
        return cls(
            position=arr[0:3],
            velocity=arr[3:6],
            quaternion=arr[6:10],
            angular_velocity=arr[10:13],
        )
    
    def __repr__(self) -> str:
        euler = np.degrees(self.get_euler_angles())
        return (
            f"DroneState(\n"
            f"  pos=[{self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}] m\n"
            f"  vel=[{self.velocity[0]:.2f}, {self.velocity[1]:.2f}, {self.velocity[2]:.2f}] m/s\n"
            f"  euler=[{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] deg\n"
            f"  omega=[{self.angular_velocity[0]:.2f}, {self.angular_velocity[1]:.2f}, {self.angular_velocity[2]:.2f}] rad/s\n"
            f")"
        )

