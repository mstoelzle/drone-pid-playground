"""Cascaded PID controller for quadrotor."""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict

from .state import DroneState
from .dynamics import DroneParams


@dataclass
class PIDGains:
    """PID gains for a single axis."""
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    
    # Anti-windup limits
    integral_limit: float = 10.0
    
    # Output saturation
    output_limit: float = float('inf')


@dataclass
class PIDState:
    """Internal state of a PID controller."""
    integral: float = 0.0
    prev_error: float = 0.0
    prev_derivative: float = 0.0  # For derivative filtering


class PIDController:
    """Single-axis PID controller with anti-windup and derivative filtering."""
    
    def __init__(self, gains: PIDGains = None):
        self.gains = gains or PIDGains()
        self.state = PIDState()
        self.derivative_filter_alpha = 0.1  # Low-pass filter coefficient
    
    def reset(self):
        """Reset controller state."""
        self.state = PIDState()
    
    def compute(self, error: float, dt: float) -> float:
        """
        Compute PID output.
        
        Args:
            error: Current error (setpoint - actual)
            dt: Time step [s]
        
        Returns:
            Control output
        """
        if dt <= 0:
            return 0.0
        
        g = self.gains
        s = self.state
        
        # Proportional term
        p_term = g.kp * error
        
        # Integral term with anti-windup
        s.integral += error * dt
        s.integral = np.clip(s.integral, -g.integral_limit, g.integral_limit)
        i_term = g.ki * s.integral
        
        # Derivative term with filtering
        derivative = (error - s.prev_error) / dt
        filtered_derivative = (
            self.derivative_filter_alpha * derivative +
            (1 - self.derivative_filter_alpha) * s.prev_derivative
        )
        d_term = g.kd * filtered_derivative
        
        # Update state
        s.prev_error = error
        s.prev_derivative = filtered_derivative
        
        # Compute output with saturation
        output = p_term + i_term + d_term
        output = np.clip(output, -g.output_limit, g.output_limit)
        
        return output


@dataclass
class ControllerGains:
    """
    All gains for the cascaded controller.
    
    These gains are tuned for a typical ~1kg quadrotor based on:
    - PX4 Autopilot default gains and tuning guidelines
    - Standard cascaded PID control theory for quadrotors
    - Empirical tuning for simulation stability
    
    The cascaded structure is:
    - Inner loop (Rate): fastest, controls angular velocities (rad/s)
    - Middle loop (Attitude): controls orientation angles (rad)
    - Outer loop (Position/Velocity): slowest, controls position/velocity
    
    Rate controller gains are critical - these operate on angular velocity
    errors in rad/s and output torques. Values should be much smaller than
    attitude gains since the error magnitudes are different.
    
    References:
    - PX4 MC_ROLLRATE_P default: 0.15, MC_ROLLRATE_I: 0.2, MC_ROLLRATE_D: 0.003
    - PX4 MC_ROLL_P default: 6.5
    """
    
    # Attitude rate controller (inner loop) - operates on angular velocity error [rad/s]
    # These gains multiply with angular velocity error to produce torque commands
    # Based on PX4 defaults: P~0.15, I~0.2, D~0.003
    rate_roll: PIDGains = field(default_factory=lambda: PIDGains(kp=0.15, ki=0.2, kd=0.003, output_limit=1.0))
    rate_pitch: PIDGains = field(default_factory=lambda: PIDGains(kp=0.15, ki=0.2, kd=0.003, output_limit=1.0))
    rate_yaw: PIDGains = field(default_factory=lambda: PIDGains(kp=0.1, ki=0.1, kd=0.0, output_limit=0.5))
    
    # Attitude controller (middle loop) - operates on angle error [rad]
    # Outputs desired angular rate. Based on PX4 MC_ROLL_P default: 6.5
    # No D term typically needed - the rate controller handles damping
    att_roll: PIDGains = field(default_factory=lambda: PIDGains(kp=6.5, ki=0.0, kd=0.0, output_limit=3.0))
    att_pitch: PIDGains = field(default_factory=lambda: PIDGains(kp=6.5, ki=0.0, kd=0.0, output_limit=3.0))
    att_yaw: PIDGains = field(default_factory=lambda: PIDGains(kp=4.0, ki=0.0, kd=0.0, output_limit=2.0))
    
    # Position controller (outer loop) - operates on position error [m]
    # Outputs desired velocity
    pos_x: PIDGains = field(default_factory=lambda: PIDGains(kp=1.0, ki=0.0, kd=0.0, output_limit=2.0))
    pos_y: PIDGains = field(default_factory=lambda: PIDGains(kp=1.0, ki=0.0, kd=0.0, output_limit=2.0))
    pos_z: PIDGains = field(default_factory=lambda: PIDGains(kp=1.5, ki=0.1, kd=0.0, output_limit=3.0))
    
    # Velocity controller - operates on velocity error [m/s]
    # Outputs desired acceleration (converted to attitude for horizontal, thrust for vertical)
    vel_x: PIDGains = field(default_factory=lambda: PIDGains(kp=2.0, ki=0.1, kd=0.0, output_limit=3.0))
    vel_y: PIDGains = field(default_factory=lambda: PIDGains(kp=2.0, ki=0.1, kd=0.0, output_limit=3.0))
    vel_z: PIDGains = field(default_factory=lambda: PIDGains(kp=4.0, ki=0.5, kd=0.0, output_limit=10.0))


@dataclass
class Setpoints:
    """Control setpoints."""
    # Velocity setpoints
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (vx, vy, vz) m/s
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (p, q, r) rad/s
    
    # Position setpoint (optional, for position hold mode)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (x, y, z) m
    yaw: float = 0.0  # Yaw setpoint [rad]
    
    # Control mode
    use_position_control: bool = False
    
    # Gravity compensation (feedforward)
    # When True: thrust = m*g + control (drone hovers with zero gains)
    # When False: thrust = control only (drone falls with zero gains)
    use_gravity_compensation: bool = True


class CascadedPIDController:
    """
    Cascaded PID controller for quadrotor.
    
    Structure:
    - Outer loop: Position/Velocity -> desired attitude
    - Middle loop: Attitude -> desired angular rates
    - Inner loop: Angular rates -> torques
    """
    
    def __init__(self, gains: ControllerGains = None, params: DroneParams = None):
        self.gains = gains or ControllerGains()
        self.params = params or DroneParams()
        
        # Create individual PID controllers
        self._init_controllers()
    
    def _init_controllers(self):
        """Initialize all PID controllers."""
        g = self.gains
        
        # Rate controllers
        self.rate_roll_pid = PIDController(g.rate_roll)
        self.rate_pitch_pid = PIDController(g.rate_pitch)
        self.rate_yaw_pid = PIDController(g.rate_yaw)
        
        # Attitude controllers
        self.att_roll_pid = PIDController(g.att_roll)
        self.att_pitch_pid = PIDController(g.att_pitch)
        self.att_yaw_pid = PIDController(g.att_yaw)
        
        # Position controllers
        self.pos_x_pid = PIDController(g.pos_x)
        self.pos_y_pid = PIDController(g.pos_y)
        self.pos_z_pid = PIDController(g.pos_z)
        
        # Velocity controllers
        self.vel_x_pid = PIDController(g.vel_x)
        self.vel_y_pid = PIDController(g.vel_y)
        self.vel_z_pid = PIDController(g.vel_z)
    
    def update_gains(self, gains: ControllerGains):
        """Update all controller gains."""
        self.gains = gains
        
        # Update individual controllers
        self.rate_roll_pid.gains = gains.rate_roll
        self.rate_pitch_pid.gains = gains.rate_pitch
        self.rate_yaw_pid.gains = gains.rate_yaw
        
        self.att_roll_pid.gains = gains.att_roll
        self.att_pitch_pid.gains = gains.att_pitch
        self.att_yaw_pid.gains = gains.att_yaw
        
        self.pos_x_pid.gains = gains.pos_x
        self.pos_y_pid.gains = gains.pos_y
        self.pos_z_pid.gains = gains.pos_z
        
        self.vel_x_pid.gains = gains.vel_x
        self.vel_y_pid.gains = gains.vel_y
        self.vel_z_pid.gains = gains.vel_z
    
    def reset(self):
        """Reset all controller states."""
        self.rate_roll_pid.reset()
        self.rate_pitch_pid.reset()
        self.rate_yaw_pid.reset()
        
        self.att_roll_pid.reset()
        self.att_pitch_pid.reset()
        self.att_yaw_pid.reset()
        
        self.pos_x_pid.reset()
        self.pos_y_pid.reset()
        self.pos_z_pid.reset()
        
        self.vel_x_pid.reset()
        self.vel_y_pid.reset()
        self.vel_z_pid.reset()
    
    def compute(
        self,
        state: DroneState,
        setpoints: Setpoints,
        dt: float,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute control outputs using cascaded PID.
        
        Args:
            state: Current drone state
            setpoints: Desired setpoints
            dt: Time step [s]
        
        Returns:
            Tuple of (thrust [N], torques [N*m])
        """
        if dt <= 0:
            return self.params.hover_thrust, np.zeros(3)
        
        # Get current state
        euler = state.get_euler_angles()  # roll, pitch, yaw
        omega = state.angular_velocity  # p, q, r
        
        # === Outer loop: Position/Velocity control ===
        if setpoints.use_position_control:
            # Position control -> velocity setpoint
            pos_error = setpoints.position - state.position
            vel_sp = np.array([
                self.pos_x_pid.compute(pos_error[0], dt),
                self.pos_y_pid.compute(pos_error[1], dt),
                self.pos_z_pid.compute(pos_error[2], dt),
            ])
        else:
            vel_sp = setpoints.velocity
        
        # Velocity control -> desired acceleration/attitude
        vel_error = vel_sp - state.velocity
        
        # Desired accelerations from velocity controller
        ax_des = self.vel_x_pid.compute(vel_error[0], dt)
        ay_des = self.vel_y_pid.compute(vel_error[1], dt)
        az_des = self.vel_z_pid.compute(vel_error[2], dt)
        
        # Convert horizontal accelerations to desired roll/pitch
        # Assuming small angles: ax ≈ g*pitch, ay ≈ -g*roll (for yaw=0)
        yaw = euler[2]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Rotate desired accelerations to body frame
        ax_body = cos_yaw * ax_des + sin_yaw * ay_des
        ay_body = -sin_yaw * ax_des + cos_yaw * ay_des
        
        # Desired angles (small angle approximation)
        pitch_des = np.clip(ax_body / self.params.gravity, -0.5, 0.5)
        roll_des = np.clip(-ay_body / self.params.gravity, -0.5, 0.5)
        
        # Thrust from vertical acceleration
        if setpoints.use_gravity_compensation:
            # With gravity compensation: thrust = m*g + control
            # Drone hovers even with zero gains
            thrust = self.params.mass * (self.params.gravity + az_des)
        else:
            # Without gravity compensation: thrust = control only
            # Drone falls with zero gains (more realistic for PID tuning)
            thrust = self.params.mass * az_des
        thrust = np.clip(thrust, 0, 4 * self.params.hover_thrust)
        
        # Yaw setpoint
        yaw_des = setpoints.yaw
        
        # === Middle loop: Attitude control ===
        roll_error = roll_des - euler[0]
        pitch_error = pitch_des - euler[1]
        yaw_error = self._wrap_angle(yaw_des - euler[2])
        
        # Desired angular rates from attitude controller
        p_des = self.att_roll_pid.compute(roll_error, dt)
        q_des = self.att_pitch_pid.compute(pitch_error, dt)
        r_des = self.att_yaw_pid.compute(yaw_error, dt)
        
        # Add feedforward angular velocity setpoints
        p_des += setpoints.angular_velocity[0]
        q_des += setpoints.angular_velocity[1]
        r_des += setpoints.angular_velocity[2]
        
        # === Inner loop: Rate control ===
        p_error = p_des - omega[0]
        q_error = q_des - omega[1]
        r_error = r_des - omega[2]
        
        # Compute torques
        tau_x = self.rate_roll_pid.compute(p_error, dt)
        tau_y = self.rate_pitch_pid.compute(q_error, dt)
        tau_z = self.rate_yaw_pid.compute(r_error, dt)
        
        torques = np.array([tau_x, tau_y, tau_z])
        
        return thrust, torques
    
    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_gains_dict(self) -> Dict[str, Dict[str, float]]:
        """Get all gains as a nested dictionary for GUI."""
        g = self.gains
        return {
            'rate_roll': {'kp': g.rate_roll.kp, 'ki': g.rate_roll.ki, 'kd': g.rate_roll.kd},
            'rate_pitch': {'kp': g.rate_pitch.kp, 'ki': g.rate_pitch.ki, 'kd': g.rate_pitch.kd},
            'rate_yaw': {'kp': g.rate_yaw.kp, 'ki': g.rate_yaw.ki, 'kd': g.rate_yaw.kd},
            'att_roll': {'kp': g.att_roll.kp, 'ki': g.att_roll.ki, 'kd': g.att_roll.kd},
            'att_pitch': {'kp': g.att_pitch.kp, 'ki': g.att_pitch.ki, 'kd': g.att_pitch.kd},
            'att_yaw': {'kp': g.att_yaw.kp, 'ki': g.att_yaw.ki, 'kd': g.att_yaw.kd},
            'pos_x': {'kp': g.pos_x.kp, 'ki': g.pos_x.ki, 'kd': g.pos_x.kd},
            'pos_y': {'kp': g.pos_y.kp, 'ki': g.pos_y.ki, 'kd': g.pos_y.kd},
            'pos_z': {'kp': g.pos_z.kp, 'ki': g.pos_z.ki, 'kd': g.pos_z.kd},
            'vel_x': {'kp': g.vel_x.kp, 'ki': g.vel_x.ki, 'kd': g.vel_x.kd},
            'vel_y': {'kp': g.vel_y.kp, 'ki': g.vel_y.ki, 'kd': g.vel_y.kd},
            'vel_z': {'kp': g.vel_z.kp, 'ki': g.vel_z.ki, 'kd': g.vel_z.kd},
        }

