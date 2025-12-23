"""Viser-based 3D visualization for the drone simulator."""

import numpy as np
import viser
import viser.transforms as tf
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from collections import deque
import time

from drone.state import DroneState
from drone.controller import ControllerGains, PIDGains, Setpoints
from drone.dynamics import DroneParams


@dataclass
class PlotData:
    """Data for real-time plotting."""
    max_length: int = 500
    
    time: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # Orientation
    roll: deque = field(default_factory=lambda: deque(maxlen=500))
    pitch: deque = field(default_factory=lambda: deque(maxlen=500))
    yaw: deque = field(default_factory=lambda: deque(maxlen=500))
    
    roll_sp: deque = field(default_factory=lambda: deque(maxlen=500))
    pitch_sp: deque = field(default_factory=lambda: deque(maxlen=500))
    yaw_sp: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # Angular velocity
    p: deque = field(default_factory=lambda: deque(maxlen=500))
    q: deque = field(default_factory=lambda: deque(maxlen=500))
    r: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # Position
    x: deque = field(default_factory=lambda: deque(maxlen=500))
    y: deque = field(default_factory=lambda: deque(maxlen=500))
    z: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # Velocity
    vx: deque = field(default_factory=lambda: deque(maxlen=500))
    vy: deque = field(default_factory=lambda: deque(maxlen=500))
    vz: deque = field(default_factory=lambda: deque(maxlen=500))
    
    vx_sp: deque = field(default_factory=lambda: deque(maxlen=500))
    vy_sp: deque = field(default_factory=lambda: deque(maxlen=500))
    vz_sp: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # Control outputs
    thrust: deque = field(default_factory=lambda: deque(maxlen=500))
    tau_x: deque = field(default_factory=lambda: deque(maxlen=500))
    tau_y: deque = field(default_factory=lambda: deque(maxlen=500))
    tau_z: deque = field(default_factory=lambda: deque(maxlen=500))
    
    def clear(self):
        """Clear all plot data."""
        for attr in ['time', 'roll', 'pitch', 'yaw', 'roll_sp', 'pitch_sp', 'yaw_sp',
                     'p', 'q', 'r', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                     'vx_sp', 'vy_sp', 'vz_sp', 'thrust', 'tau_x', 'tau_y', 'tau_z']:
            getattr(self, attr).clear()


class DroneVisualizer:
    """
    Viser-based visualization for the drone PID playground.
    
    Features:
    - 3D drone visualization with quadrotor mesh
    - Ground grid and world frame
    - PID gain sliders organized by controller
    - Setpoint controls
    - Real-time plots
    """
    
    def __init__(self, params: DroneParams = None, port: int = 8080):
        self.params = params or DroneParams()
        self.port = port
        
        # State
        self.gains = ControllerGains()
        self.setpoints = Setpoints()
        self.plot_data = PlotData()
        self.paused = False
        self.start_time = time.time()
        
        # Callbacks
        self.on_reset: Optional[Callable] = None
        self.on_disturbance: Optional[Callable] = None
        
        # Disturbance state
        self.disturbance_active = False
        self.disturbance_torque = np.zeros(3)
        self.disturbance_force = np.zeros(3)
        self.disturbance_duration = 0.2  # seconds
        self.disturbance_start_time = 0.0
        
        # Anti-windup state (disabled by default)
        self.use_anti_windup = False
        
        # Trail points for trajectory
        self.trail_points: deque = deque(maxlen=200)
        
        # Initialize Viser server
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        
        # Setup visualization
        self._setup_scene()
        self._setup_gui()
    
    def _setup_scene(self):
        """Setup the 3D scene with ground grid and drone."""
        # Ground grid
        self._create_ground_grid()
        
        # World frame axes
        self._create_world_frame()
        
        # Drone frame (will be updated each step)
        self._create_drone()
        
        # Trajectory trail
        self.trail_handle = None
        
        # Setpoint marker
        self._create_setpoint_marker()
    
    def _create_ground_grid(self):
        """Create a ground grid for spatial reference."""
        grid_size = 5.0
        grid_divisions = 20
        
        # Create grid lines
        points = []
        for i in range(grid_divisions + 1):
            t = -grid_size + (2 * grid_size * i / grid_divisions)
            # Lines parallel to X
            points.append([t, -grid_size, 0])
            points.append([t, grid_size, 0])
            # Lines parallel to Y
            points.append([-grid_size, t, 0])
            points.append([grid_size, t, 0])
        
        points = np.array(points)
        
        # Create line segments
        for i in range(0, len(points), 2):
            self.server.scene.add_spline_catmull_rom(
                f"/grid/line_{i}",
                positions=points[i:i+2],
                tension=0.0,
                color=(80, 80, 80),
                line_width=1.0,
            )
    
    def _create_world_frame(self):
        """Create world frame coordinate axes."""
        axis_length = 0.5
        axis_radius = 0.01
        
        # X axis - Red
        self.server.scene.add_frame(
            "/world_frame",
            axes_length=axis_length,
            axes_radius=axis_radius,
        )
    
    def _create_drone(self):
        """Create the quadrotor drone mesh."""
        arm_length = self.params.arm_length
        arm_width = 0.02
        body_size = 0.08
        rotor_radius = 0.08
        
        # Drone base frame
        self.drone_frame = self.server.scene.add_frame(
            "/drone",
            axes_length=0.2,
            axes_radius=0.005,
        )
        
        # Central body (box)
        self.server.scene.add_box(
            "/drone/body",
            dimensions=(body_size, body_size, body_size * 0.5),
            color=(60, 60, 70),
        )
        
        # Arms and rotors
        arm_positions = [
            (arm_length, 0, 0),    # Front (red)
            (-arm_length, 0, 0),   # Back (blue)
            (0, arm_length, 0),    # Right
            (0, -arm_length, 0),   # Left
        ]
        
        arm_colors = [
            (220, 60, 60),   # Front - Red
            (60, 60, 220),   # Back - Blue
            (100, 100, 100), # Right - Gray
            (100, 100, 100), # Left - Gray
        ]
        
        for i, (pos, color) in enumerate(zip(arm_positions, arm_colors)):
            # Arm
            arm_center = (pos[0] / 2, pos[1] / 2, 0)
            if pos[0] != 0:
                arm_dims = (arm_length, arm_width, arm_width)
            else:
                arm_dims = (arm_width, arm_length, arm_width)
            
            self.server.scene.add_box(
                f"/drone/arm_{i}",
                dimensions=arm_dims,
                position=arm_center,
                color=(80, 80, 90),
            )
            
            # Rotor disk
            self.server.scene.add_mesh_simple(
                f"/drone/rotor_{i}",
                vertices=self._create_disk_vertices(rotor_radius, 16),
                faces=self._create_disk_faces(16),
                position=pos,
                color=color,
            )
    
    def _create_disk_vertices(self, radius: float, segments: int) -> np.ndarray:
        """Create vertices for a disk mesh."""
        vertices = [[0, 0, 0]]  # Center
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            vertices.append([radius * np.cos(angle), radius * np.sin(angle), 0])
        return np.array(vertices, dtype=np.float32)
    
    def _create_disk_faces(self, segments: int) -> np.ndarray:
        """Create faces for a disk mesh."""
        faces = []
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([0, i + 1, next_i + 1])
        return np.array(faces, dtype=np.uint32)
    
    def _create_setpoint_marker(self):
        """Create a marker showing the position setpoint."""
        self.setpoint_marker = self.server.scene.add_icosphere(
            "/setpoint",
            radius=0.05,
            color=(50, 200, 50),
        )
    
    def _setup_gui(self):
        """Setup the GUI controls."""
        
        # === Control Panel ===
        with self.server.gui.add_folder("Simulation"):
            self.pause_button = self.server.gui.add_button("Pause")
            self.reset_button = self.server.gui.add_button("Reset")
            self.clear_plots_button = self.server.gui.add_button("Clear Plots")
            
            @self.pause_button.on_click
            def _(_):
                self.paused = not self.paused
                self.pause_button.label = "Resume" if self.paused else "Pause"
            
            @self.reset_button.on_click
            def _(_):
                if self.on_reset:
                    self.on_reset()
                self.trail_points.clear()
                self.plot_data.clear()
                self.start_time = time.time()
            
            @self.clear_plots_button.on_click
            def _(_):
                self.plot_data.clear()
                self.start_time = time.time()
            
            self.reset_setpoints_button = self.server.gui.add_button("Reset Setpoints")
            self.reset_gains_button = self.server.gui.add_button("Reset PID Gains")
            
            @self.reset_setpoints_button.on_click
            def _(_):
                self._reset_setpoints()
            
            @self.reset_gains_button.on_click
            def _(_):
                self._reset_gains()
        
        # === Disturbance Controls ===
        with self.server.gui.add_folder("Disturbance"):
            self.disturbance_button = self.server.gui.add_button("Apply Disturbance")
            
            self.disturbance_type = self.server.gui.add_dropdown(
                "Type",
                options=["Torque", "Force"],
                initial_value="Torque",
            )
            
            self.disturbance_magnitude_slider = self.server.gui.add_slider(
                "Magnitude", min=-10.0, max=10.0, step=0.1, initial_value=1.0
            )
            self.disturbance_duration_slider = self.server.gui.add_slider(
                "Duration [s]", min=0.05, max=1.0, step=0.05, initial_value=0.2
            )
            self.disturbance_axis = self.server.gui.add_dropdown(
                "Axis",
                options=["X (Roll/Forward)", "Y (Pitch/Right)", "Z (Yaw/Up)", "Random"],
                initial_value="X (Roll/Forward)",
            )
            
            @self.disturbance_button.on_click
            def _(_):
                self._trigger_disturbance()
            
            @self.disturbance_duration_slider.on_update
            def _(_):
                self.disturbance_duration = self.disturbance_duration_slider.value
        
        # === Setpoint Controls ===
        with self.server.gui.add_folder("Setpoints"):
            self.mode_checkbox = self.server.gui.add_checkbox(
                "Position Hold Mode",
                initial_value=False,
            )
            
            self.gravity_comp_checkbox = self.server.gui.add_checkbox(
                "Gravity Compensation",
                initial_value=True,
            )
            
            @self.mode_checkbox.on_update
            def _(_):
                self.setpoints.use_position_control = self.mode_checkbox.value
            
            @self.gravity_comp_checkbox.on_update
            def _(_):
                self.setpoints.use_gravity_compensation = self.gravity_comp_checkbox.value
            
            self.anti_windup_checkbox = self.server.gui.add_checkbox(
                "Anti-Windup (Integral Saturation)",
                initial_value=False,
            )
            
            @self.anti_windup_checkbox.on_update
            def _(_):
                self.use_anti_windup = self.anti_windup_checkbox.value
            
            # Linear velocity setpoints
            with self.server.gui.add_folder("Linear Velocity"):
                self.vel_x_slider = self.server.gui.add_slider(
                    "Vx [m/s]", min=-2.0, max=2.0, step=0.1, initial_value=0.0
                )
                self.vel_y_slider = self.server.gui.add_slider(
                    "Vy [m/s]", min=-2.0, max=2.0, step=0.1, initial_value=0.0
                )
                self.vel_z_slider = self.server.gui.add_slider(
                    "Vz [m/s]", min=-2.0, max=2.0, step=0.1, initial_value=0.0
                )
                
                @self.vel_x_slider.on_update
                def _(_): self.setpoints.velocity[0] = self.vel_x_slider.value
                @self.vel_y_slider.on_update
                def _(_): self.setpoints.velocity[1] = self.vel_y_slider.value
                @self.vel_z_slider.on_update
                def _(_): self.setpoints.velocity[2] = self.vel_z_slider.value
            
            # Angular velocity setpoints
            with self.server.gui.add_folder("Angular Velocity"):
                self.omega_x_slider = self.server.gui.add_slider(
                    "Roll Rate [rad/s]", min=-2.0, max=2.0, step=0.1, initial_value=0.0
                )
                self.omega_y_slider = self.server.gui.add_slider(
                    "Pitch Rate [rad/s]", min=-2.0, max=2.0, step=0.1, initial_value=0.0
                )
                self.omega_z_slider = self.server.gui.add_slider(
                    "Yaw Rate [rad/s]", min=-2.0, max=2.0, step=0.1, initial_value=0.0
                )
                
                @self.omega_x_slider.on_update
                def _(_): self.setpoints.angular_velocity[0] = self.omega_x_slider.value
                @self.omega_y_slider.on_update
                def _(_): self.setpoints.angular_velocity[1] = self.omega_y_slider.value
                @self.omega_z_slider.on_update
                def _(_): self.setpoints.angular_velocity[2] = self.omega_z_slider.value
            
            # Position setpoints
            with self.server.gui.add_folder("Position (Hold Mode)"):
                self.pos_x_slider = self.server.gui.add_slider(
                    "X [m]", min=-3.0, max=3.0, step=0.1, initial_value=0.0
                )
                self.pos_y_slider = self.server.gui.add_slider(
                    "Y [m]", min=-3.0, max=3.0, step=0.1, initial_value=0.0
                )
                self.pos_z_slider = self.server.gui.add_slider(
                    "Z [m]", min=0.0, max=3.0, step=0.1, initial_value=1.0
                )
                self.yaw_slider = self.server.gui.add_slider(
                    "Yaw [deg]", min=-180.0, max=180.0, step=5.0, initial_value=0.0
                )
                
                @self.pos_x_slider.on_update
                def _(_): self.setpoints.position[0] = self.pos_x_slider.value
                @self.pos_y_slider.on_update
                def _(_): self.setpoints.position[1] = self.pos_y_slider.value
                @self.pos_z_slider.on_update
                def _(_): self.setpoints.position[2] = self.pos_z_slider.value
                @self.yaw_slider.on_update
                def _(_): self.setpoints.yaw = np.radians(self.yaw_slider.value)
        
        # === PID Gain Sliders ===
        self._setup_pid_sliders()
    
    def _setup_pid_sliders(self):
        """Setup all PID gain sliders organized by controller."""
        
        # Rate Controller (Inner Loop)
        # These gains operate on angular velocity errors (rad/s) -> torque output
        # Typical values: Kp=0.1-0.3, Ki=0.1-0.5, Kd=0.001-0.01 (based on PX4)
        with self.server.gui.add_folder("Rate PID (Inner)"):
            self._create_pid_folder("Roll Rate", "rate_roll", 
                                   kp_range=(0, 1.0), ki_range=(0, 1.0), kd_range=(0, 0.05),
                                   kp_step=0.01, ki_step=0.01, kd_step=0.001)
            self._create_pid_folder("Pitch Rate", "rate_pitch",
                                   kp_range=(0, 1.0), ki_range=(0, 1.0), kd_range=(0, 0.05),
                                   kp_step=0.01, ki_step=0.01, kd_step=0.001)
            self._create_pid_folder("Yaw Rate", "rate_yaw",
                                   kp_range=(0, 0.5), ki_range=(0, 0.5), kd_range=(0, 0.02),
                                   kp_step=0.01, ki_step=0.01, kd_step=0.001)
        
        # Attitude Controller (Middle Loop)
        # These gains operate on angle errors (rad) -> angular rate output
        # Typical values: Kp=4-10, Ki=0, Kd=0 (rate controller handles damping)
        with self.server.gui.add_folder("Attitude PID"):
            self._create_pid_folder("Roll", "att_roll",
                                   kp_range=(0, 15), ki_range=(0, 2), kd_range=(0, 1),
                                   kp_step=0.5, ki_step=0.1, kd_step=0.1)
            self._create_pid_folder("Pitch", "att_pitch",
                                   kp_range=(0, 15), ki_range=(0, 2), kd_range=(0, 1),
                                   kp_step=0.5, ki_step=0.1, kd_step=0.1)
            self._create_pid_folder("Yaw", "att_yaw",
                                   kp_range=(0, 10), ki_range=(0, 2), kd_range=(0, 1),
                                   kp_step=0.5, ki_step=0.1, kd_step=0.1)
        
        # Velocity Controller
        # Operates on velocity errors (m/s) -> acceleration/attitude output
        with self.server.gui.add_folder("Velocity PID"):
            self._create_pid_folder("Vel X", "vel_x",
                                   kp_range=(0, 10), ki_range=(0, 2), kd_range=(0, 1),
                                   kp_step=0.1, ki_step=0.05, kd_step=0.05)
            self._create_pid_folder("Vel Y", "vel_y",
                                   kp_range=(0, 10), ki_range=(0, 2), kd_range=(0, 1),
                                   kp_step=0.1, ki_step=0.05, kd_step=0.05)
            self._create_pid_folder("Vel Z", "vel_z",
                                   kp_range=(0, 15), ki_range=(0, 3), kd_range=(0, 2),
                                   kp_step=0.1, ki_step=0.1, kd_step=0.1)
        
        # Position Controller (Outer Loop)
        # Operates on position errors (m) -> velocity output
        with self.server.gui.add_folder("Position PID (Outer)"):
            self._create_pid_folder("Pos X", "pos_x",
                                   kp_range=(0, 5), ki_range=(0, 1), kd_range=(0, 1),
                                   kp_step=0.1, ki_step=0.05, kd_step=0.05)
            self._create_pid_folder("Pos Y", "pos_y",
                                   kp_range=(0, 5), ki_range=(0, 1), kd_range=(0, 1),
                                   kp_step=0.1, ki_step=0.05, kd_step=0.05)
            self._create_pid_folder("Pos Z", "pos_z",
                                   kp_range=(0, 10), ki_range=(0, 2), kd_range=(0, 2),
                                   kp_step=0.1, ki_step=0.1, kd_step=0.1)
    
    def _create_pid_folder(
        self,
        name: str,
        gains_key: str,
        kp_range: tuple = (0, 10),
        ki_range: tuple = (0, 5),
        kd_range: tuple = (0, 2),
        kp_step: float = 0.1,
        ki_step: float = 0.05,
        kd_step: float = 0.05,
    ):
        """
        Create a folder with Kp, Ki, Kd number inputs for a PID controller.
        
        Uses number inputs instead of sliders to allow values outside the typical range.
        The ranges are used as hints (visible in the UI) but not enforced.
        """
        gains: PIDGains = getattr(self.gains, gains_key)
        
        with self.server.gui.add_folder(name):
            # Use number inputs to allow arbitrary values
            # The step parameter controls precision, no min/max clamping
            kp_input = self.server.gui.add_number(
                f"Kp [{kp_range[0]}–{kp_range[1]}]", 
                initial_value=gains.kp,
                step=kp_step,
            )
            ki_input = self.server.gui.add_number(
                f"Ki [{ki_range[0]}–{ki_range[1]}]",
                initial_value=gains.ki,
                step=ki_step,
            )
            kd_input = self.server.gui.add_number(
                f"Kd [{kd_range[0]}–{kd_range[1]}]",
                initial_value=gains.kd,
                step=kd_step,
            )
            
            # Store input references on self so they persist and callbacks work
            # We store them with unique attribute names
            kp_attr = f'pid_kp_{gains_key}'
            ki_attr = f'pid_ki_{gains_key}'
            kd_attr = f'pid_kd_{gains_key}'
            setattr(self, kp_attr, kp_input)
            setattr(self, ki_attr, ki_input)
            setattr(self, kd_attr, kd_input)
            
            # Also keep dict for reset functionality
            setattr(self, f'pid_sliders_{gains_key}', {
                'kp': kp_input,
                'ki': ki_input,
                'kd': kd_input,
            })
            
            # Register callbacks - these update gains when inputs change
            @kp_input.on_update
            def _(event, key=gains_key, attr=kp_attr):
                input_field = getattr(self, attr)
                getattr(self.gains, key).kp = input_field.value
            
            @ki_input.on_update
            def _(event, key=gains_key, attr=ki_attr):
                input_field = getattr(self, attr)
                getattr(self.gains, key).ki = input_field.value
            
            @kd_input.on_update
            def _(event, key=gains_key, attr=kd_attr):
                input_field = getattr(self, attr)
                getattr(self.gains, key).kd = input_field.value
    
    def update(
        self,
        state: DroneState,
        thrust: float = 0.0,
        torques: np.ndarray = None,
    ):
        """
        Update the visualization with current state.
        
        Args:
            state: Current drone state
            thrust: Current thrust command
            torques: Current torque commands
        """
        if torques is None:
            torques = np.zeros(3)
        
        # Update drone position and orientation
        position = state.position
        quat = state.quaternion  # (w, x, y, z)
        
        # Viser uses (w, x, y, z) for quaternions
        self.drone_frame.position = position
        self.drone_frame.wxyz = quat
        
        # Update trail
        self.trail_points.append(position.copy())
        if len(self.trail_points) >= 2:
            trail_array = np.array(list(self.trail_points))
            if self.trail_handle is not None:
                self.trail_handle.remove()
            self.trail_handle = self.server.scene.add_spline_catmull_rom(
                "/trail",
                positions=trail_array,
                tension=0.5,
                color=(100, 180, 255),
                line_width=2.0,
            )
        
        # Update setpoint marker
        if self.setpoints.use_position_control:
            self.setpoint_marker.position = self.setpoints.position
        else:
            # In velocity mode, show marker at a projected position
            self.setpoint_marker.position = position + self.setpoints.velocity * 0.5
        
        # Log data for plots
        current_time = time.time() - self.start_time
        euler = state.get_euler_angles()
        
        self.plot_data.time.append(current_time)
        
        # Orientation (degrees)
        self.plot_data.roll.append(np.degrees(euler[0]))
        self.plot_data.pitch.append(np.degrees(euler[1]))
        self.plot_data.yaw.append(np.degrees(euler[2]))
        
        self.plot_data.roll_sp.append(0.0)  # Will be updated with actual setpoints
        self.plot_data.pitch_sp.append(0.0)
        self.plot_data.yaw_sp.append(np.degrees(self.setpoints.yaw))
        
        # Angular velocity
        self.plot_data.p.append(state.angular_velocity[0])
        self.plot_data.q.append(state.angular_velocity[1])
        self.plot_data.r.append(state.angular_velocity[2])
        
        # Position
        self.plot_data.x.append(position[0])
        self.plot_data.y.append(position[1])
        self.plot_data.z.append(position[2])
        
        # Velocity
        self.plot_data.vx.append(state.velocity[0])
        self.plot_data.vy.append(state.velocity[1])
        self.plot_data.vz.append(state.velocity[2])
        
        self.plot_data.vx_sp.append(self.setpoints.velocity[0])
        self.plot_data.vy_sp.append(self.setpoints.velocity[1])
        self.plot_data.vz_sp.append(self.setpoints.velocity[2])
        
        # Control outputs
        self.plot_data.thrust.append(thrust)
        self.plot_data.tau_x.append(torques[0])
        self.plot_data.tau_y.append(torques[1])
        self.plot_data.tau_z.append(torques[2])
    
    def get_gains(self) -> ControllerGains:
        """Get current controller gains by reading slider values directly."""
        # Sync gains from sliders (callbacks may not fire reliably)
        for gains_key in ['rate_roll', 'rate_pitch', 'rate_yaw',
                          'att_roll', 'att_pitch', 'att_yaw',
                          'vel_x', 'vel_y', 'vel_z',
                          'pos_x', 'pos_y', 'pos_z']:
            sliders = getattr(self, f'pid_sliders_{gains_key}', None)
            if sliders:
                gains_obj = getattr(self.gains, gains_key)
                gains_obj.kp = sliders['kp'].value
                gains_obj.ki = sliders['ki'].value
                gains_obj.kd = sliders['kd'].value
        return self.gains
    
    def get_setpoints(self) -> Setpoints:
        """Get current setpoints."""
        return self.setpoints
    
    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return self.paused
    
    def _trigger_disturbance(self):
        """Trigger a disturbance impulse (torque or force)."""
        magnitude = self.disturbance_magnitude_slider.value
        axis = self.disturbance_axis.value
        dist_type = self.disturbance_type.value
        
        # Determine disturbance direction
        if axis == "X (Roll/Forward)":
            direction = np.array([1.0, 0.0, 0.0])
        elif axis == "Y (Pitch/Right)":
            direction = np.array([0.0, 1.0, 0.0])
        elif axis == "Z (Yaw/Up)":
            direction = np.array([0.0, 0.0, 1.0])
        else:  # Random
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
        
        # Apply disturbance based on type (magnitude can be positive or negative)
        if dist_type == "Torque":
            self.disturbance_torque = magnitude * direction
            self.disturbance_force = np.zeros(3)
            print(f"Torque disturbance applied: {self.disturbance_torque} N⋅m for {self.disturbance_duration}s")
        else:  # Force
            self.disturbance_force = magnitude * direction
            self.disturbance_torque = np.zeros(3)
            print(f"Force disturbance applied: {self.disturbance_force} N for {self.disturbance_duration}s")
        
        self.disturbance_active = True
        self.disturbance_start_time = time.time()
    
    def _reset_setpoints(self):
        """Reset all velocity setpoints to zero."""
        # Reset internal state
        self.setpoints.velocity = np.zeros(3)
        self.setpoints.angular_velocity = np.zeros(3)
        
        # Update sliders
        self.vel_x_slider.value = 0.0
        self.vel_y_slider.value = 0.0
        self.vel_z_slider.value = 0.0
        self.omega_x_slider.value = 0.0
        self.omega_y_slider.value = 0.0
        self.omega_z_slider.value = 0.0
        
        print("Setpoints reset to zero")
    
    def _reset_gains(self):
        """Reset all PID gains to default values."""
        # Create fresh default gains
        default_gains = ControllerGains()
        self.gains = default_gains
        
        # Update all PID sliders to default values
        for gains_key in ['rate_roll', 'rate_pitch', 'rate_yaw',
                          'att_roll', 'att_pitch', 'att_yaw',
                          'vel_x', 'vel_y', 'vel_z',
                          'pos_x', 'pos_y', 'pos_z']:
            if hasattr(self, f'pid_sliders_{gains_key}'):
                sliders = getattr(self, f'pid_sliders_{gains_key}')
                gains = getattr(default_gains, gains_key)
                sliders['kp'].value = gains.kp
                sliders['ki'].value = gains.ki
                sliders['kd'].value = gains.kd
        
        print("PID gains reset to defaults")
    
    def get_disturbance_torque(self) -> np.ndarray:
        """Get current disturbance torque (returns zero if not active)."""
        if not self.disturbance_active:
            return np.zeros(3)
        
        # Check if disturbance has expired
        elapsed = time.time() - self.disturbance_start_time
        if elapsed > self.disturbance_duration:
            self.disturbance_active = False
            self.disturbance_torque = np.zeros(3)
            self.disturbance_force = np.zeros(3)
            return np.zeros(3)
        
        return self.disturbance_torque
    
    def get_disturbance_force(self) -> np.ndarray:
        """Get current disturbance force (returns zero if not active)."""
        if not self.disturbance_active:
            return np.zeros(3)
        
        # Check if disturbance has expired
        elapsed = time.time() - self.disturbance_start_time
        if elapsed > self.disturbance_duration:
            self.disturbance_active = False
            self.disturbance_torque = np.zeros(3)
            self.disturbance_force = np.zeros(3)
            return np.zeros(3)
        
        return self.disturbance_force
    
    def get_anti_windup_enabled(self) -> bool:
        """Get whether anti-windup is enabled."""
        return self.use_anti_windup

