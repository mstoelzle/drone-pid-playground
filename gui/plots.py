"""Real-time plotting for the drone simulator using Viser and Plotly."""

import numpy as np
from typing import Dict, List, Optional
from collections import deque

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not available, plots disabled")


class PlotManager:
    """
    Manages real-time plots for the drone simulator.
    
    Uses Plotly to generate interactive plots displayed in Viser.
    """
    
    def __init__(self, server, plot_data):
        """
        Initialize the plot manager.
        
        Args:
            server: Viser server instance
            plot_data: PlotData instance with time series data
        """
        self.server = server
        self.plot_data = plot_data
        self.plot_handles = {}
        
        if not HAS_PLOTLY:
            return
        
        # Create plot GUI elements
        self._setup_plots()
    
    def _create_empty_figure(self, title: str, y_label: str = "") -> go.Figure:
        """Create an empty Plotly figure with dark theme."""
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=title, font=dict(size=12, color="#ccc")),
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font=dict(color="#ccc", size=10),
            xaxis=dict(
                title="Time [s]",
                gridcolor="#2a2a4e",
                zerolinecolor="#3a3a5e",
            ),
            yaxis=dict(
                title=y_label,
                gridcolor="#2a2a4e",
                zerolinecolor="#3a3a5e",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=9),
            ),
            margin=dict(l=50, r=20, t=40, b=40),
            height=180,
        )
        return fig
    
    def _setup_plots(self):
        """Setup the plot display in Viser GUI."""
        with self.server.gui.add_folder("Plots"):
            # Orientation plot
            self.orientation_fig = self._create_empty_figure("Orientation", "Degrees")
            self.orientation_plot = self.server.gui.add_plotly(self.orientation_fig)
            
            # Angular velocity plot
            self.angular_vel_fig = self._create_empty_figure("Angular Velocity", "rad/s")
            self.angular_vel_plot = self.server.gui.add_plotly(self.angular_vel_fig)
            
            # Velocity plot
            self.velocity_fig = self._create_empty_figure("Velocity", "m/s")
            self.velocity_plot = self.server.gui.add_plotly(self.velocity_fig)
            
            # Position plot
            self.position_fig = self._create_empty_figure("Position", "m")
            self.position_plot = self.server.gui.add_plotly(self.position_fig)
    
    def update(self):
        """Update all plots with current data."""
        if not HAS_PLOTLY:
            return
        
        if len(self.plot_data.time) < 2:
            return
        
        time_array = list(self.plot_data.time)
        
        # Update orientation plot
        self.orientation_fig.data = []
        self.orientation_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.roll),
            name="Roll", mode="lines", line=dict(color="#ff6b6b", width=2)
        ))
        self.orientation_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.pitch),
            name="Pitch", mode="lines", line=dict(color="#4ecdc4", width=2)
        ))
        self.orientation_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.yaw),
            name="Yaw", mode="lines", line=dict(color="#ffe66d", width=2)
        ))
        self.orientation_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.yaw_sp),
            name="Yaw SP", mode="lines", line=dict(color="#ffe66d", width=1, dash="dash")
        ))
        self.orientation_plot.figure = self.orientation_fig
        
        # Update angular velocity plot
        self.angular_vel_fig.data = []
        self.angular_vel_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.p),
            name="p (roll)", mode="lines", line=dict(color="#ff6b6b", width=2)
        ))
        self.angular_vel_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.q),
            name="q (pitch)", mode="lines", line=dict(color="#4ecdc4", width=2)
        ))
        self.angular_vel_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.r),
            name="r (yaw)", mode="lines", line=dict(color="#ffe66d", width=2)
        ))
        self.angular_vel_plot.figure = self.angular_vel_fig
        
        # Update velocity plot
        self.velocity_fig.data = []
        self.velocity_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.vx),
            name="Vx", mode="lines", line=dict(color="#ff6b6b", width=2)
        ))
        self.velocity_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.vy),
            name="Vy", mode="lines", line=dict(color="#4ecdc4", width=2)
        ))
        self.velocity_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.vz),
            name="Vz", mode="lines", line=dict(color="#ffe66d", width=2)
        ))
        self.velocity_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.vx_sp),
            name="Vx SP", mode="lines", line=dict(color="#ff6b6b", width=1, dash="dash")
        ))
        self.velocity_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.vy_sp),
            name="Vy SP", mode="lines", line=dict(color="#4ecdc4", width=1, dash="dash")
        ))
        self.velocity_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.vz_sp),
            name="Vz SP", mode="lines", line=dict(color="#ffe66d", width=1, dash="dash")
        ))
        self.velocity_plot.figure = self.velocity_fig
        
        # Update position plot
        self.position_fig.data = []
        self.position_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.x),
            name="X", mode="lines", line=dict(color="#ff6b6b", width=2)
        ))
        self.position_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.y),
            name="Y", mode="lines", line=dict(color="#4ecdc4", width=2)
        ))
        self.position_fig.add_trace(go.Scatter(
            x=time_array, y=list(self.plot_data.z),
            name="Z", mode="lines", line=dict(color="#ffe66d", width=2)
        ))
        self.position_plot.figure = self.position_fig
