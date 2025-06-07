"""
TorchSlim Visualization Tools
Comprehensive visualization utilities for compression analysis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')

# Import visualization libraries with fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

class VisualizationTools:
    """Comprehensive visualization utilities for compression analysis"""
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8)):
        self.style = style
        self.figsize = figsize
        self.colors = self._setup_color_palette()
        
        if MATPLOTLIB_AVAILABLE:
            plt.style.use(style if style in plt.style.available else 'default')
            sns.set_palette("husl")
    
    def _setup_color_palette(self) -> Dict[str, str]:
        """Setup color palette for consistent visualization"""
        return {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#6A994E',
            'light': '#F5F5F5',
            'dark': '#2D3748',
            'compression': '#FF6B6B',
            'speed': '#4ECDC4',
            'accuracy': '#45B7D1',
            'memory': '#96CEB4'
        }
    
    def plot_compression_comparison(
        self,
        results: Dict[str, Any],
        metrics: List[str] = ['compression_ratio', 'inference_speedup', 'accuracy_preservation'],
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> Union[plt.Figure, go.Figure]:
        """
        Create a comprehensive comparison plot of compression methods
        
        Args:
            results: Dictionary mapping method names to results
            metrics: List of metrics to compare
            save_path: Optional path to save the plot
            interactive: Whether to create interactive plot (requires plotly)
            
        Returns:
            Matplotlib or Plotly figure object
        """
        
        if not results:
            raise ValueError("No results provided for visualization")
        
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_interactive_comparison(results, metrics, save_path)
        elif MATPLOTLIB_AVAILABLE:
            return self._plot_static_comparison(results, metrics, save_path)
        else:
            raise ImportError("No visualization library available")
    
    def _plot_static_comparison(
        self,
        results: Dict[str, Any],
        metrics: List[str],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create static comparison plot using matplotlib"""
        
        methods = list(results.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = []
            colors = []
            
            for method in methods:
                if hasattr(results[method], metric):
                    values.append(getattr(results[method], metric))
                    colors.append(self.colors.get(metric, self.colors['primary']))
                else:
                    values.append(0)
                    colors.append(self.colors['light'])
            
            # Create bar plot
            bars = axes[i].bar(methods, values, color=colors, alpha=0.7, edgecolor='black')
            
            # Customize plot
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Value', fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Static comparison plot saved to {save_path}")
        
        return fig
    
    def _plot_interactive_comparison(
        self,
        results: Dict[str, Any],
        metrics: List[str],
        save_path: Optional[str]
    ) -> go.Figure:
        """Create interactive comparison plot using plotly"""
        
        methods = list(results.keys())
        
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=[metric.replace('_', ' ').title() for metric in metrics]
        )
        
        for i, metric in enumerate(metrics, 1):
            values = []
            for method in methods:
                if hasattr(results[method], metric):
                    values.append(getattr(results[method], metric))
                else:
                    values.append(0)
            
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=values,
                    name=metric.replace('_', ' ').title(),
                    marker_color=self.colors.get(metric, self.colors['primary']),
                    text=[f'{v:.2f}' for v in values],
                    textposition='auto',
                ),
                row=1, col=i
            )
        
        fig.update_layout(
            title="Compression Methods Comparison",
            showlegend=False,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive comparison plot saved to {save_path}")
        
        return fig
    
    def plot_compression_scatter(
        self,
        results: Dict[str, Any],
        x_metric: str = 'compression_ratio',
        y_metric: str = 'inference_speedup',
        size_metric: str = 'accuracy_preservation',
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> Union[plt.Figure, go.Figure]:
        """
        Create scatter plot showing relationship between compression metrics
        
        Args:
            results: Dictionary mapping method names to results
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis  
            size_metric: Metric for point size
            save_path: Optional path to save the plot
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure object
        """
        
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_interactive_scatter(results, x_metric, y_metric, size_metric, save_path)
        elif MATPLOTLIB_AVAILABLE:
            return self._plot_static_scatter(results, x_metric, y_metric, size_metric, save_path)
        else:
            raise ImportError("No visualization library available")
    
    def _plot_static_scatter(
        self,
        results: Dict[str, Any],
        x_metric: str,
        y_metric: str,
        size_metric: str,
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create static scatter plot using matplotlib"""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        methods = list(results.keys())
        x_values = []
        y_values = []
        sizes = []
        
        for method in methods:
            result = results[method]
            x_values.append(getattr(result, x_metric, 0))
            y_values.append(getattr(result, y_metric, 0))
            sizes.append(getattr(result, size_metric, 0.5) * 200)  # Scale for visibility
        
        # Create scatter plot
        scatter = ax.scatter(
            x_values, y_values, s=sizes,
            c=range(len(methods)), cmap='viridis',
            alpha=0.7, edgecolors='black', linewidth=1
        )
        
        # Add method labels
        for i, method in enumerate(methods):
            ax.annotate(
                method, (x_values[i], y_values[i]),
                xytext=(5, 5), textcoords='offset points',
                fontweight='bold', fontsize=10
            )
        
        # Customize plot
        ax.set_xlabel(x_metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(y_metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'{y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Method Index', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Static scatter plot saved to {save_path}")
        
        return fig
    
    def _plot_interactive_scatter(
        self,
        results: Dict[str, Any],
        x_metric: str,
        y_metric: str,
        size_metric: str,
        save_path: Optional[str]
    ) -> go.Figure:
        """Create interactive scatter plot using plotly"""
        
        methods = list(results.keys())
        x_values = [getattr(results[method], x_metric, 0) for method in methods]
        y_values = [getattr(results[method], y_metric, 0) for method in methods]
        size_values = [getattr(results[method], size_metric, 0.5) * 20 for method in methods]
        
        fig = go.Figure(data=go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers+text',
            marker=dict(
                size=size_values,
                color=range(len(methods)),
                colorscale='viridis',
                line=dict(width=2, color='black'),
                opacity=0.7
            ),
            text=methods,
            textposition="top center",
            hovertemplate=(
                f"<b>%{{text}}</b><br>"
                f"{x_metric}: %{{x:.2f}}<br>"
                f"{y_metric}: %{{y:.2f}}<br>"
                f"{size_metric}: %{{marker.size:.2f}}<br>"
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title=f'{y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}',
            xaxis_title=x_metric.replace('_', ' ').title(),
            yaxis_title=y_metric.replace('_', ' ').title(),
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive scatter plot saved to {save_path}")
        
        return fig
    
    def plot_layer_compression_breakdown(
        self,
        layer_info: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot layer-by-layer compression breakdown
        
        Args:
            layer_info: Dictionary with layer compression information
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for layer breakdown plot")
        
        layer_names = list(layer_info.keys())
        compression_ratios = [info.get('compression_ratio', 1.0) for info in layer_info.values()]
        layer_types = [info.get('layer_type', 'Unknown') for info in layer_info.values()]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Compression ratios by layer
        colors = [self.colors['compression'] if ratio > 1.0 else self.colors['light'] 
                 for ratio in compression_ratios]
        
        bars = ax1.bar(range(len(layer_names)), compression_ratios, color=colors, alpha=0.7)
        ax1.set_xlabel('Layer Index', fontweight='bold')
        ax1.set_ylabel('Compression Ratio', fontweight='bold')
        ax1.set_title('Compression Ratio by Layer', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, ratio) in enumerate(zip(bars, compression_ratios)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{ratio:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Layer type distribution
        unique_types = list(set(layer_types))
        type_counts = [layer_types.count(t) for t in unique_types]
        
        pie_colors = [self.colors[list(self.colors.keys())[i % len(self.colors)]] 
                     for i in range(len(unique_types))]
        
        wedges, texts, autotexts = ax2.pie(
            type_counts, labels=unique_types, autopct='%1.1f%%',
            colors=pie_colors, startangle=90
        )
        ax2.set_title('Layer Type Distribution', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Layer breakdown plot saved to {save_path}")
        
        return fig
    
    def plot_compression_timeline(
        self,
        compression_history: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot compression progress over time
        
        Args:
            compression_history: List of compression snapshots over time
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for timeline plot")
        
        if not compression_history:
            raise ValueError("No compression history provided")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        timestamps = range(len(compression_history))
        
        # Plot 1: Compression ratio over time
        ratios = [step.get('compression_ratio', 1.0) for step in compression_history]
        axes[0].plot(timestamps, ratios, marker='o', linewidth=2, 
                    color=self.colors['compression'], markersize=6)
        axes[0].set_title('Compression Ratio Over Time', fontweight='bold')
        axes[0].set_ylabel('Compression Ratio')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Model size over time
        sizes = [step.get('model_size_mb', 0) for step in compression_history]
        axes[1].plot(timestamps, sizes, marker='s', linewidth=2,
                    color=self.colors['memory'], markersize=6)
        axes[1].set_title('Model Size Over Time', fontweight='bold')
        axes[1].set_ylabel('Size (MB)')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Inference speed over time
        speeds = [step.get('inference_speedup', 1.0) for step in compression_history]
        axes[2].plot(timestamps, speeds, marker='^', linewidth=2,
                    color=self.colors['speed'], markersize=6)
        axes[2].set_title('Inference Speedup Over Time', fontweight='bold')
        axes[2].set_ylabel('Speedup Factor')
        axes[2].set_xlabel('Compression Step')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Accuracy preservation over time
        accuracies = [step.get('accuracy_preservation', 1.0) for step in compression_history]
        axes[3].plot(timestamps, accuracies, marker='d', linewidth=2,
                    color=self.colors['accuracy'], markersize=6)
        axes[3].set_title('Accuracy Preservation Over Time', fontweight='bold')
        axes[3].set_ylabel('Accuracy Ratio')
        axes[3].set_xlabel('Compression Step')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Timeline plot saved to {save_path}")
        
        return fig
    
    def plot_weight_distribution(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        layer_name: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot weight distribution comparison between original and compressed models
        
        Args:
            original_model: Original model
            compressed_model: Compressed model
            layer_name: Specific layer to analyze (if None, analyzes all weights)
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for weight distribution plot")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract weights
        if layer_name:
            orig_weights = dict(original_model.named_parameters())[layer_name].detach().flatten()
            comp_weights = dict(compressed_model.named_parameters())[layer_name].detach().flatten()
        else:
            orig_weights = torch.cat([p.detach().flatten() for p in original_model.parameters()])
            comp_weights = torch.cat([p.detach().flatten() for p in compressed_model.parameters()])
        
        # Convert to numpy
        orig_weights = orig_weights.cpu().numpy()
        comp_weights = comp_weights.cpu().numpy()
        
        # Plot 1: Histograms
        axes[0, 0].hist(orig_weights, bins=50, alpha=0.7, label='Original', 
                       color=self.colors['primary'], density=True)
        axes[0, 0].hist(comp_weights, bins=50, alpha=0.7, label='Compressed',
                       color=self.colors['secondary'], density=True)
        axes[0, 0].set_title('Weight Distribution Comparison', fontweight='bold')
        axes[0, 0].set_xlabel('Weight Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Box plots
        data_to_plot = [orig_weights, comp_weights]
        box_plot = axes[0, 1].boxplot(data_to_plot, labels=['Original', 'Compressed'],
                                     patch_artist=True)
        box_plot['boxes'][0].set_facecolor(self.colors['primary'])
        box_plot['boxes'][1].set_facecolor(self.colors['secondary'])
        axes[0, 1].set_title('Weight Distribution Statistics', fontweight='bold')
        axes[0, 1].set_ylabel('Weight Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative distribution
        orig_sorted = np.sort(orig_weights)
        comp_sorted = np.sort(comp_weights)
        orig_cum = np.arange(1, len(orig_sorted) + 1) / len(orig_sorted)
        comp_cum = np.arange(1, len(comp_sorted) + 1) / len(comp_sorted)
        
        axes[1, 0].plot(orig_sorted, orig_cum, label='Original', 
                       color=self.colors['primary'], linewidth=2)
        axes[1, 0].plot(comp_sorted, comp_cum, label='Compressed',
                       color=self.colors['secondary'], linewidth=2)
        axes[1, 0].set_title('Cumulative Distribution Function', fontweight='bold')
        axes[1, 0].set_xlabel('Weight Value')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Scatter plot (original vs compressed)
        if len(orig_weights) == len(comp_weights):
            sample_size = min(10000, len(orig_weights))  # Sample for performance
            indices = np.random.choice(len(orig_weights), sample_size, replace=False)
            
            axes[1, 1].scatter(orig_weights[indices], comp_weights[indices],
                             alpha=0.6, s=1, color=self.colors['info'])
            
            # Add diagonal line
            min_val = min(orig_weights.min(), comp_weights.min())
            max_val = max(orig_weights.max(), comp_weights.max())
            axes[1, 1].plot([min_val, max_val], [min_val, max_val], 
                           'r--', linewidth=2, label='y=x')
            
            axes[1, 1].set_title('Original vs Compressed Weights', fontweight='bold')
            axes[1, 1].set_xlabel('Original Weight')
            axes[1, 1].set_ylabel('Compressed Weight')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Different number of weights\nCannot create scatter plot',
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 1].set_title('Weight Comparison Unavailable', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Weight distribution plot saved to {save_path}")
        
        return fig
    
    def plot_pareto_frontier(
        self,
        results: Dict[str, Any],
        x_metric: str = 'compression_ratio',
        y_metric: str = 'accuracy_preservation',
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot Pareto frontier for compression methods
        
        Args:
            results: Dictionary mapping method names to results
            x_metric: Metric for x-axis (to maximize)
            y_metric: Metric for y-axis (to maximize)
            save_path: Optional path to save the plot
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure object
        """
        
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_interactive_pareto(results, x_metric, y_metric, save_path)
        elif MATPLOTLIB_AVAILABLE:
            return self._plot_static_pareto(results, x_metric, y_metric, save_path)
        else:
            raise ImportError("No visualization library available")
    
    def _plot_static_pareto(
        self,
        results: Dict[str, Any],
        x_metric: str,
        y_metric: str,
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create static Pareto frontier plot using matplotlib"""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        methods = list(results.keys())
        x_values = [getattr(results[method], x_metric, 0) for method in methods]
        y_values = [getattr(results[method], y_metric, 0) for method in methods]
        
        # Find Pareto frontier
        pareto_points = self._find_pareto_frontier(x_values, y_values)
        pareto_x = [x_values[i] for i in pareto_points]
        pareto_y = [y_values[i] for i in pareto_points]
        
        # Sort Pareto points for line drawing
        pareto_sorted = sorted(zip(pareto_x, pareto_y))
        pareto_x_sorted, pareto_y_sorted = zip(*pareto_sorted)
        
        # Plot all points
        ax.scatter(x_values, y_values, c='lightblue', s=100, alpha=0.6, 
                  edgecolors='black', linewidth=1, label='All methods')
        
        # Highlight Pareto frontier
        ax.scatter(pareto_x, pareto_y, c=self.colors['warning'], s=150, 
                  edgecolors='black', linewidth=2, label='Pareto efficient', zorder=5)
        
        # Draw Pareto frontier line
        ax.plot(pareto_x_sorted, pareto_y_sorted, 'r--', linewidth=2, 
               alpha=0.7, label='Pareto frontier')
        
        # Add method labels
        for i, method in enumerate(methods):
            color = self.colors['warning'] if i in pareto_points else 'black'
            weight = 'bold' if i in pareto_points else 'normal'
            ax.annotate(method, (x_values[i], y_values[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontweight=weight, color=color, fontsize=10)
        
        ax.set_xlabel(x_metric.replace('_', ' ').title(), fontweight='bold', fontsize=12)
        ax.set_ylabel(y_metric.replace('_', ' ').title(), fontweight='bold', fontsize=12)
        ax.set_title(f'Pareto Frontier: {y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}',
                    fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Static Pareto frontier plot saved to {save_path}")
        
        return fig
    
    def _plot_interactive_pareto(
        self,
        results: Dict[str, Any],
        x_metric: str,
        y_metric: str,
        save_path: Optional[str]
    ) -> go.Figure:
        """Create interactive Pareto frontier plot using plotly"""
        
        methods = list(results.keys())
        x_values = [getattr(results[method], x_metric, 0) for method in methods]
        y_values = [getattr(results[method], y_metric, 0) for method in methods]
        
        # Find Pareto frontier
        pareto_points = self._find_pareto_frontier(x_values, y_values)
        
        # Create figure
        fig = go.Figure()
        
        # Add all points
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers+text',
            marker=dict(size=10, color='lightblue', line=dict(width=1, color='black')),
            text=methods,
            textposition="top center",
            name='All Methods',
            hovertemplate=f"<b>%{{text}}</b><br>{x_metric}: %{{x:.2f}}<br>{y_metric}: %{{y:.2f}}<extra></extra>"
        ))
        
        # Highlight Pareto frontier
        pareto_x = [x_values[i] for i in pareto_points]
        pareto_y = [y_values[i] for i in pareto_points]
        pareto_methods = [methods[i] for i in pareto_points]
        
        fig.add_trace(go.Scatter(
            x=pareto_x,
            y=pareto_y,
            mode='markers+text',
            marker=dict(size=15, color=self.colors['warning'], line=dict(width=2, color='black')),
            text=pareto_methods,
            textposition="top center",
            name='Pareto Efficient',
            hovertemplate=f"<b>%{{text}}</b><br>{x_metric}: %{{x:.2f}}<br>{y_metric}: %{{y:.2f}}<br><b>Pareto Efficient</b><extra></extra>"
        ))
        
        # Add Pareto frontier line
        pareto_sorted = sorted(zip(pareto_x, pareto_y))
        pareto_x_sorted, pareto_y_sorted = zip(*pareto_sorted)
        
        fig.add_trace(go.Scatter(
            x=pareto_x_sorted,
            y=pareto_y_sorted,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Pareto Frontier',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=f'Pareto Frontier: {y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}',
            xaxis_title=x_metric.replace('_', ' ').title(),
            yaxis_title=y_metric.replace('_', ' ').title(),
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive Pareto frontier plot saved to {save_path}")
        
        return fig
    
    def _find_pareto_frontier(self, x_values: List[float], y_values: List[float]) -> List[int]:
        """Find indices of points on the Pareto frontier (maximizing both x and y)"""
        
        pareto_points = []
        n = len(x_values)
        
        for i in range(n):
            is_pareto = True
            for j in range(n):
                if i != j:
                    # Check if point j dominates point i
                    if (x_values[j] >= x_values[i] and y_values[j] >= y_values[i] and
                        (x_values[j] > x_values[i] or y_values[j] > y_values[i])):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_points.append(i)
        
        return pareto_points
    
    def create_dashboard(
        self,
        results: Dict[str, Any],
        layer_info: Optional[Dict[str, Dict[str, Any]]] = None,
        save_path: Optional[str] = None
    ) -> Union[plt.Figure, go.Figure]:
        """
        Create a comprehensive dashboard with multiple visualizations
        
        Args:
            results: Dictionary mapping method names to results
            layer_info: Optional layer-wise compression information
            save_path: Optional path to save the dashboard
            
        Returns:
            Dashboard figure object
        """
        
        if PLOTLY_AVAILABLE:
            return self._create_interactive_dashboard(results, layer_info, save_path)
        elif MATPLOTLIB_AVAILABLE:
            return self._create_static_dashboard(results, layer_info, save_path)
        else:
            raise ImportError("No visualization library available")
    
    def _create_static_dashboard(
        self,
        results: Dict[str, Any],
        layer_info: Optional[Dict[str, Dict[str, Any]]],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create static dashboard using matplotlib"""
        
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Compression ratio comparison
        ax1 = fig.add_subplot(gs[0, 0])
        methods = list(results.keys())
        ratios = [getattr(results[method], 'compression_ratio', 1.0) for method in methods]
        ax1.bar(methods, ratios, color=self.colors['compression'], alpha=0.7)
        ax1.set_title('Compression Ratios', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Speed comparison
        ax2 = fig.add_subplot(gs[0, 1])
        speeds = [getattr(results[method], 'inference_speedup', 1.0) for method in methods]
        ax2.bar(methods, speeds, color=self.colors['speed'], alpha=0.7)
        ax2.set_title('Inference Speedup', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Accuracy preservation
        ax3 = fig.add_subplot(gs[0, 2])
        accuracies = [getattr(results[method], 'accuracy_preservation', 1.0) for method in methods]
        ax3.bar(methods, accuracies, color=self.colors['accuracy'], alpha=0.7)
        ax3.set_title('Accuracy Preservation', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Scatter plot (compression vs speed)
        ax4 = fig.add_subplot(gs[1, :2])
        scatter = ax4.scatter(ratios, speeds, s=100, c=accuracies, 
                             cmap='viridis', alpha=0.7, edgecolors='black')
        for i, method in enumerate(methods):
            ax4.annotate(method, (ratios[i], speeds[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax4.set_xlabel('Compression Ratio')
        ax4.set_ylabel('Inference Speedup')
        ax4.set_title('Compression vs Speed (colored by accuracy)', fontweight='bold')
        plt.colorbar(scatter, ax=ax4, label='Accuracy Preservation')
        
        # Plot 5: Layer breakdown (if available)
        if layer_info:
            ax5 = fig.add_subplot(gs[1, 2])
            layer_ratios = [info.get('compression_ratio', 1.0) for info in layer_info.values()]
            ax5.hist(layer_ratios, bins=10, color=self.colors['info'], alpha=0.7)
            ax5.set_xlabel('Compression Ratio')
            ax5.set_ylabel('Number of Layers')
            ax5.set_title('Layer Compression Distribution', fontweight='bold')
        
        # Plot 6: Summary statistics
        ax6 = fig.add_subplot(gs[2, :])
        summary_data = []
        for method in methods:
            result = results[method]
            summary_data.append([
                method,
                f"{getattr(result, 'compression_ratio', 1.0):.2f}x",
                f"{getattr(result, 'inference_speedup', 1.0):.2f}x",
                f"{getattr(result, 'accuracy_preservation', 1.0):.3f}",
                f"{getattr(result, 'memory_reduction_mb', 0):.1f} MB",
                f"{getattr(result, 'compression_time', 0):.2f}s"
            ])
        
        # Create table
        table = ax6.table(cellText=summary_data,
                         colLabels=['Method', 'Compression', 'Speedup', 'Accuracy', 'Memory Saved', 'Time'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax6.axis('off')
        ax6.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Static dashboard saved to {save_path}")
        
        return fig
    
    def _create_interactive_dashboard(
        self,
        results: Dict[str, Any],
        layer_info: Optional[Dict[str, Dict[str, Any]]],
        save_path: Optional[str]
    ) -> go.Figure:
        """Create interactive dashboard using plotly"""
        
        # This would create a comprehensive interactive dashboard
        # For brevity, creating a simplified version
        
        methods = list(results.keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Compression Ratios', 'Inference Speedup', 
                          'Accuracy Preservation', 'Method Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Add bar charts
        ratios = [getattr(results[method], 'compression_ratio', 1.0) for method in methods]
        speeds = [getattr(results[method], 'inference_speedup', 1.0) for method in methods]
        accuracies = [getattr(results[method], 'accuracy_preservation', 1.0) for method in methods]
        
        fig.add_trace(go.Bar(x=methods, y=ratios, name='Compression Ratio'), row=1, col=1)
        fig.add_trace(go.Bar(x=methods, y=speeds, name='Inference Speedup'), row=1, col=2)
        fig.add_trace(go.Bar(x=methods, y=accuracies, name='Accuracy'), row=2, col=1)
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=ratios, y=speeds, 
            mode='markers+text',
            text=methods,
            textposition="top center",
            marker=dict(size=10, color=accuracies, colorscale='viridis'),
            name='Methods'
        ), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False, title_text="TorchSlim Compression Dashboard")
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        return fig
    
    def save_all_plots(
        self,
        results: Dict[str, Any],
        output_dir: str,
        layer_info: Optional[Dict[str, Dict[str, Any]]] = None,
        formats: List[str] = ['png', 'pdf']
    ):
        """
        Save all available plots for the given results
        
        Args:
            results: Dictionary mapping method names to results
            output_dir: Directory to save plots
            layer_info: Optional layer-wise information
            formats: List of formats to save ('png', 'pdf', 'html')
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plots_to_generate = [
            ('comparison', lambda: self.plot_compression_comparison(results)),
            ('scatter', lambda: self.plot_compression_scatter(results)),
            ('pareto', lambda: self.plot_pareto_frontier(results)),
            ('dashboard', lambda: self.create_dashboard(results, layer_info))
        ]
        
        if layer_info:
            plots_to_generate.append(
                ('layer_breakdown', lambda: self.plot_layer_compression_breakdown(layer_info))
            )
        
        for plot_name, plot_func in plots_to_generate:
            try:
                fig = plot_func()
                
                for fmt in formats:
                    if fmt in ['png', 'pdf'] and MATPLOTLIB_AVAILABLE:
                        save_path = output_path / f"{plot_name}.{fmt}"
                        fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    elif fmt == 'html' and PLOTLY_AVAILABLE and hasattr(fig, 'write_html'):
                        save_path = output_path / f"{plot_name}.html"
                        fig.write_html(save_path)
                
                logger.info(f"Generated {plot_name} plot")
                
            except Exception as e:
                logger.error(f"Failed to generate {plot_name} plot: {e}")
        
        logger.info(f"All plots saved to {output_dir}")


# Convenience functions
def quick_visualize(
    results: Dict[str, Any],
    plot_type: str = 'comparison',
    save_path: Optional[str] = None,
    interactive: bool = False
) -> Union[plt.Figure, go.Figure]:
    """
    Quick visualization of compression results
    
    Args:
        results: Dictionary mapping method names to results
        plot_type: Type of plot ('comparison', 'scatter', 'pareto', 'dashboard')
        save_path: Optional path to save the plot
        interactive: Whether to create interactive plot
        
    Returns:
        Figure object
    """
    
    viz = VisualizationTools()
    
    if plot_type == 'comparison':
        return viz.plot_compression_comparison(results, save_path=save_path, interactive=interactive)
    elif plot_type == 'scatter':
        return viz.plot_compression_scatter(results, save_path=save_path, interactive=interactive)
    elif plot_type == 'pareto':
        return viz.plot_pareto_frontier(results, save_path=save_path, interactive=interactive)
    elif plot_type == 'dashboard':
        return viz.create_dashboard(results, save_path=save_path)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

def create_compression_report(
    results: Dict[str, Any],
    output_dir: str,
    layer_info: Optional[Dict[str, Dict[str, Any]]] = None
):
    """
    Create a complete visual compression report
    
    Args:
        results: Dictionary mapping method names to results
        output_dir: Directory to save the report
        layer_info: Optional layer-wise information
    """
    
    viz = VisualizationTools()
    viz.save_all_plots(results, output_dir, layer_info, formats=['png', 'html'])
    
    logger.info(f"Complete compression report saved to {output_dir}")