import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.patches as patches
import trimesh
from Dataloader import loader3D
import plotly.graph_objects as go
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings('ignore')

''' 
This modules introduces visuals utils for plotting and selected training points visualization
'''

def create_obstacle_patch(shape, shape_type="polygon", **kwargs):
    """
    Create a matplotlib patch for visualization.
    
    Parameters:
        shape: 
            - if shape_type="circle", dict with keys: 'center' (Tensor or array), 'radius' (float)
            - if shape_type="polygon", (N, 2) array-like of vertex coordinates
        shape_type: "circle" or "polygon"
        kwargs: passed to the patch constructor (e.g., color, alpha, linestyle)
        
    Returns:
        matplotlib patch object (Circle or Polygon)
    """
    if shape_type == "circle":
        cx, cy = shape['center']
        if isinstance(cx, torch.Tensor):  # handle tensors
            cx, cy = cx.item(), cy.item()
        R = shape['radius']
        return patches.Circle((cx, cy), R, **kwargs)

    elif shape_type == "polygon":
        vertices = np.asarray(shape)
        vertices = vertices[:, [1, 0]]
        return patches.Polygon(vertices, closed=True, **kwargs)

    else:
        raise ValueError(f"Unsupported shape_type: {shape_type}")
    

def visualize_3d_points_matplotlib(points, mesh=None, title="3D Points Visualization", 
                                 point_size=1, alpha=0.6, mesh_alpha=0.3, max_points=50000):
    """
    Visualize 3D points using matplotlib with optional mesh overlay.
    
    Args:
        points: numpy array of shape (N, 3) containing the points
        mesh: optional trimesh object to overlay
        title: plot title
        point_size: size of points
        alpha: transparency of points
        mesh_alpha: transparency of mesh
        max_points: maximum number of points to display for performance
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with sampling for performance
    if len(points) > 0:
        if len(points) > max_points:
            print(f"Sampling {max_points} points from {len(points)} total points for visualization")
            indices = np.random.choice(len(points), max_points, replace=False)
            sampled_points = points[indices]
        else:
            sampled_points = points
            
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], 
                  c='blue', s=point_size, alpha=alpha, 
                  label=f'Points ({len(sampled_points)} shown/{len(points)} total)')
    
    # Plot mesh if provided
    if mesh is not None:
        vertices = mesh.vertices
        # Sample vertices for visualization (if too many)
        if len(vertices) > 5000:
            print(f"Sampling mesh vertices for visualization: {len(vertices)} -> 5000")
            indices = np.random.choice(len(vertices), 5000, replace=False)
            vertices = vertices[indices]
        
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c='red', s=point_size*2, alpha=mesh_alpha, label='Mesh vertices (sampled)')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Make axes equal using all points (not just sampled)
    all_coords = points if len(points) > 0 else np.array([[0, 0, 0]])
    max_range = np.array([all_coords[:, 0].max()-all_coords[:, 0].min(),
                         all_coords[:, 1].max()-all_coords[:, 1].min(),
                         all_coords[:, 2].max()-all_coords[:, 2].min()]).max() / 2.0
    mid_x = (all_coords[:, 0].max()+all_coords[:, 0].min()) * 0.5
    mid_y = (all_coords[:, 1].max()+all_coords[:, 1].min()) * 0.5
    mid_z = (all_coords[:, 2].max()+all_coords[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig

def visualize_3d_points_plotly(points, mesh=None, title="3D Points Visualization", max_points=50000):
    """
    Interactive 3D visualization using Plotly.
    
    Args:
        points: numpy array of shape (N, 3) containing the points
        mesh: optional trimesh object to overlay
        title: plot title
        max_points: maximum number of points to display for performance
    """
    fig = go.Figure()
    
    # Add points with sampling for performance
    if len(points) > 0:
        # Sample points if too many for performance
        if len(points) > max_points:
            print(f"Sampling {max_points} points from {len(points)} total points for visualization")
            indices = np.random.choice(len(points), max_points, replace=False)
            sampled_points = points[indices]
        else:
            sampled_points = points
            
        # Color points by distance from origin for better visualization
        distances = np.linalg.norm(sampled_points, axis=1)
        
        fig.add_trace(go.Scatter3d(
            x=sampled_points[:, 0],
            y=sampled_points[:, 1], 
            z=sampled_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=distances,
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title="Distance from Origin")
            ),
            name=f'Generated Points ({len(sampled_points)} shown/{len(points)} total)',
            hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<br>Distance: %{marker.color:.3f}<extra></extra>'
        ))
    
    # Add mesh if provided
    if mesh is not None:
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Sample mesh vertices for better performance if too large
        if len(vertices) > 10000:
            print(f"Sampling mesh vertices for visualization: {len(vertices)} -> 10000")
            # Sample vertices instead of using simplification
            vertex_indices = np.random.choice(len(vertices), 10000, replace=False)
            sampled_vertices = vertices[vertex_indices]
            
            # Add as scatter points instead of mesh
            fig.add_trace(go.Scatter3d(
                x=sampled_vertices[:, 0],
                y=sampled_vertices[:, 1],
                z=sampled_vertices[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color='red',
                    opacity=0.3
                ),
                name=f'Mesh Vertices (sampled)',
                hovertemplate='Mesh Vertex<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ))
        else:
            # For smaller meshes, try to show as mesh or points
            try:
                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    opacity=0.3,
                    color='red',
                    name='Mesh'
                ))
            except:
                # Fallback to scatter points
                fig.add_trace(go.Scatter3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    mode='markers',
                    marker=dict(size=1, color='red', opacity=0.3),
                    name='Mesh Vertices'
                ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        width=900,
        height=700
    )
    
    return fig

def analyze_point_distribution(points, mesh_center=None, title="Point Distribution Analysis"):
    """
    Analyze and visualize the distribution of generated points.
    
    Args:
        points: numpy array of shape (N, 3) containing the points
        mesh_center: center of the mesh for distance calculations
        title: analysis title
    """
    if len(points) == 0:
        print("No points to analyze!")
        return
    
    if mesh_center is None:
        mesh_center = np.array([0., 0., 0.])
    
    # Calculate distances from mesh center
    distances = np.linalg.norm(points - mesh_center, axis=1)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # Distance histogram
    axes[0, 0].hist(distances, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Distance from Center')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distance Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Coordinate distributions
    coords = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    for i, (coord, color) in enumerate(zip(coords, colors)):
        axes[0, 1].hist(points[:, i], bins=30, alpha=0.5, label=coord, color=color)
    axes[0, 1].set_xlabel('Coordinate Value')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Coordinate Distributions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2D projections
    axes[1, 0].scatter(points[:, 0], points[:, 1], alpha=0.5, s=1)
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_title('XY Projection')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')
    
    axes[1, 1].scatter(points[:, 0], points[:, 2], alpha=0.5, s=1)
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Z')
    axes[1, 1].set_title('XZ Projection')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\n=== Point Distribution Statistics ===")
    print(f"Total points: {len(points)}")
    print(f"Distance from center - Min: {distances.min():.4f}, Max: {distances.max():.4f}, Mean: {distances.mean():.4f}")
    print(f"X range: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}]")
    print(f"Y range: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")
    print(f"Z range: [{points[:, 2].min():.4f}, {points[:, 2].max():.4f}]")
    
    return fig

def visualize_dataloader_points(config, cache_dir='.', mesh_path=None, 
                              phase='adam', interactive=True, analyze=True, 
                              max_points=50000, sample_batches=None):
    """
    Complete visualization pipeline for dataloader points.
    
    Args:
        config: configuration dictionary
        cache_dir: directory containing cached points
        mesh_path: path to the mesh file for overlay
        phase: which phase to visualize ('adam' or 'fine')
        interactive: whether to use interactive Plotly visualization
        analyze: whether to perform distribution analysis
        max_points: maximum number of points to visualize
        sample_batches: number of batches to sample from dataloader (None = all)
    """
    print(f"Visualizing {phase} phase points...")
    
    # Load points from cache
    try:
        points_data = loader3D(config, cache_dir)
        dataloader = points_data[phase]
        
        # Extract points from dataloader with optional batch sampling
        all_points = []
        batch_count = 0
        
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                all_points.append(batch.numpy())
            else:
                all_points.append(batch)
            
            batch_count += 1
            
            # Stop after sampling specified number of batches
            if sample_batches is not None and batch_count >= sample_batches:
                print(f"Sampled {batch_count} batches from dataloader")
                break
        
        if len(all_points) > 0:
            points = np.vstack(all_points)
        else:
            print("No points found in dataloader!")
            return
            
    except Exception as e:
        print(f"Error loading points: {e}")
        return
    
    # Load mesh if provided
    mesh = None
    if mesh_path is not None:
        try:
            mesh = trimesh.load(mesh_path)
            print(f"Loaded mesh with {len(mesh.vertices)} vertices")
        except Exception as e:
            print(f"Error loading mesh: {e}")
    
    print(f"Total points loaded: {len(points)}")
    
    # Sample points if too many for performance
    if len(points) > max_points:
        print(f"Sampling {max_points} points from {len(points)} for performance")
        sample_indices = np.random.choice(len(points), max_points, replace=False)
        sampled_points_for_analysis = points[sample_indices]
    else:
        sampled_points_for_analysis = points
    
    # Create visualizations
    if interactive:
        # Interactive Plotly visualization
        fig_interactive = visualize_3d_points_plotly(
            points, mesh, f"{phase.capitalize()} Phase - 3D Points", max_points
        )
        fig_interactive.show()
    else:
        # Static matplotlib visualization
        fig_static = visualize_3d_points_matplotlib(
            points, mesh, f"{phase.capitalize()} Phase - 3D Points", max_points=max_points
        )
        plt.show()
    
    # Analysis on sampled points
    if analyze:
        mesh_center = mesh.centroid if mesh is not None else np.array([0., 0., 0.])
        fig_analysis = analyze_point_distribution(
            sampled_points_for_analysis, mesh_center, 
            f"{phase.capitalize()} Phase - Distribution Analysis (sampled)"
        )
        plt.show()

# Example usage function
def example_usage():
    """
    Example of how to use the visualization functions.
    Modify the paths and config to match your setup.
    """
    # Example configuration (modify as needed)
    config = {
        'adam': {
            'batch_size': 512,
            'epochs': 1000,
            'keeping_time': 100
        },
        'fine': {
            'batch_size': 256,
            'epochs': 2000,
            'keeping_time': 200
        },
        'L': 2.0  # Your L parameter
    }
    
    # Paths (modify as needed)
    cache_dir = '.'
    mesh_path = "/path/to/your/mesh.stl"  # Update this path
    
    # Visualize both phases with performance optimizations
    for phase in ['adam', 'fine']:
        print(f"\n{'='*50}")
        print(f"VISUALIZING {phase.upper()} PHASE")
        print('='*50)
        
        visualize_dataloader_points(
            config=config,
            cache_dir=cache_dir,
            mesh_path=mesh_path,
            phase=phase,
            interactive=True,  # Set to False for matplotlib
            analyze=True,
            max_points=50000,  # Limit points for performance
            sample_batches=10   # Only load first 10 batches for quick preview
        )
