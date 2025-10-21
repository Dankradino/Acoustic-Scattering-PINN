import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import trimesh


'''
This module introduces shape generation and related methods used for test example.
'''



#####################################################################################
# 2D Shape generation
#####################################################################################

def generate_star(num_points=5, inner_radius=0.5, outer_radius=1.0, center=(0.0, 0.0), rotation=0.0, smoothness=100):
    """
    Generates a smooth (rounded) star shape using sinusoidal modulation of radius.
    `smoothness` controls how smooth the outline is (number of sampled points).
    Returns: (smoothness x 2) numpy array of (x, y) points
    """
    angles = np.linspace(0, 2 * np.pi, smoothness, endpoint=False) + rotation
    # Sinusoidal modulation to smoothly alternate between inner and outer radii
    radii = (outer_radius + inner_radius)/2 + (outer_radius - inner_radius)/2 * np.cos(num_points * angles)
    
    x = radii * np.cos(angles) + center[0]
    y = radii * np.sin(angles) + center[1]
    return np.stack([x, y], axis=1)


def generate_ellipse(rx=1.0, ry=0.5, center=(0.0, 0.0), num_points=100, rotation=0.0):
    """
    Generates an ellipse with radii rx and ry, centered at `center`.
    Returns: (num_points x 2) numpy array of (x, y) points
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = rx * np.cos(angles)
    y = ry * np.sin(angles)
    
    # Apply rotation if needed
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    x_rot = x * cos_r - y * sin_r + center[0]
    y_rot = x * sin_r + y * cos_r + center[1]
    
    return np.stack([x_rot, y_rot], axis=1)


def generate_square(side=1.0, center=(0.0, 0.0), corner_radius=0.1, segments_per_corner=8):
    """
    Generate a square with rounded corners.
    
    Args:
        side: Side length of the square
        center: Center point (x, y) of the square
        corner_radius: Radius of the rounded corners
        segments_per_corner: Number of line segments to approximate each rounded corner
    
    Returns:
        numpy array of (x, y) coordinates forming the rounded square
    """
    half = side / 2
    cx, cy = center
    
    # Clamp corner radius to maximum possible value
    max_radius = side / 2
    corner_radius = min(corner_radius, max_radius)
    
    # Calculate the straight edge length after accounting for rounded corners
    straight_length = half - corner_radius
    
    points = []
    
    # Define the four corner centers
    corners = [
        (cx + straight_length, cy + straight_length),  # Top-right
        (cx - straight_length, cy + straight_length),  # Top-left  
        (cx - straight_length, cy - straight_length),  # Bottom-left
        (cx + straight_length, cy - straight_length),  # Bottom-right
    ]
    
    # Define start angles for each corner (in radians)
    start_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    
    for i, ((corner_x, corner_y), start_angle) in enumerate(zip(corners, start_angles)):
        # Add the rounded corner
        for j in range(segments_per_corner):
            angle = start_angle + (j * np.pi/2) / (segments_per_corner - 1)
            x = corner_x + corner_radius * np.cos(angle)
            y = corner_y + corner_radius * np.sin(angle)
            points.append([x, y])
    
    return np.array(points)


def generate_circle(radius=1.0, center=(0.0, 0.0), num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(angles) + center[0]
    y = radius * np.sin(angles) + center[1]
    return np.stack([x, y], axis=1)


def densify_polygon_with_normals(polygon, total_points=1000):
    """
    Densify the boundary of a polygon and compute unit outward normals.
    
    Args:
        polygon: matplotlib.patches.Polygon or (N, 2) array-like
        total_points: number of total boundary samples
    
    Returns:
        points: (total_points, 2) array of sampled points
        normals: (total_points, 2) array of unit normals at those points
    """
    if isinstance(polygon, Path):
        vertices = polygon.vertices
    else:
        vertices = np.asarray(polygon)

    # Ensure closed loop
    if not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])

    edges = vertices[1:] - vertices[:-1]
    lengths = np.linalg.norm(edges, axis=1)
    total_length = np.sum(lengths)

    edge_counts = np.maximum((lengths / total_length * total_points).astype(int), 1)

    sampled_points = []
    normals = []

    for i in range(len(edges)):
        start = vertices[i]
        end = vertices[i + 1]
        edge_vec = end - start
        edge_len = np.linalg.norm(edge_vec)
        
        if edge_len == 0:
            continue
        
        # Perpendicular (normal) vector: rotate 90° counter-clockwise
        normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len

        # Interpolate points along the edge
        n_samples = edge_counts[i]
        t = np.linspace(0, 1, n_samples, endpoint=False)
        points = start + (end - start)[None, :] * t[:, None]

        sampled_points.append(points)
        normals.append(np.tile(normal, (n_samples, 1)))

    return np.vstack(sampled_points), np.vstack(normals)



#####################################################################################
# 3D Shape generation
#####################################################################################



def generate_sphere(filename='trimesh_sphere.obj', radius=1.0, subdivisions=2, center=(0, 0, 0)):
    """
    Use trimesh's built-in sphere generation (most efficient and robust).
    
    Args:
        filename: Output filename for the mesh
        radius: Radius of the sphere
        subdivisions: Number of subdivision levels
        center: Tuple of (x, y, z) for the sphere's center
    """
    # Create the sphere at origin
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    
    # Translate to desired center
    sphere.apply_translation(center)
    
    # Export to OBJ
    sphere.export(filename)
    
    print(f"Created trimesh sphere: {filename}")
    print(f"  Center: {center}")
    print(f"  Radius: {radius}")
    print(f"  Subdivisions: {subdivisions}")
    print(f"  Vertices: {len(sphere.vertices)}")
    print(f"  Faces: {len(sphere.faces)}")
    print(f"  Watertight: {sphere.is_watertight}")
    print(f"  Volume: {sphere.volume:.6f}")
    
    return filename

def get_sphere_param(vertices):
    R = np.linalg.norm(vertices[0])
    return vertices, vertices/R
