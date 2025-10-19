from numba import njit
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

'''
This module introduces 2D training coordinates generator.
'''

@njit
def is_point_inside_polygon_numba(x, y, polygon_x, polygon_y):
    n = polygon_x.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_x[i], polygon_y[i]
        xj, yj = polygon_x[j], polygon_y[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside

@njit
def is_point_inside_any_polygon(x, y, all_polygons_x, all_polygons_y, polygon_starts, polygon_lengths):
    """Check if point is inside any of the polygons"""
    n_polygons = polygon_starts.shape[0]
    for poly_idx in range(n_polygons):
        start = polygon_starts[poly_idx]
        length = polygon_lengths[poly_idx]
        
        # Extract current polygon coordinates
        polygon_x = all_polygons_x[start:start + length]
        polygon_y = all_polygons_y[start:start + length]
        
        if is_point_inside_polygon_numba(x, y, polygon_x, polygon_y):
            return True
    return False

@njit
def batch_points_inside_any_polygon(points_x, points_y, all_polygons_x, all_polygons_y, polygon_starts, polygon_lengths):
    """Batch check if points are inside any of the polygons"""
    n_points = points_x.shape[0]
    results = np.zeros(n_points, dtype=np.bool_)
    for i in range(n_points):
        results[i] = is_point_inside_any_polygon(
            points_x[i], points_y[i], 
            all_polygons_x, all_polygons_y, 
            polygon_starts, polygon_lengths
        )
    return results

@njit
def batch_points_inside_polygon(points_x, points_y, polygon_x, polygon_y):
    """Original single polygon function for backward compatibility"""
    n_points = points_x.shape[0]
    results = np.zeros(n_points, dtype=np.bool_)
    for i in range(n_points):
        results[i] = is_point_inside_polygon_numba(points_x[i], points_y[i], polygon_x, polygon_y)
    return results

class OutsidePointsDataset(Dataset):
    def __init__(self, points):
        self.points = torch.tensor(points, dtype=torch.float32)
    
    def __len__(self):
        return self.points.shape[0]
    
    def __getitem__(self, idx):
        return self.points[idx]

def prepare_polygons_for_numba(polygons):
    """
    Convert list of polygons to flattened arrays for numba processing
    
    Args:
        polygons: list of numpy arrays, each representing a polygon
    
    Returns:
        tuple: (all_polygons_x, all_polygons_y, polygon_starts, polygon_lengths)
    """
    if not isinstance(polygons, list):
        # Single polygon case - convert to list for uniform processing
        polygons = [polygons]
    
    all_x = []
    all_y = []
    polygon_starts = []
    polygon_lengths = []
    
    current_start = 0
    for polygon in polygons:
        polygon_x = polygon[:, 0]
        polygon_y = polygon[:, 1]
        
        all_x.extend(polygon_x)
        all_y.extend(polygon_y)
        polygon_starts.append(current_start)
        polygon_lengths.append(len(polygon_x))
        current_start += len(polygon_x)
    
    return (
        np.array(all_x, dtype=np.float64),
        np.array(all_y, dtype=np.float64),
        np.array(polygon_starts, dtype=np.int64),
        np.array(polygon_lengths, dtype=np.int64)
    )

def generate_outside_points(polygons, target_num_points, batch_size=4096, length=3):
    """
    Generate points outside the given polygon(s)
    
    Args:
        polygons: numpy array (single polygon) or list of numpy arrays (multiple polygons)
        target_num_points: number of outside points to generate
        batch_size: batch size for generation
        length: boundary length for random point generation
    
    Returns:
        numpy array of outside points
    """
    # Handle both single polygon and list of polygons
    if isinstance(polygons, list):
        # Multiple polygons
        all_polygons_x, all_polygons_y, polygon_starts, polygon_lengths = prepare_polygons_for_numba(polygons)
        use_multiple = True
    else:
        # Single polygon - backward compatibility
        polygon_x = polygons[:, 0]
        polygon_y = polygons[:, 1]
        use_multiple = False
    
    # Calculate bounding box
    min_x, max_x = -length, length
    min_y, max_y = -length, length
    
    collected_points = []
    collected_count = 0
    
    while collected_count < target_num_points:
        # Generate batch of random points inside expanded bounding box
        points_x = np.random.uniform(min_x, max_x, batch_size * 2)
        points_y = np.random.uniform(min_y, max_y, batch_size * 2)
        
        # Test which are inside polygon(s)
        if use_multiple:
            inside_mask = batch_points_inside_any_polygon(
                points_x, points_y, 
                all_polygons_x, all_polygons_y, 
                polygon_starts, polygon_lengths
            )
        else:
            inside_mask = batch_points_inside_polygon(points_x, points_y, polygon_x, polygon_y)
        
        # Keep only outside points
        outside_points_x = points_x[~inside_mask]
        outside_points_y = points_y[~inside_mask]
        
        # Append until we reach target count
        needed = target_num_points - collected_count
        take = min(needed, outside_points_x.size)
        
        if take > 0:
            new_points = np.stack([outside_points_x[:take], outside_points_y[:take]], axis=1)
            collected_points.append(new_points)
            collected_count += take
        
        print(f"Collected {collected_count}/{target_num_points} points...", end='\r')
    
    all_points = np.vstack(collected_points)
    return all_points

def create_dataloader(polygons, config, fake_bd):
    """
    Create dataloader for outside points
    
    Args:
        polygons: numpy array (single polygon) or list of numpy arrays (multiple polygons)
        config: configuration dictionary
        fake_bd: boundary length for point generation
    
    Returns:
        DataLoader with outside points
    """
    num_sample = 2 * int(config['epochs'] / config['keeping_time'])
    target_points = num_sample * config['batch_size']
    
    outside_points = generate_outside_points(polygons, target_points, batch_size=config['batch_size'], length=fake_bd)
    print(f"\nTotal outside points generated: {outside_points.shape[0]}")
    
    # Create dataset and dataloader
    dataset = OutsidePointsDataset(outside_points)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    return dataloader
