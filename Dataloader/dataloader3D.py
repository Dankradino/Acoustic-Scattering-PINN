import numpy as np
from numba import jit
import torch
import os
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, Generator
import time
import trimesh
from .dataloader2D import OutsidePointsDataset
import psutil
import gc

'''
This module introduces 3D training coordinates generator and loader.
'''

@jit(nopython=True)
def filter_points_by_mask(points, mask):
    """Filter points using boolean mask (inverted for outside points)"""
    return points[~mask]

@jit(nopython=True)
def generate_spherical_constrained_points(batch_size, min_radius, max_radius, center, L):
    """
    Generate points in a spherical shell between min_radius and max_radius,
    constrained to fit within the box [-L, L]³.
    """
    points = np.empty((batch_size, 3), dtype=np.float64)
    generated = 0
    while generated < batch_size :
        # Generate random direction (uniform on sphere)
        # Using Marsaglia's method for uniform sphere sampling
        while True:
            x1 = 2.0 * np.random.random() - 1.0
            x2 = 2.0 * np.random.random() - 1.0
            if x1*x1 + x2*x2 < 1.0:
                break
        
        # Convert to 3D unit vector
        sqrt_term = np.sqrt(1.0 - x1*x1 - x2*x2)
        direction = np.array([2.0*x1*sqrt_term, 2.0*x2*sqrt_term, 1.0 - 2.0*(x1*x1 + x2*x2)])
        # Generate random radius in shell (uniform volume distribution)
        u = np.random.random()
        r3_min = min_radius**3
        r3_max = max_radius**3
        r = (u * (r3_max - r3_min) + r3_min)**(1.0/3.0)
        
        # Scale direction by radius and add center
        point = center + r * direction
        # Check if point fits within the box bounds
        if (point[0] >= -L and point[0] <= L and 
            point[1] >= -L and point[1] <= L and 
            point[2] >= -L and point[2] <= L):
            points[generated] = point
            generated += 1

    return points


@jit(nopython=True)
def generate_pml_points(batch_size, center, L, L_pml):
    """
    Generate PML (Perfectly Matched Layer) points in 6 rectangular slabs around a central cube.
    
    Parameters:
    - batch_size: Total number of points to generate
    - center: Center position of the domain
    - L: Size of the inner computation cube
    - L_pml: Size of the PML domain
    """
    mid_dist = (L_pml - L) / 2 + L
    diff_dist = L_pml - L
    # Define the 3 slice orientations (x, y, z slabs)
    slice_cube = np.array([
        [diff_dist / 2, L + diff_dist / 2, L + diff_dist / 2],  # x-oriented slab
        [L + diff_dist / 2, diff_dist / 2, L + diff_dist / 2],  # y-oriented slab
        [L + diff_dist / 2, L + diff_dist / 2, diff_dist / 2]   # z-oriented slab
    ], dtype=np.float64)
    
    # Define centroids for 6 slabs (±x, ±y, ±z directions)
    centroid = np.array([
        [mid_dist, diff_dist / 2, diff_dist / 2],   # +x slab
        [- diff_dist / 2, mid_dist, diff_dist / 2],   # +y slab
        [- diff_dist / 2, - diff_dist / 2, mid_dist],   # +z slab
        [-mid_dist, -diff_dist / 2, - diff_dist / 2],  # -x slab
        [diff_dist / 2, - mid_dist, - diff_dist / 2],  # -y slab
        [diff_dist / 2, diff_dist / 2, - mid_dist]   # -z slab
    ], dtype=np.float64)
    
    mini_batch = batch_size // 6
    points = np.zeros((batch_size, 3))
    
    # Generate points for each of the 6 slabs
    for i in range(6):
        # Generate random points in [-1, 1]^3 cube
        cube_points = 1 - 2 * np.random.random((mini_batch, 3))
        
        # Transform to appropriate slab
        slice_idx = i % 3
        points[i*mini_batch: (i+1)*mini_batch] = (
            slice_cube[slice_idx] * cube_points + centroid[i]
        )
    
    return points + center

@jit(nopython=True)
def generate_bbox_with__constraint(n_points, bounds_min, bounds_max, center, min_radius):
    """
    Generate points uniformly in a bounding box with minimum radius constraint.
    """
    points = np.empty((n_points, 3), dtype=np.float64)
    generated = 0
    attempts = 0
    
    while generated < n_points:
        attempts += 1
        
        # Generate random point in bounding box
        point = np.empty(3, dtype=np.float64)
        for j in range(3):
            point[j] = np.random.random() * (bounds_max[j] - bounds_min[j]) + bounds_min[j]
        
        # Check radius constraint
        distance = np.sqrt(np.sum((point - center)**2))
        if distance >= min_radius:
            points[generated] = point
            generated += 1
    
    # If we couldn't generate enough points, return what we have
    if generated < n_points:
        return points[:generated]
    return points

@jit(nopython=True)
def calculate_max_radius_in_box(L):
    """Calculate maximum radius that fits entirely within the box [-L, L]³"""
    # Distance from center to corner of box
    return np.sqrt(3)*L


class MeshOutsidePointGenerator:
    """
    Optimized point generator for creating training data outside mesh objects.
    Uses Numba-accelerated functions for fast point generation.
    """
    
    def __init__(self, mesh, config: Dict[str, Any], pml_boundary: float, L: float):
        """
        Initialize the optimized point generator.
        
        Args:
            mesh: Trimesh object or path to mesh file
            config: Configuration dictionary with generation parameters
            pml_boundary: PML boundary value (L parameter for box constraints)
        """
        # Load mesh if path is provided
        if isinstance(mesh, str):
            self.mesh = trimesh.load(mesh)
        else:
            self.mesh = mesh
        
        # Ensure mesh is watertight for reliable inside/outside testing
        if not self.mesh.is_watertight:
            print("Warning: Mesh is not watertight. Results may be unreliable.")
        
        self.config = config
        self.L = L
        self.pml_boundary = pml_boundary
        
        # Calculate mesh center and bounding sphere
        self.mesh_center = self.mesh.centroid
        self.mesh_bounds = self.mesh.bounds
        
        # Calculate bounding sphere radius
        vertices_centered = self.mesh.vertices - self.mesh_center
        self.mesh_radius = np.min(np.linalg.norm(vertices_centered, axis=1))
        
        # Set sampling bounds
        self.bounds = np.array([
            [-pml_boundary, -pml_boundary, -pml_boundary],
            [pml_boundary, pml_boundary, pml_boundary]
        ], dtype=np.float64)
        
        # Calculate maximum radius that fits in the box
        self.max_radius_in_box = calculate_max_radius_in_box(pml_boundary)
        
        # Initialize generation statistics
        self.generation_stats = []
        # Pre-compile Numba functions

        # print("Pre-compiling Numba functions...")
        self._warmup_numba()
        # print("Numba functions compiled and ready.")
    
    def _warmup_numba(self):
        """Warm up Numba functions by running them once with small arrays."""
        dummy_points = generate_spherical_constrained_points(
            10, 0, self.L, self.mesh_center, self.pml_boundary
        )
        dummy_bbox_points = generate_bbox_with__constraint(
            10, self.bounds[0], self.bounds[1], self.mesh_center, .0
        )
        dummy_pml_points = generate_pml_points(10, self.mesh_center, self.L, self.pml_boundary)
        dummy_mask = np.array([True, False, True, False, True], dtype=bool)
        dummy_filtered = filter_points_by_mask(dummy_points[:5], dummy_mask)
    
    def generate_random_points_spherical_shell(self, n_points: int, 
                                             min_radius: float = None,
                                             max_radius: float = None) -> np.ndarray:
        """Generate random points in a spherical shell around the mesh"""
        if min_radius is None:
            min_radius = self.mesh_radius
        if max_radius is None:
            max_radius = self.max_radius_in_box
        
        max_radius = min(max_radius, self.max_radius_in_box)
        
        if min_radius >= max_radius:
            print(f"Warning: min_radius ({min_radius}) >= max_radius ({max_radius})")
            min_radius = max_radius 
        
        return generate_spherical_constrained_points(
            n_points, min_radius, max_radius, self.mesh_center, self.pml_boundary
        )
    
    def generate_random_points_in_bbox(self, n_points: int, min_radius: float = None) -> np.ndarray:
        """Generate random points within the bounding box with minimum radius constraint"""
        if min_radius is None:
            min_radius = self.mesh_radius 
        
        return generate_bbox_with__constraint(
            n_points, self.bounds[0], self.bounds[1], self.mesh_center, min_radius
        )
    
    def filter_outside_points_raycast(self, points: np.ndarray, min_radius: float = None ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter points to keep only those outside the mesh"""
        if len(points) == 0:
            return np.array([]).reshape(0, 3), np.array([], dtype=bool)
        
        # Use trimesh's contains method for reliable inside/outside testing
        inside_mask = self.mesh.contains(points)
        outside_points = filter_points_by_mask(points, inside_mask)
        return outside_points, inside_mask
    
    def generate_outside_dataset(self, method: str = 'spherical_shell',
                                 max_iterations: int = 10,
                                 min_radius=None,
                               **kwargs) -> np.ndarray:
        """
        Generate a dataset of points outside the mesh.
        
        Args:
            method: Generation method ('spherical_shell' or 'bbox')
            target_points: Number of points to generate (uses config if None)
            max_iterations: Maximum iterations for generation
            min_radius: Minimum distance from mesh center
            **kwargs: Additional parameters for generation methods
            
        Returns:
            numpy array of points outside the mesh
        """
        num_sample = int(1.3 * self.config['epochs'] / self.config['keeping_time'])   
        batch_size = int(self.config['batch_size'])
        num_points = num_sample * batch_size
        if min_radius is None:
            min_radius = self.mesh_radius 
        
        factor =  2
        num_pml_points = num_points//factor
        num_in_box = num_points-num_pml_points
        all_in_box_points = []
        total_generated = 0
        self.generation_stats = []  # Reset statistics
        
        start_time = time.time()
        
        for iteration in range(max_iterations):
            if total_generated >= num_in_box:
                break
            
            remaining_points = num_in_box - total_generated
            
            # Adjust efficiency factor based on method
            if method == 'spherical_shell':
                efficiency_factor = 1.5  # More efficient for shell sampling
            else:
                efficiency_factor = 3.0  # Less efficient for bbox sampling
            
            candidate_count = max(int(remaining_points * efficiency_factor), 1000)
            
            # Generate candidate points
            if method == 'bbox':
                candidate_points = self.generate_random_points_in_bbox(candidate_count, min_radius)
            elif method == 'spherical_shell':
                candidate_points = self.generate_random_points_spherical_shell(
                    candidate_count, min_radius=min_radius, **kwargs
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if len(candidate_points) == 0:
                print(f"  No candidate points generated")
                continue
            
            # Filter to get only outside points
            outside_points, inside_mask = self.filter_outside_points_raycast(candidate_points)
            
            outside_count = len(outside_points)
            efficiency = outside_count / len(candidate_points) * 100 if len(candidate_points) > 0 else 0
            
            # Store statistics
            self.generation_stats.append({
                'iteration': iteration + 1,
                'candidates': len(candidate_points),
                'generated': outside_count,
                'efficiency': efficiency
            })
            
            if outside_count > 0:
                all_in_box_points.append(outside_points)
                total_generated += outside_count
                
            else:
                print(f"  No outside points generated this iteration")
        generation_time = time.time() - start_time
        print(f'Total generation time: {generation_time:.4f} seconds')
        
        if len(all_in_box_points) == 0:
            print("Warning: No outside points generated!")
            return np.array([]).reshape(0, 3)
        
        # Combine all points and trim to target
        in_points = np.vstack(all_in_box_points)
        if len(in_points) > num_in_box:
            indices = np.random.choice(len(in_points), num_in_box, replace=False)
            in_points = in_points[indices]
        
        pml_points = generate_pml_points(num_pml_points, self.mesh_center, self.L, self.pml_boundary)
        split_pml = batch_size // factor
        split_in_box = batch_size - split_pml

        # Compute number of full batches possible from X and Y
        max_batches_pml = len(pml_points) // split_pml
        max_batches_in_box = len(in_points) // split_in_box
        num_batches = min(max_batches_pml, max_batches_in_box)
        
        num_batches = min(len(pml_points) // split_pml, len(in_points) // split_in_box)

        # Reshape into batches
        pml_points = pml_points[:num_batches * split_pml].reshape(num_batches, split_pml, 3)
        in_points = in_points[:num_batches * split_in_box].reshape(num_batches, split_in_box, 3)

        final_points = np.concatenate([pml_points, in_points], axis=1) 
        final_points = np.copy(final_points)
        final_points = final_points.reshape(-1, 3)
        print(f"Final dataset: {len(final_points)} outside points")
        if len(final_points) > 0:
            distances = np.linalg.norm(final_points - self.mesh_center, axis=1)
            min_distance = np.min(distances)
            max_distance = np.max(distances)
            print(f'Distance range: [{min_distance:.4f}, {max_distance:.4f}]')
            print(f'All points satisfy radius constraint (>= {min_radius:.4f}): {np.all(distances >= min_radius)}')
            
            in_box = np.all((final_points >= -self.pml_boundary) & (final_points <= self.pml_boundary), axis=1)
            print(f'All points within box bounds: {np.all(in_box)}')
        
        return final_points
    

def create_dataloader3D(mesh, config: Dict[str, Any], pml_boundary: float, 
                       cache_dir: str = '.', method: str = 'spherical_shell',
                       visualize: bool = False, save_plots: bool = False):
    """
    Create 3D dataloaders for Adam and fine-tuning phases using optimized point generation.
    
    Args:
        mesh: Trimesh object or path to mesh file
        config: Configuration dictionary with 'adam' and 'fine' sub-configs
        pml_boundary: PML boundary value
        cache_dir: Directory to save cached point files
        method: Point generation method ('spherical_shell' or 'bbox')
        visualize: Whether to create visualizations
        save_plots: Whether to save plots to files
    """
    print("Creating optimized 3D dataloaders...")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Adam phase points
    print("\n" + "="*50)
    print("GENERATING ADAM PHASE POINTS")
    print("="*50)
    generator_adam = MeshOutsidePointGenerator(mesh, config['adam'], config['L'], pml_boundary)
    
    points_adam = generator_adam.generate_outside_dataset(
        method=method,
    )
    
    # Save Adam points
    adam_path = os.path.join(cache_dir, 'points_adams.pt')
    torch.save(points_adam, adam_path)
    print(f"Saved Adam points to: {adam_path}")
    
    # Fine-tuning phase points
    print("\n" + "="*50)
    print("GENERATING FINE-TUNING PHASE POINTS")
    print("="*50)
    
    generator_fine = MeshOutsidePointGenerator(mesh, config['fine'], config['L'], pml_boundary)
    
    points_fine = generator_fine.generate_outside_dataset(
        method=method,
    )
    
    # Save fine-tuning points
    fine_path = os.path.join(cache_dir, 'points_fine.pt')
    torch.save(points_fine, fine_path)
    print(f"Saved fine-tuning points to: {fine_path}")
    
    print("\nDataloader creation completed successfully!")
    return adam_path, fine_path


def loader3D(config: Dict[str, Any], cache_dir: str = '.'):
    """
    Load 3D dataloaders from cached point files.
    
    Args:
        config: Configuration dictionary with 'adam' and 'fine' sub-configs
        cache_dir: Directory containing cached point files
        
    Returns:
        Dictionary containing 'adam' and 'fine' dataloaders
    """
    print("Loading 3D dataloaders from cache...")
    
    # Load Adam points
    adam_path = os.path.join(cache_dir, 'points_adams.pt')
    if not os.path.exists(adam_path):
        raise FileNotFoundError(f"Adam points file not found: {adam_path}")
    
    points_adam = torch.load(adam_path, weights_only=False)
    print(f"Loaded {len(points_adam)} Adam points from: {adam_path}")
    
    # Create Adam dataset and dataloader
    dataset_adam = OutsidePointsDataset(points_adam)
    dataloader_adam = DataLoader(
        dataset_adam, 
        batch_size=config['adam']['batch_size'], 
        shuffle=True,
        num_workers=config['adam'].get('num_workers', 1),
        pin_memory=config['adam'].get('pin_memory', False)
    )
    
    # Load fine-tuning points
    fine_path = os.path.join(cache_dir, 'points_fine.pt')
    if not os.path.exists(fine_path):
        raise FileNotFoundError(f"Fine-tuning points file not found: {fine_path}")
    
    points_fine = torch.load(fine_path, weights_only=False)
    print(f"Loaded {len(points_fine)} fine-tuning points from: {fine_path}")
    
    # Create fine-tuning dataset and dataloader
    dataset_fine = OutsidePointsDataset(points_fine)
    dataloader_fine = DataLoader(
        dataset_fine, 
        batch_size=config['fine']['batch_size'], 
        shuffle=True,
        num_workers=config['fine'].get('num_workers', 0),
        pin_memory=config['fine'].get('pin_memory', False)
    )
    
    print("3D dataloaders loaded successfully!")
    
    return {
        'adam': dataloader_adam,
        'fine': dataloader_fine
    }



class MemoryEfficientMeshOutsidePointGenerator(MeshOutsidePointGenerator):
    """
    Memory-efficient version that generates points in chunks and saves incrementally
    """
    
    def __init__(self, mesh, config: Dict[str, Any], pml_boundary: float, L: float, 
                 max_memory_mb: int = 1000):
        """
        Args:
            max_memory_mb: Maximum memory to use for point generation in MB
        """
        super().__init__(mesh, config, pml_boundary, L)
        self.max_memory_mb = max_memory_mb
        
        # Estimate points per MB (assuming float64, 3 coords = 24 bytes per point)
        self.points_per_mb = (max_memory_mb * 1024 * 1024) // (24 * 8)  # Safety factor
        print(f"Max points per chunk: {self.points_per_mb}")
    
    def generate_points_chunked(self, total_points: int, method: str = 'spherical_shell', 
                               min_radius: float = None, **kwargs) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields chunks of points instead of all at once.
        
        Yields:
            numpy arrays of points (chunks)
        """
        if min_radius is None:
            min_radius = self.mesh_radius
            
        remaining_points = total_points
        chunk_id = 0
        
        while remaining_points > 0:
            # Calculate chunk size
            chunk_size = min(remaining_points, self.points_per_mb)
            print(f"Generating chunk {chunk_id + 1}: {chunk_size} points")
            
            # Generate points for this chunk
            efficiency_factor = 1.5 if method == 'spherical_shell' else 3.0
            candidate_count = max(int(chunk_size * efficiency_factor), 1000)
            
            chunk_points = []
            attempts = 0
            max_attempts = 10
            
            while len(chunk_points) < chunk_size and attempts < max_attempts:
                attempts += 1
                
                # Generate candidates
                if method == 'bbox':
                    candidates = self.generate_random_points_in_bbox(candidate_count, min_radius)
                elif method == 'spherical_shell':
                    candidates = self.generate_random_points_spherical_shell(
                        candidate_count, min_radius=min_radius, **kwargs
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Filter outside points
                if len(candidates) > 0:
                    outside_points, _ = self.filter_outside_points_raycast(candidates)
                    if len(outside_points) > 0:
                        chunk_points.append(outside_points)
                
                # Clean up candidates from memory
                del candidates
                if 'outside_points' in locals():
                    del outside_points
                gc.collect()
            
            if chunk_points:
                # Combine chunk points
                combined = np.vstack(chunk_points)
                
                # Filter points too close to mesh center (CRITICAL STEP)
                distances_from_mesh_center = np.linalg.norm(combined - self.mesh_center, axis=1)
                valid_mesh_distance = distances_from_mesh_center >= min_radius
                mesh_violations = np.sum(~valid_mesh_distance)
                
                if mesh_violations > 0:
                    print(f"  Filtering out {mesh_violations} points too close to mesh center (< {min_radius:.4f})")
                    combined = combined[valid_mesh_distance]
                
                # Take only what we need after filtering
                if len(combined) > chunk_size:
                    indices = np.random.choice(len(combined), chunk_size, replace=False)
                    combined = combined[indices]
                elif len(combined) == 0:
                    print(f"  Warning: No valid points remain after filtering in chunk {chunk_id + 1}")
                    # Try again with next iteration
                    del chunk_points, combined
                    continue
                
                yield combined
                remaining_points -= len(combined)
                chunk_id += 1
                
                # Clean up
                del chunk_points, combined
                gc.collect()
            else:
                print(f"Warning: Could not generate points for chunk {chunk_id + 1}")
                break
    
    def generate_and_save_dataset_chunked(self, cache_dir: str, file_prefix: str, 
                                        method: str = 'spherical_shell',
                                        min_radius: float = None, **kwargs) -> str:
        """
        Generate dataset in chunks and save to disk incrementally.
        
        Returns:
            Path to the saved dataset file
        """
        num_sample = int(1.3 * self.config['epochs'] / self.config['keeping_time'])   
        batch_size = int(self.config['batch_size'])
        num_points = num_sample * batch_size
        
        factor = 2
        num_pml_points = num_points // factor
        num_in_box = num_points - num_pml_points
        
        print(f"Total target points: {num_points} (PML: {num_pml_points}, In-box: {num_in_box})")
        
        # Generate PML points (these are usually small enough to fit in memory)
        print("Generating PML points...")
        pml_points = generate_pml_points(num_pml_points, self.mesh_center, self.L, self.pml_boundary)
        
        # Save chunks to temporary files
        temp_dir = os.path.join(cache_dir, 'temp_chunks')
        os.makedirs(temp_dir, exist_ok=True)
        
        chunk_files = []
        total_generated = 0
        
        # Generate in-box points in chunks
        print("Generating in-box points in chunks...")
        for chunk_idx, chunk in enumerate(self.generate_points_chunked(
            num_in_box, method=method, min_radius=min_radius, **kwargs)):
            
            # Save chunk to temporary file
            chunk_file = os.path.join(temp_dir, f'chunk_{chunk_idx}.npy')
            np.save(chunk_file, chunk)
            chunk_files.append(chunk_file)
            total_generated += len(chunk)
            
            print(f"Saved chunk {chunk_idx + 1} with {len(chunk)} points to {chunk_file}")
            print(f"Total generated so far: {total_generated}/{num_in_box}")
            
            # Clean up chunk from memory
            del chunk
            gc.collect()
        
        # Now combine chunks with PML points in batches
        print("Combining chunks into final batches...")
        final_file = os.path.join(cache_dir, f'{file_prefix}.pt')
        
        split_pml = batch_size // factor
        split_in_box = batch_size - split_pml
        
        # Calculate how many batches we can make
        max_batches_pml = len(pml_points) // split_pml
        max_batches_in_box = total_generated // split_in_box
        num_batches = min(max_batches_pml, max_batches_in_box)
        
        print(f"Creating {num_batches} batches...")
        
        # Process chunks and combine with PML
        final_batches = []
        in_box_used = 0
        pml_used = 0
        
        for batch_idx in range(num_batches):
            # Get PML points for this batch
            pml_batch = pml_points[pml_used:pml_used + split_pml]
            pml_used += split_pml
            
            # Get in-box points for this batch
            in_box_batch = []
            points_needed = split_in_box
            
            # Load from chunk files as needed
            for chunk_file in chunk_files:
                if points_needed <= 0:
                    break
                
                # Load chunk
                chunk = np.load(chunk_file)
                available_start = max(0, in_box_used - sum(len(np.load(f)) for f in chunk_files[:chunk_files.index(chunk_file)]))
                
                if available_start < len(chunk):
                    take = min(points_needed, len(chunk) - available_start)
                    in_box_batch.append(chunk[available_start:available_start + take])
                    points_needed -= take
                    in_box_used += take
            
            if in_box_batch:
                in_box_batch = np.vstack(in_box_batch)
                # Combine PML and in-box for this batch
                batch = np.concatenate([pml_batch.reshape(1, split_pml, 3), 
                                      in_box_batch.reshape(1, split_in_box, 3)], axis=1)
                final_batches.append(batch)
            
            # Save periodically to avoid memory buildup
            if (batch_idx + 1) % 100 == 0:  # Save every 100 batches
                if final_batches:
                    partial_data = np.vstack(final_batches).reshape(-1, 3)
                    if batch_idx == 99:  # First save
                        torch.save(partial_data, final_file)
                    else:  # Append to existing
                        existing = torch.load(final_file, weights_only=False)
                        combined = np.vstack([existing, partial_data])
                        torch.save(combined, final_file)
                        del existing
                    
                    final_batches.clear()
                    del partial_data
                    gc.collect()
                    print(f"Saved progress: {batch_idx + 1}/{num_batches} batches")
        
        # Save remaining batches
        if final_batches:
            partial_data = np.vstack(final_batches).reshape(-1, 3)
            if os.path.exists(final_file):
                existing = torch.load(final_file, weights_only=False)
                combined = np.vstack([existing, partial_data])
                torch.save(combined, final_file)
                del existing
            else:
                torch.save(partial_data, final_file)
            del partial_data
        
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
        os.rmdir(temp_dir)
        
        # Load and verify final result
        final_points = torch.load(final_file, weights_only=False)
        print(f"Final dataset saved: {len(final_points)} points to {final_file}")
        
        return final_file

def create_memory_efficient_dataloader3D(mesh, config: Dict[str, Any], pml_boundary: float, 
                                        cache_dir: str = '.', method: str = 'spherical_shell',
                                        max_memory_mb: int = 1000):
    """
    Create 3D dataloaders using memory-efficient generation
    
    Args:
        max_memory_mb: Maximum memory to use for generation (MB)
    """
    print(f"Creating memory-efficient 3D dataloaders (max {max_memory_mb} MB)...")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Adam phase
    print("\n" + "="*50)
    print("GENERATING ADAM PHASE POINTS (MEMORY-EFFICIENT)")
    print("="*50)
    config = chunk_config(config)
    generator_adam = MemoryEfficientMeshOutsidePointGenerator(
        mesh, config['adam'], pml_boundary, config['L'], config['memory_settings']['max_memory_mb']
    )
    
    adam_path = generator_adam.generate_and_save_dataset_chunked(
        cache_dir, 'points_adams', method=method
    )
    
    # Fine-tuning phase
    print("\n" + "="*50)
    print("GENERATING FINE-TUNING PHASE POINTS (MEMORY-EFFICIENT)")
    print("="*50)
    
    generator_fine = MemoryEfficientMeshOutsidePointGenerator(
        mesh, config['fine'], pml_boundary, config['L'], config['memory_settings']['max_memory_mb']
    )
    
    fine_path = generator_fine.generate_and_save_dataset_chunked(
        cache_dir, 'points_fine', method=method
    )
    
    print("\nMemory-efficient dataloader creation completed!")
    return adam_path, fine_path



# Additional utility functions
def monitor_memory_usage():
    """Monitor current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.1f} MB")
    return memory_mb

def cleanup_memory():
    """Force garbage collection and cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def chunk_config(config): 
        config['memory_settings'] = {
            # Use larger chunks but not too large to avoid NumPy limitations
            'max_memory_mb': 2000,  # 2GB chunks - good balance
            'target_ram_usage_percent': 80,  # Can use more with this much RAM
            'safety_factor': 0.8,
            'max_points_per_chunk': 50_000_000,  # Limit to avoid array issues
        }
    
        
        config['parallel_settings'] =  {
            'num_workers': 12,  # Use half your cores for I/O
            'pin_memory': True,
            'prefetch_factor': 4,  # Higher prefetch with more RAM
            'persistent_workers': True,  # Keep workers alive
        }
        
        config['generation_settings'] = {
            'max_iterations': 15,
            'efficiency_factors': {
                'spherical_shell': 1.2,
                'bbox': 2.0
            }
        }
        return config



def loader3D(config: Dict[str, Any], cache_dir: str = '.', 
                     adam_filename: str = None, fine_filename: str = None):
    """
    Load 3D dataloaders from cached point files with flexible filenames.
    
    Args:
        config: Configuration dictionary with 'adam' and 'fine' sub-configs
        cache_dir: Directory containing cached point files
        adam_filename: Custom filename for adam points (without extension)
        fine_filename: Custom filename for fine points (without extension)
    
    Returns:
        Dictionary containing 'adam' and 'fine' dataloaders
    """
    print("Loading 3D dataloaders from cache...")
    
    # Auto-detect filenames if not provided
    if adam_filename is None:
        # Try different possible names
        possible_names = ['points_adams'] #, 'points_adams_efficient', 'points_adams_ear_aware']
        for name in possible_names:
            adam_path = os.path.join(cache_dir, f'{name}.pt')
            if os.path.exists(adam_path):
                adam_filename = name
                break
        
        if adam_filename is None:
            raise FileNotFoundError(f"No Adam points file found in {cache_dir}. Tried: {possible_names}")
    else:
        adam_path = os.path.join(cache_dir, f'{adam_filename}.pt')
        
    if fine_filename is None:
        # Try different possible names
        possible_names = ['points_fine'] #, 'points_fine_efficient', 'points_fine_ear_aware']
        for name in possible_names:
            fine_path = os.path.join(cache_dir, f'{name}.pt')
            if os.path.exists(fine_path):
                fine_filename = name
                break
                
        if fine_filename is None:
            raise FileNotFoundError(f"No fine points file found in {cache_dir}. Tried: {possible_names}")
    else:
        fine_path = os.path.join(cache_dir, f'{fine_filename}.pt')
    
    # Load Adam points
    adam_path = os.path.join(cache_dir, f'{adam_filename}.pt')
    if not os.path.exists(adam_path):
        raise FileNotFoundError(f"Adam points file not found: {adam_path}")
        
    adam_data = torch.load(adam_path, weights_only=False)
    
    # Handle different data formats
    if isinstance(adam_data, dict) and 'points' in adam_data:
        points_adam = adam_data['points']
    else:
        points_adam = adam_data  # Assume data is directly the points
        
    print(f"Loaded {len(points_adam)} Adam points from: {adam_path}")
    
    # Create Adam dataset and dataloader
    dataset_adam = OutsidePointsDataset(points_adam)
    dataloader_adam = DataLoader(
        dataset_adam,
        batch_size=config['adam']['batch_size'],
        shuffle=True,
        num_workers=config['adam'].get('num_workers', 1),
        pin_memory=config['adam'].get('pin_memory', False)
    )
    
    # Load fine-tuning points
    fine_path = os.path.join(cache_dir, f'{fine_filename}.pt')
    if not os.path.exists(fine_path):
        raise FileNotFoundError(f"Fine-tuning points file not found: {fine_path}")
        
    fine_data = torch.load(fine_path, weights_only=False)
    
    # Handle different data formats
    if isinstance(fine_data, dict) and 'points' in fine_data:
        points_fine = fine_data['points']
    else:
        points_fine = fine_data  # Assume data is directly the points
        
    print(f"Loaded {len(points_fine)} fine-tuning points from: {fine_path}")
    
    # Create fine-tuning dataset and dataloader
    dataset_fine = OutsidePointsDataset(points_fine)
    dataloader_fine = DataLoader(
        dataset_fine,
        batch_size=config['fine']['batch_size'],
        shuffle=True,
        num_workers=config['fine'].get('num_workers', 0),
        pin_memory=config['fine'].get('pin_memory', False)
    )
    
    print("3D dataloaders loaded successfully!")
    return {
        'adam': dataloader_adam,
        'fine': dataloader_fine
    }


