import os
import copy
import numpy as np
import open3d as o3d

np.random.seed(2024)


def load_point_cloud(file_path):
    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    print("Point cloud loaded.")
    print(f"Number of points: {len(pcd.points)}")

    # Check if point cloud is empty
    if len(pcd.points) == 0:
        print("Point cloud is empty. Please check the input file.")
        return None

    return pcd


def downsample_point_cloud(pcd, **kwargs):
    down_pcd = pcd.voxel_down_sample(voxel_size=kwargs["voxel_size"])
    return down_pcd


def estimate_normals(pcd, **kwargs):
    if len(pcd.points) == 0:
        print("Point cloud is empty. Cannot estimate normals.")
        return
    try:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=kwargs["radius"], max_nn=kwargs["max_nn"]
            )
        )
        # Check if normals are computed
        if not pcd.has_normals():
            print("Normals were not estimated.")
            return
        # Orient normals consistently
        pcd.normalize_normals()
        pcd.orient_normals_towards_camera_location(
            camera_location=np.array([10, 10.0, 10.0])
        )
        pcd.orient_normals_consistent_tangent_plane(k=25)
    except Exception as e:
        print(f"Error estimating normals: {e}")


def remove_statistical_outliers(pcd, **kwargs):
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=kwargs["nb_neighbors"], std_ratio=kwargs["std_ratio"]
    )
    inlier_cloud = pcd.select_by_index(ind)
    return inlier_cloud


def detect_floor_plane(pcd, **kwargs):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=kwargs["distance_threshold"],
        ransac_n=kwargs["ransac_n"],
        num_iterations=kwargs["num_iterations"],
    )
    [a, b, c, d] = np.round(plane_model, 2)
    if len(inliers) == 0:
        print("No plane detected.")
        return None, None
    return plane_model, inliers


def detect_floor_plane_with_normal(pcd, expected_normal, angle_threshold=np.pi / 4):
    max_iterations = 3
    for _ in range(max_iterations):
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        if len(inliers) == 0:
            break
        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal)
        angle = np.arccos(np.clip(np.dot(normal, expected_normal), -1.0, 1.0))
        if angle < angle_threshold:
            return plane_model, inliers
        # Remove the detected plane and continue
        pcd = pcd.select_by_index(inliers, invert=True)
    return None, None


def compute_rotation_matrix_from_vectors(v1, v2):
    """Compute rotation matrix to align v1 to v2."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross_product = np.cross(v1, v2)
    dot_product = np.dot(v1, v2)
    skew_symmetric = np.array(
        [
            [0, -cross_product[2], cross_product[1]],
            [cross_product[2], 0, -cross_product[0]],
            [-cross_product[1], cross_product[0], 0],
        ]
    )
    rotation_matrix = (
        np.eye(3)
        + skew_symmetric
        + np.dot(skew_symmetric, skew_symmetric)
        * ((1 - dot_product) / (np.linalg.norm(cross_product) ** 2))
    )
    return rotation_matrix


def align_floor_to_xz(pcd, plane_model, **kwargs):
    # Make a copy of the point cloud to avoid modifying the original
    pcd = copy.deepcopy(pcd)

    # Compute the initial normal of the detected plane
    plane_normal = np.array(plane_model[:3])
    plane_normal = plane_normal / np.linalg.norm(
        plane_normal
    )  # Normalize the normal vector
    desired_normal = np.array([0.0, -1.0, 0.0])  # Align to the XZ plane

    # Compute rotation matrix to align the plane's normal to the desired normal
    rotation_matrix = compute_rotation_matrix_from_vectors(plane_normal, desired_normal)

    # Apply the rotation to the point cloud
    rotated_pcd = pcd.rotate(rotation_matrix, center=(0, 0, 0))

    # Re-detect the plane after rotation
    plane_model_aligned, inliers = detect_floor_plane(rotated_pcd, **kwargs)
    if plane_model_aligned is None:
        raise ValueError("No floor plane detected after rotation.")
    [a, b, c, d] = plane_model_aligned
    if kwargs["log"]:
        print(
            f"Plane equation after rotation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0"
        )

    # Translate the plane to pass through the origin (set d = 0)
    plane_normal = np.array([a, b, c])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # Normalize again
    translation_to_origin = -d * plane_normal  # Translate along the normal vector
    rotated_pcd.translate(translation_to_origin)

    # Compute the centroid of the entire point cloud after alignment
    centroid = np.mean(np.asarray(rotated_pcd.select_by_index(inliers).points), axis=0)
    if kwargs["log"]:
        print(f"Centroid before translation: {centroid}")

    # Translate the point cloud so its centroid lies at the origin
    transformed_pcd = rotated_pcd.translate(-centroid)

    # Verify the alignment
    final_plane_model, final_plane_inliers = detect_floor_plane(
        transformed_pcd, **kwargs
    )
    [a, b, c, d] = final_plane_model

    # Construct the final transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix  # Add rotation
    total_translation = translation_to_origin  # Combine translations

    tries = 0
    final_centroid = [1, 1, 1]
    while not np.allclose(final_centroid, np.zeros_like(final_centroid), atol=1e-6):
        final_centroid = np.mean(
            np.asarray(transformed_pcd.select_by_index(final_plane_inliers).points),
            axis=0,
        )
        total_translation -= final_centroid
        transformed_pcd.translate(-final_centroid)
        if kwargs["log"]:
            print(f"Adjusted centroid after translation pass {tries}: {final_centroid}")
        tries += 1

    transformation_matrix[:3, 3] = total_translation  # Add translation

    [a, b, c, d] = final_plane_model
    if kwargs["log"]:
        print(f"Final plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # Check if alignment is correct
    if (
        np.allclose(a, 0, atol=1e-2)
        and np.allclose(b, 1, atol=1e-2)
        and np.allclose(c, 0, atol=1e-2)
    ):
        print(
            "Alignment successful: The plane is aligned with the XZ plane, and the centroid is at the origin."
        )
    else:
        print("Alignment requires further refinement.")

    return (
        transformed_pcd,
        transformation_matrix,
        final_plane_model,
        final_plane_inliers,
    )


def controlled_poisson_reconstruction(point_cloud, **kwargs):
    # Ensure normals are computed and oriented
    if not point_cloud.has_normals():
        point_cloud.estimate_normals()
        point_cloud.orient_normals_consistent_tangent_plane(100)

    # Perform initial Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=kwargs["depth"]
    )

    # If preventing hole filling is desired
    if kwargs["prevent_hole_filling"]:
        # Compute density statistics
        density_mean = np.mean(densities)
        density_std = np.std(densities)

        # Conservative density threshold to preserve surface details
        threshold = density_mean - (kwargs["min_density_threshold"] * density_std)

        # Remove vertices with very low density carefully
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

    # Additional mesh cleaning
    mesh.remove_degenerate_triangles()

    return mesh


def advanced_hole_preservation(point_cloud, **kwargs):
    # Ensure normals are computed and oriented
    if not point_cloud.has_normals():
        point_cloud.estimate_normals()
        point_cloud.orient_normals_consistent_tangent_plane(100)

    # Perform Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=kwargs["depth"]
    )

    # Compute density quantiles for selective removal
    lower_quantile = kwargs["min_density_threshold"]

    # Remove only the lowest density vertices
    vertices_to_remove = densities < np.quantile(densities, lower_quantile)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Preserve mesh topology
    mesh.remove_degenerate_triangles()

    return mesh


def add_random_transformation(pcd):
    angle = np.random.uniform(-np.pi / 10, np.pi / 10)  # Limit to +/- 18 degrees
    axis = np.random.uniform(-1, 1, 3)
    axis = axis / np.linalg.norm(axis)
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    t = np.random.uniform(-0.2, 0.2, 3)  # Limit translation

    transformed_pcd = copy.deepcopy(pcd)
    transformed_pcd.rotate(R, center=(0, 0, 0))
    transformed_pcd.translate(t)

    return transformed_pcd


def sample_points_from_mesh(mesh, number_of_points):
    # Uniformly sample points from the mesh surface
    sampled_pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    return sampled_pcd


def apply_transformation_to_point_cloud(pcd, transformation_matrix):
    transformed_pcd = copy.deepcopy(pcd)
    transformed_pcd.transform(transformation_matrix)
    return transformed_pcd


def unit_test_floor_alignment(file_path, **kwargs):
    print("Starting unit test for floor alignment...")
    original_pcd = load_point_cloud(file_path)
    if original_pcd is None:
        return False

    # Apply random transformations
    transformed_pcd = add_random_transformation(original_pcd)
    print("Random transformations applied.")

    # Process the transformed point cloud
    # Downsample
    pcd_down = downsample_point_cloud(transformed_pcd, **kwargs)
    # Remove outliers
    pcd_clean = remove_statistical_outliers(pcd_down, **kwargs)
    # Estimate normals
    estimate_normals(pcd_clean, **kwargs)
    # Detect floor plane
    plane_model, inliers = detect_floor_plane(pcd_clean, **kwargs)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.1f}x + {b:.1f}y + {c:.1f}z + {d:.1f} = 0")
    if plane_model is None:
        print("No floor plane detected in unit test.")
        return False

    # Align the floor to XZ plane
    aligned_pcd, transformation_matrix, plane_model_aligned, inliers_aligned = (
        align_floor_to_xz(pcd_clean, plane_model)
    )

    if plane_model_aligned is None:
        print("No floor plane detected after alignment in unit test.")

    [a, b, c, d] = plane_model_aligned
    if (
        np.allclose(a, 0, atol=1e-2)
        and np.allclose(b, 1, atol=1e-2)
        and np.allclose(c, 0, atol=1e-2)
    ):
        print("** Floor plane aligned correctly in unit test.")
    else:
        print("-- Floor plane is not aligned correctly in unit test.")

    # Also, check if the centroid of the floor plane is close to the origin
    floor_points = aligned_pcd.select_by_index(inliers_aligned)
    centroid = np.mean(np.asarray(floor_points.points), axis=0)
    print(f"Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
    if np.allclose(centroid, np.zeros_like(centroid), atol=1e-2):
        print("** Floor is centered at the origin in unit test.")
    else:
        print("-- Floor is not centered at the origin in unit test.")
        return False

    print("Unit test passed: The floor plane was correctly aligned.")
    return True


def create_double_sided_mesh(input_mesh):
    """
    Convert an existing Open3D mesh to a double-sided mesh while preserving colors.

    Args:
        input_mesh (o3d.geometry.TriangleMesh): Input mesh to be converted

    Returns:
        o3d.geometry.TriangleMesh: Double-sided mesh with original colors
    """
    # Ensure input is a triangle mesh
    if not isinstance(input_mesh, o3d.geometry.TriangleMesh):
        raise TypeError("Input must be an Open3D TriangleMesh")

    # Check if the mesh has colors
    has_colors = input_mesh.has_vertex_colors()

    # Get vertices, triangles, and colors
    vertices = np.asarray(input_mesh.vertices)
    triangles = np.asarray(input_mesh.triangles)

    # Create a new mesh
    double_sided_mesh = o3d.geometry.TriangleMesh()

    # Prepare colors if they exist
    if has_colors:
        colors = np.asarray(input_mesh.vertex_colors)

    # Small offset to prevent z-fighting
    offset = 0.0001

    # Compute vertex normals
    input_mesh.compute_vertex_normals()
    front_normals = np.asarray(input_mesh.vertex_normals)

    # Create two sets of vertices
    front_vertices = vertices
    back_vertices = vertices.copy()

    # Slightly offset back vertices along their normal direction
    back_vertices += front_normals * offset

    # Combine vertices
    combined_vertices = np.vstack((front_vertices, back_vertices))

    # Duplicate and adjust triangles
    front_triangles = triangles
    back_triangles = triangles + len(front_vertices)
    back_triangles = back_triangles[:, ::-1]  # Invert winding order

    # Combine triangles
    full_triangles = np.vstack((front_triangles, back_triangles))

    # Prepare colors for double-sided mesh
    if has_colors:
        # Duplicate colors for both vertex sets
        combined_colors = np.vstack((colors, colors))

    # Set vertices and triangles
    double_sided_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
    double_sided_mesh.triangles = o3d.utility.Vector3iVector(full_triangles)

    # Reapply colors if they existed
    if has_colors:
        double_sided_mesh.vertex_colors = o3d.utility.Vector3dVector(combined_colors)

    # Recompute normals
    double_sided_mesh.compute_vertex_normals()
    double_sided_mesh.normalize_normals()

    return double_sided_mesh


def main(file_path, out_path, **kwargs):
    # Load the point cloud
    original_pcd = load_point_cloud(file_path)
    if original_pcd is None:
        return
    file = os.path.basename(file_path).split(".")[0]
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    pcd = copy.deepcopy(original_pcd)
    print(f"Number of points in pointcloud: {len(pcd.points)}")

    # Downsample the point cloud
    pcd = downsample_point_cloud(pcd, **kwargs)
    print("Point cloud downsampled.")
    print(f"Number of points after downsampling: {len(pcd.points)}")

    # Remove statistical outliers
    pcd = remove_statistical_outliers(pcd, **kwargs)
    print("Statistical outliers removed.")
    print(f"Number of points after outlier removal: {len(pcd.points)}")

    # Estimate normals
    estimate_normals(pcd, **kwargs)
    print("Normals estimated.")

    # Check if normals were estimated
    if not pcd.has_normals():
        print("Normals were not estimated after estimation step.")
        return
    else:
        print("Normals are present in the point cloud.")

    # Detect the floor plane
    plane_model, inliers = detect_floor_plane(pcd, **kwargs)
    print("Floor plane detected.")

    # Align the floor to XZ plane and center it at the origin
    aligned_pcd, transformation_matrix, plane_model, inliers = align_floor_to_xz(
        pcd, plane_model, **kwargs
    )
    print("Point cloud aligned.")

    # Detect the floor plane
    print("Plane equation after alignment.")
    plane_model, inliers = detect_floor_plane(aligned_pcd, **kwargs)

    # Save the aligned point cloud and mesh
    pc_aligned_path = os.path.join(out_path, f"{file}_aligned.ply")
    o3d.io.write_point_cloud(pc_aligned_path, aligned_pcd)
    print("Aligned point cloud saved.")

    # Initial surface reconstruction to get a preliminary mesh
    # if len(aligned_pcd.points) < 12000:
    #     original_pcd = apply_transformation_to_point_cloud(
    #         original_pcd, transformation_matrix
    #     )
    #     aligned_pcd = original_pcd

    if kwargs["reconstruction_method"] == "poisson":
        mesh = controlled_poisson_reconstruction(aligned_pcd, **kwargs["poisson"])
    elif kwargs["reconstruction_method"] == "hole_preserve":
        mesh = advanced_hole_preservation(aligned_pcd, **kwargs["hole_preserve"])
    print("Surface reconstruction completed.")
    mesh = mesh.filter_smooth_simple(number_of_iterations=kwargs["smooth_itr"])
    print("Smoothing mesh completed.")
    mesh.compute_vertex_normals()
    vertex_normals = np.asarray(mesh.vertex_normals)
    inverted_normals = -vertex_normals
    # Create a new vertex normal array with both original and inverted normals
    double_sided_normals = np.vstack((vertex_normals, inverted_normals))
    # Modify the mesh to include both sets of normals
    mesh.vertex_normals = o3d.utility.Vector3dVector(double_sided_normals)
    mesh = create_double_sided_mesh(mesh)

    # Save the final mesh
    mesh_path = os.path.join(out_path, f"{file}_mesh.obj")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print("Reconstructed mesh saved.")
    return mesh, aligned_pcd
