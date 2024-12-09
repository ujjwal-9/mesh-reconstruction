# Mesh Reconstruction
Geometric alignment, surface reconstruction.

## Code Execution
```bash
Usage: ./run.sh [options] <input_file>

Options:
  -i, --input FILE       Input point cloud file (PLY format)
  -m, --method METHOD    Reconstruction method (default: poisson)
                         Available methods: poisson, hole_preserve
  -o, --output DIR       Output directory (default: outputs)
  -h, --help             Show this help message

Example:
  ./run.sh -i point_clouds/shoe_pc.ply -m poisson -o outputs/shoe_pc.ply
```

## Results

Results can be found here: https://github.com/ujjwal-9/mesh-reconstruction/tree/main/outputs

## Approach

### Step 1: Floor Detection and Reorientation

#### 1. **Loading the Point Cloud**
- The input point cloud is loaded using Open3D. This step ensures that the data is properly formatted and ready for processing.
- Basic statistics such as the number of points are checked to verify the quality of the input.

#### 2. **Downsampling**
- The point cloud is voxel-downsampled to reduce its density and computational load while preserving its geometric integrity. This ensures efficient processing in subsequent steps.
- Due to resource constraints, I had to reduce number of points, this lead to certain mesh reconstruction failure in some objects like glove. 

#### 3. **Outlier Removal**
- Statistical outlier removal is applied to filter noise and artifacts, improving the overall quality of the point cloud for plane detection.

#### 4. **Floor Plane Detection**
- Using RANSAC-based segmentation, the floor plane is detected. This method identifies the largest planar region within the point cloud, making it ideal for detecting flat surfaces like floors.
- The detected plane is defined by its normal vector and the distance from the origin.

#### 5. **Reorientation of the Floor**
- The floor plane is reoriented to align with the XZ plane (`y = 0`).
- A rotation matrix is computed to align the detected plane's normal with the desired axis (e.g., aligning the floor's normal to `[0, -1, 0]`).
- After rotation, the point cloud is translated so that the floor passes through the origin, ensuring it becomes the reference point for the scene.
- Iterative centroid adjustments ensure that the point cloud is centered at the origin, with the floor aligned perfectly on the YZ plane.

---

### Step 2: Surface Reconstruction

#### 1. **Normals Estimation**
- Point normals are estimated for the point cloud, ensuring consistent orientation and lighting effects during rendering.
- Normals are oriented consistently using the tangent plane method to maintain visual coherence.

#### 2. **Surface Reconstruction via Poisson Method**
- Poisson surface reconstruction is applied to generate a smooth, watertight mesh from the point cloud. This method transforms the discrete points into a continuous surface representation, improving visual realism.
- Density thresholds are used to preserve critical details while reducing unnecessary artifacts.

#### 3. **Hole Preservation**
- To address sparse regions in the point cloud, advanced hole-preservation techniques are applied. This involves selectively removing low-density vertices to maintain the topology and visual continuity of the scene.

#### 4. **Smoothing**
- The reconstructed surface undergoes iterative smoothing to eliminate jagged edges and ensure a visually appealing representation.

#### 5. **Double-Sided Mesh Generation**
- A double-sided mesh is created to enhance rendering from all viewing angles. This involves duplicating and offsetting the mesh’s vertices slightly to ensure consistent visualization without z-fighting.

---

### Step 3: Unit Testing

#### 1. **Random Transformations**
- Random transformations, including rotations and translations, are applied to the input point cloud to test the robustness of the alignment and reconstruction pipeline.
- These transformations simulate real-world scenarios where point clouds might not be aligned or are partially distorted.

#### 2. **Validation of Floor Reorientation**
- The transformed point cloud is passed through the pipeline, and the alignment of the floor plane is verified.
- Metrics such as the alignment of the floor normal to `[0, -1, 0]` and the position of the centroid at the origin are used as validation criteria.

#### 3. **Comparison of Results**
- Before-and-after visualizations of the point cloud ensure that the transformations and reconstructions were applied correctly.
- The aligned point cloud and reconstructed mesh are evaluated for consistency and visual fidelity.
---

## Where the Approach Works Well

- **Scenes with Clear Planar Regions**  
  The approach is highly effective in scenes where the floor or other planar surfaces are well-represented, enabling accurate detection and alignment.

- **Dense Point Clouds**  
  Reconstruction methods like Poisson work best on dense point clouds, producing smooth and visually appealing surfaces.

- **Controlled Transformations**  
  The alignment pipeline performs well under moderate rotations and translations, consistently reorienting the floor and centering the scene.

---

## Where the Approach May Struggle

- **Noisy or Sparse Point Clouds**  
  High noise levels or sparse regions can lead to incorrect plane detection or artifacts in the reconstructed surface.

- **Complex Scenes**  
  In scenes with multiple planar regions or overlapping objects, the floor detection may fail, leading to misalignment.

---

### Ideas for Improvement

#### 1. **Advanced Plane Detection**
- Incorporate multi-scale or machine-learning-based segmentation methods to improve robustness in noisy or complex scenes.

#### 2. **Hybrid Reconstruction**
- Combine Poisson reconstruction with other techniques like Delaunay triangulation to better preserve fine details and handle sparse regions.
---

### Conclusion

The approach effectively addressed the problem by aligning the floor plane, reconstructing smooth surfaces, and validating the robustness of the solution. While the methodology performed well under controlled conditions, enhancements in plane detection, reconstruction, and alignment could further improve its applicability to diverse and challenging point cloud datasets.

## Outputs

![glove](https://github.com/user-attachments/assets/492f3fce-acc6-44ab-a229-478d0be4cedd)
Thumb Missing, voxelization may have reduced some object complexity here.

![image](https://github.com/user-attachments/assets/088e158c-893a-4509-9789-d5b514069239)
Buld holder didn't get picked up for mesh contruction, maybe due to sparse points there, I tried adjusting the density but I didn't run thorogh experiements.


![image](https://github.com/user-attachments/assets/f97e3169-e942-4148-a680-b3c3d45e6eee)
Pretty good, even depression in pillow is captured and smoothly rendered.

![image](https://github.com/user-attachments/assets/affcc230-54b1-4388-b6cc-79338e5690b3)
Sparse points in tree branches got missed.

![image](https://github.com/user-attachments/assets/06ad4836-b84a-441b-9005-7c55399b3bbf)
Even nike logo is picked up here.

![image](https://github.com/user-attachments/assets/f901bea7-727f-4e34-a1ae-32df01ee71e1)
Though it did a pretty good job in maitaining the structre of its legs but part of it missing again due to downsampling.

![image](https://github.com/user-attachments/assets/ccc38432-d387-48dc-aa30-eab0be109a11)
Good result. vase is full and has mouth at the top which wasn't covered up.




