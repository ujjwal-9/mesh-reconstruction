# Mesh Reconstruction

### Code Execution
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