config = {
    "reconstruction_method": "poisson",
    "poisson": {"depth": 7, "prevent_hole_filling": True, "min_density_threshold": 0.1},
    "hole_preserve": {"depth": 7, "hole_preservation_level": 0.002},
    "smooth_itr": 3,
    "voxel_size": 0.005,
    "radius": 0.05,
    "max_nn": 200,
    "nb_neighbors": 10,
    "std_ratio": 0.02,
    "distance_threshold": 0.01,
    "ransac_n": 5,
    "num_iterations": 1000,
    "log": True,
}

thresholds = {
    "chair_pc": 0.5,
    "craddle_pc": 1.5,
    "glove_pc": 0.1,
    "lamp_pc": 0.1,
    "pillow_pc": 1,
    "plant_pc": 1,
    "shoe_pc": 2.5,
    "shoe2_pc": 0.1,
    "stool_pc": 1,
    "vase_pc": 3,
}
