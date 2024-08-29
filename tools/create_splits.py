import glob
import os
from tifffile import imread, imsave
from tqdm import tqdm


if __name__ == "__main__":
    # bright field = train_A (input)
    # dapi = train_B (ground truth)
    splits_dir = "/hpc/scratch/rdip1/mar83345/diffusion_data/splits/"
    data_dir = "/hpc/scratch/rdip1/mar83345/diffusion_data/"