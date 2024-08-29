import h5py 
import skimage.io as io
import glob
import os
from tifffile import imread, imsave
from tqdm import tqdm
import random

BRIGHT_CHANNEL = "Brightfield_em0_ex740"
DAPI_CHANNEL = "DAPI_em456_ex375"
SPLITS = [0.7, 0.2] # train, val

def convert_to_tiff(h5_files, bright_channel, nuclei_channel, bright_outdir, nuclei_outdir):
    for h5_path in tqdm(h5_files):
        h5 = h5py.File(h5_path,"r")
        basename = os.path.basename(h5_path)        
        #get intersection of FOVs between bright field and DAPI
        fovs = list(set(h5['Brightfield_em0_ex740'].keys()).intersection(set(h5['DAPI_em456_ex375'].keys())))
        for fov in fovs:
            new_basename = basename+f"_{fov}.tiff"
            bright_img_path = os.path.join(bright_outdir,new_basename)
            nuclei_img_path = os.path.join(nuclei_outdir,new_basename)
            try:
                img_bright = h5[f"{bright_channel}/{fov}"][...]
                img_bright = img_bright.astype("float32")  
                        
                img_nuclei = h5[f"{nuclei_channel}/{fov}"][...]
                img_nuclei = img_nuclei.astype("float32")                

                imsave(bright_img_path,data=img_bright)                
                imsave(nuclei_img_path,data=img_nuclei)    
            except Exception as e:                 
                if os.path.exists(bright_img_path):
                    os.remove(bright_img_path)
                if os.path.exists(nuclei_img_path):
                    os.remove(nuclei_img_path)    
                print(str(e))
                      
        h5.close()
                

if __name__ == "__main__": 
    train_out_bright = "/hpc/scratch/rdip1/mar83345/diffusion_data/splits/train_A"
    train_out_dapi = "/hpc/scratch/rdip1/mar83345/diffusion_data/splits/train_B"
    val_out_bright = "/hpc/scratch/rdip1/mar83345/diffusion_data/splits/val_A"
    val_out_dapi = "/hpc/scratch/rdip1/mar83345/diffusion_data/splits/val_B"
    test_out_bright = "/hpc/scratch/rdip1/mar83345/diffusion_data/splits/test_A"
    test_out_dapi = "/hpc/scratch/rdip1/mar83345/diffusion_data/splits/test_B" 
    
    # h5_dir = "/hpc/aiml/upt/cell_imaging/stage_base/Squelette_data/IOC_cycle1/ICF/20221117094441" 
    # print(f"Converting files in {h5_dir}")
    # h5_files = glob.glob(os.path.join(h5_dir,"ICF.*.h5"))
    # convert_to_tiff(h5_files,BRIGHT_CHANNEL,DAPI_CHANNEL,output_bright,output_dapi)
    
    # h5_dir = "/hpc/aiml/upt/cell_imaging/stage_base/Squelette_data/IOC_cycle1/ICF/20221117094447"
    # print(f"Converting files in {h5_dir}") 
    # h5_files = glob.glob(os.path.join(h5_dir,"ICF.*.h5"))
    # convert_to_tiff(h5_files,BRIGHT_CHANNEL,DAPI_CHANNEL,output_bright,output_dapi)
    
    # h5_dir = "/hpc/aiml/upt/cell_imaging/stage_base/Squelette_data/IOC_cycle1/ICF/20221205173853"
    # print(f"Converting files in {h5_dir}") 
    # h5_files = glob.glob(os.path.join(h5_dir,"ICF.*.h5"))
    # convert_to_tiff(h5_files,BRIGHT_CHANNEL,DAPI_CHANNEL,output_bright,output_dapi)
    
    h5_dir = "/hpc/aiml/upt/cell_imaging/stage_base/Squelette_data/IOC_cycle1/ICF/20230116052445820280"
    print(f"Converting files in {h5_dir}") 
    all_h5_files = glob.glob(os.path.join(h5_dir,"ICF.*.h5"))
    random.shuffle(all_h5_files)
    num_files = len(all_h5_files)
    num_train = round(num_files*SPLITS[0])    
    num_val = round(num_files*SPLITS[1])
    
    train_h5_files = all_h5_files[0:num_train]
    val_h5_files = all_h5_files[num_train:num_train+num_val]
    test_h5_files = all_h5_files[(num_train+num_val):]   

    # convert train data
    print("__train__")
    convert_to_tiff(train_h5_files,BRIGHT_CHANNEL,DAPI_CHANNEL,train_out_bright,train_out_dapi)
    # convert val data
    print("__val__")
    convert_to_tiff(val_h5_files,BRIGHT_CHANNEL,DAPI_CHANNEL,val_out_bright,val_out_dapi)
    # convert test data
    print("__test__")
    convert_to_tiff(test_h5_files,BRIGHT_CHANNEL,DAPI_CHANNEL,test_out_bright,test_out_dapi)
    
    
    
    
    