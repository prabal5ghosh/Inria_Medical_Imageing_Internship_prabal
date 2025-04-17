
import nibabel as nib
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator
# import numpy as np
# import os
# import matplotlib.pyplot as plt


input_path_1 = Path("/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/mri.nii.gz")



output_path_total_mr = Path("/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/segmentations_total_mr")
output_path_body_mr = Path("/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/segmentations_body_mr")
output_path_vertebrae_mr = Path("/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/segmentations_vertebrae_mr")
output_path_liver_mr = Path("/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/segmentations_liver_mr")
output_path_appendicular_bones_mr = Path("/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/segmentations_appendicular_bones_mr")
output_path_tissue_types_mr = Path("/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/segmentations_tissue_types_mr")
output_path_thigh_shoulder_muscles_mr = Path("/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/segmentations_thigh_shoulder_muscles_mr")
output_path_face_mr = Path("/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/segmentations_face_mr")




img = nib.load(input_path_1).get_fdata()
print(img.shape)
print(f"*****The .nii files are stored in memory as numpy's: {type(img)}.*****")






if __name__ == "__main__":


    print("********Total MRI Segmentation********")

    # Segment the first MRI image
    print(f"******Segmenting {input_path_1}*******")
    
    # totalsegmentator(input=input_path_1, output=output_path_1,device='gpu', task="total_mr", roi_subset= ["lung_left", "lung_right"])
    totalsegmentator(input=input_path_1, output=output_path_total_mr,device='gpu', task="total_mr")



    print(f"***********Segmentation completed for {input_path_1}. Results saved to {output_path_total_mr}.**********")

    print()
    print()
    print()
    print()
    print()
    print()

    print("*********body mri segmentation*********")

    # Segment the first MRI image
    print(f"******Segmenting {input_path_1}*******")
    
    # totalsegmentator(input=input_path_1, output=output_path_1,device='gpu', task="total_mr", roi_subset= ["lung_left", "lung_right"])
    totalsegmentator(input=input_path_1, output=output_path_body_mr,device='gpu', task="body_mr")



    print(f"***********Segmentation completed for {input_path_1}. Results saved to {output_path_body_mr}.**********")
    print()
    print()
    print()
    print()
    print()
    print()


    print("**************vertebrae mr segmentation**************")

      # Segment the first MRI image
    print(f"******Segmenting {input_path_1}*******")
    
    # totalsegmentator(input=input_path_1, output=output_path_1,device='gpu', task="total_mr", roi_subset= ["lung_left", "lung_right"])
    totalsegmentator(input=input_path_1, output=output_path_vertebrae_mr,device='gpu', task="vertebrae_mr")



    print(f"***********Segmentation completed for {input_path_1}. Results saved to {output_path_vertebrae_mr}.**********")


    print()
    print()
    print()
    print()
    print()
    print()

    print("**************liver segmentations mr**************")

      # Segment the first MRI image
    print(f"******Segmenting {input_path_1}*******")
    
    # totalsegmentator(input=input_path_1, output=output_path_1,device='gpu', task="total_mr", roi_subset= ["lung_left", "lung_right"])
    totalsegmentator(input=input_path_1, output=output_path_liver_mr,device='gpu', task="liver_segments_mr")



    print(f"***********Segmentation completed for {input_path_1}. Results saved to {output_path_liver_mr}.**********")

    print()
    print()
    print()
    print()
    print()
    print()


    print("**************appendicular bones mr**************")

      # Segment the first MRI image
    print(f"******Segmenting {input_path_1}*******")
    
    # totalsegmentator(input=input_path_1, output=output_path_1,device='gpu', task="total_mr", roi_subset= ["lung_left", "lung_right"])
    totalsegmentator(input=input_path_1, output=output_path_appendicular_bones_mr,device='gpu', task="appendicular_bones_mr")



    print(f"***********Segmentation completed for {input_path_1}. Results saved to {output_path_appendicular_bones_mr}.**********")


    print()
    print()
    print()
    print()
    print()
    print()

    print("**************tissue types mr**************")

      # Segment the first MRI image
    print(f"******Segmenting {input_path_1}*******")
    
    # totalsegmentator(input=input_path_1, output=output_path_1,device='gpu', task="total_mr", roi_subset= ["lung_left", "lung_right"])
    totalsegmentator(input=input_path_1, output=output_path_tissue_types_mr,device='gpu', task="tissue_types_mr")



    print(f"***********Segmentation completed for {input_path_1}. Results saved to {output_path_tissue_types_mr}.**********")

    print()
    print()
    print()
    print()
    print()
    print()

    print("**************thigh shoulder muscles mr**************")

      # Segment the first MRI image
    print(f"******Segmenting {input_path_1}*******")
    
    # totalsegmentator(input=input_path_1, output=output_path_1,device='gpu', task="total_mr", roi_subset= ["lung_left", "lung_right"])
    totalsegmentator(input=input_path_1, output=output_path_thigh_shoulder_muscles_mr,device='gpu', task="thigh_shoulder_muscles_mr")



    print(f"***********Segmentation completed for {input_path_1}. Results saved to {output_path_thigh_shoulder_muscles_mr}.**********")
    
    print()
    print()
    print()
    print()
    print()
    print()


    print("**************face mr**************")


  # Segment the first MRI image
    print(f"******Segmenting {input_path_1}*******")
    
    # totalsegmentator(input=input_path_1, output=output_path_1,device='gpu', task="total_mr", roi_subset= ["lung_left", "lung_right"])
    totalsegmentator(input=input_path_1, output=output_path_face_mr,device='gpu', task="face_mr")



    print(f"***********Segmentation completed for {input_path_1}. Results saved to {output_path_face_mr}.**********")

    print("**************all segmentations completed**************")