import argparse
import os
from uuid import uuid4

import cv2
import nibabel
import numpy as np
from tqdm import tqdm
import pydicom


def extract_int(input_string):
    num = ''.join(c for c in input_string if c.isdigit())
    return int(num) if num else 0


def sort_list_of_strings_by_int_key(input_list):
    return sorted(input_list, key=extract_int)


def read_segmentation_from_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = np.where(image == 255, 1, 0)
    return binary_mask


def apply_orientation(np_ndarr):
    # TODO: this information should come from the DICOM file and the transformation should not be hardcoded
    rotated = np.rot90(np_ndarr, k=1)  # Counterclockwise 90-degree rotation
    flipped = np.fliplr(rotated)       # Horizontal flip
    return flipped


def construct_data_np_arr(sample_dicoms_dir):
    dicom_files = os.listdir(sample_dicoms_dir)
    dicom_files = sort_list_of_strings_by_int_key(dicom_files)

    slices_np_arrs = []
    for dicom_file_name in dicom_files:
        dicom_file_path = os.path.join(sample_dicoms_dir, dicom_file_name)
        ds = pydicom.dcmread(dicom_file_path)
        pixel_array = ds.pixel_array
        pixel_array = apply_orientation(pixel_array)
        slices_np_arrs.append(pixel_array)
    
    data_np_arr = np.stack(slices_np_arrs, axis=2)
    return data_np_arr


def construct_segmentation_np_arr(sample_segmentations_dir):
    segmentation_image_files = os.listdir(sample_segmentations_dir)
    segmentation_image_files = sort_list_of_strings_by_int_key(segmentation_image_files)

    slice_segmentations_np_arrs = []
    for segmentation_image_file_name in segmentation_image_files:
        segmentation_image_file_path = os.path.join(sample_segmentations_dir, segmentation_image_file_name)
        segmentation_array = read_segmentation_from_image(segmentation_image_file_path)
        slice_segmentations_np_arrs.append(segmentation_array)

    segmentation_np_arr = np.stack(slice_segmentations_np_arrs, axis=2)
    segmentation_np_arr = segmentation_np_arr.astype(np.uint16)
    return segmentation_np_arr


def save_np_ndarr_to_file(args, file_path_without_extension, np_ndarr):
    if args.save_format == "nii_gz_no_metadata":
        nii_image = nibabel.Nifti1Image(np_ndarr, affine=np.eye(4))
        nibabel.save(nii_image, file_path_without_extension + ".nii.gz")
        return
    
    if args.save_format == "npy_gz":
        with gzip.open(file_path_without_extension + ".npy.gz", 'wb') as f:
            np.save(f, np_ndarr)
        return
    
    if args.save_format == "npy":
        np.save(file_path_without_extension + ".npy", np_ndarr)
        return
    
    raise ValueError("Unknown save format.")


def preprocess_sample_dicoms_and_ground_truths(args, sample_dicoms_dir, sample_ground_truths_dir):
    data_np_arr = construct_data_np_arr(sample_dicoms_dir)
    segmentation_np_arr = construct_segmentation_np_arr(sample_ground_truths_dir)

    sample_id = str(uuid4())
    sample_save_dir = os.path.join(args.destination_dir, sample_id)
    os.makedirs(sample_save_dir, exist_ok=True)

    data_save_path_without_extension = os.path.join(sample_save_dir, "data")
    segmentation_save_path_without_extension = os.path.join(sample_save_dir, "segmentation")
    
    save_np_ndarr_to_file(args, data_save_path_without_extension, data_np_arr)
    save_np_ndarr_to_file(args, segmentation_save_path_without_extension, segmentation_np_arr)

    exit()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process source and destination directories with optional gzip saving.")
    parser.add_argument("--source_dir", type=str, default=os.path.dirname(__file__), help="Path to the source directory (default: ./source)")
    parser.add_argument("--destination_dir", type=str, required=True, help="Path to the destination directory (default: ./destination)")
    parser.add_argument("--save_format", type=str, default="nii_gz_no_metadata", help="TODO: add help")
    return parser.parse_args()


def main():
    args = parse_arguments()
    # print(f"Source Directory: {args.source_dir}")
    # print(f"Destination Directory: {args.destination_dir}")
    # print(f"Save as Gzip: {args.save_gzip}")

    for split_dir_name in ["Train_Sets", "Test_Sets"]:
        print(f"Processing {split_dir_name}")

        split_dir = os.path.join(args.source_dir, split_dir_name)

        # MR and CT have to be processed differently as there folder structures are different
        
        # Process CT
        ct_dir = os.path.join(split_dir, "CT")
        for sample_name in tqdm(os.listdir(ct_dir)):
            sample_dir = os.path.join(ct_dir, sample_name)
            sample_dicoms_dir = os.path.join(sample_dir, "DICOM_anon")
            sample_ground_truths_dir = os.path.join(sample_dir, "Ground")

            preprocess_sample_dicoms_and_ground_truths(
                args,
                sample_dicoms_dir,
                sample_ground_truths_dir
            )

            # for dicom_file_name in os.listdir(sample_dicoms_dir):
            #     dicom_file_path = os.path.join(sample_dicoms_dir, dicom_file_name)
            #     extract_dicom_info(dicom_file_path)









        # for modality in ["CT", "MR"]:
        #     split_modality_dir = os.path.join(split_dir, modality)

        #     for sample_name in os.listdir(split_modality_dir):
        #         sample_dir = os.path.join(split_modality_dir, sample_name)




    # train_sets_dir = os.path.join(args.source_dir, "Train_Sets")



if __name__ == "__main__":
    main()