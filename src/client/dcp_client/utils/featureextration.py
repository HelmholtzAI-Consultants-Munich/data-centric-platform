# Feature extraction per image triggered each time the user moves mask and image to curated in progress

from dcp_client.app import Postprocessing
import SimpleITK as sitk
from radiomics.shape2D import RadiomicsShape2D
import os
from skimage.io import imread
from skimage.measure import regionprops
from dcp_client.utils import utils
import numpy as np


class FeatureExtraction(Postprocessing):

    def __init__(self):
        super().__init__()
    

    # TODO after we test: init with app to access imagestorage
    def postprocess(self, from_directory, current_img):

        image_file = utils.get_path_stem(current_img)
        image_path = os.path.join(from_directory, current_img)
    
        seg_file = self._search_segs(from_directory, current_img)[0] #it should be only one now because we are in the curated folder
        #seg_file = self.image_storage.search_segs(from_directory, current_img)[0]

        seg_path = os.path.join(from_directory, seg_file)
        output_csv = os.path.join(from_directory, image_file + '.csv') 

        print(f"Seg path: {seg_path}")
        print(f"image_path: {image_path}")
        print(f'Will be saved into {output_csv}')

        image = imread(image_path)
        mask = imread(seg_path)[0]

        # Load image & mask
        image = imread(image_path)
        mask = imread(seg_path)[0]  # Assuming first channel

        # Use regionprops to get features
        regions = regionprops(mask, intensity_image=image)

        not_border_regions = []
        max_diameters = []
        region_stats = []

        for region in regions:
            
            # Exclude objects touching the image border using the updated function
            if not self._is_touching_border(region, mask.shape, border_buffer = self.border_buffer):
                not_border_regions.append(region)

            area = region.area
            perimeter = region.perimeter
            mean_intensity = region.mean_intensity
            eccentricity = region.eccentricity
            solidity = region.solidity
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            max_diameter = self._calculate_max_diameter(region.label, mask)
            max_diameters.append(max_diameter)

            region_stats.append({
                'label': region.label,
                'area': area,
                'perimeter': perimeter,
                'mean_intensity': mean_intensity,
                'eccentricity': eccentricity,
                'solidity': solidity,
                'circularity': circularity,
                'max_diameter': max_diameter
            })

        max_diameter_overall = np.max(max_diameters) if max_diameters else 0
        mean_area = np.mean([r.area for r in not_border_regions]) if not_border_regions else 0

         # Write the summary line
        with open(output_csv, 'w') as f:

            f.write(f"################ Summary ################\n")
            f.write(f"image file name: {image_file}\n")
            f.write(f"number of detected objects: {len(regions)}\n")
            f.write(f"number of objects not touching the border: {len(not_border_regions)}\n")
            f.write(f"mean area of objects not touching the border: {mean_area:.2f}\n")
            f.write(f"maximum diameter overall: {max_diameter_overall:.2f}\n")
            f.write(f"##########################\n\n")

            # Write headers for the regionprops table (only once for readability)
            f.write("image_file,region_label,area,perimeter,mean_intensity,circularity,eccentricity,solidity,max_diameter\n")
            for stats in region_stats:
                f.write(f"{image_file},{stats['label']},{stats['area']},{stats['perimeter']:.2f},"
                f"{stats['mean_intensity']:.2f},{stats['circularity']:.3f},"
                f"{stats['eccentricity']:.3f},{stats['solidity']:.3f},{stats['max_diameter']:.3f}\n")

            # Optional empty line between images
            f.write("\n")

        print(f"Saved into {output_csv}")

    def _is_touching_border(self, region, mask_shape, border_buffer):

        coords = region.coords  # all (row, col) of the region
        n_rows, n_cols = mask_shape

        return np.any(
            (coords[:, 0] < border_buffer) |  # Top buffer
            (coords[:, 0] >= n_rows - border_buffer) |  # Bottom buffer
            (coords[:, 1] < border_buffer) |  # Left buffer
            (coords[:, 1] >= n_cols - border_buffer)  # Right buffer
        )
    

    def _calculate_max_diameter(self, region_label, mask):

        binary_mask = (mask == region_label).astype(np.uint8)
        sitk_mask = sitk.GetImageFromArray(binary_mask)
        sitk_image = sitk.GetImageFromArray(np.zeros_like(binary_mask))
        shape_calc = RadiomicsShape2D(sitk_image, sitk_mask)
        shape_calc._calculateFeatures()

        return shape_calc.getMaximumDiameterFeatureValue()


    def _search_segs(self, img_directory, cur_selected_img):
        # TODO - has to be removed and the one from app.py should be used - only due to redundancy
        """Returns a list of full paths of segmentations for an image"""
        # Take all segmentations of the image from the current directory:
        search_string = utils.get_path_stem(cur_selected_img) + "_seg"
        seg_files = [
            file_name
            for file_name in os.listdir(img_directory)
            if (
                search_string == utils.get_path_stem(file_name)
                or str(file_name).startswith(search_string)
            )
        ]
        return seg_files



# # Main to test the functions
# from_directory = "/Users/helena.pelin/Desktop/Workmap/Projects/Active-Learning-Tool/postprocessing_featuresextraction/data/"
# current_img = "159-22_1.tif"
# postpr_class = FeatureExtraction()
# postpr_class.postprocess(from_directory, current_img, border_buffer = 5)
# print("Done")