from typing import List
import numpy as np
from skimage.measure import find_contours, label
from skimage.draw import polygon_perimeter
from scipy.ndimage import label as labelnd

class Compute4Mask:
    """
    Compute4Mask provides methods for manipulating masks to make visualisation in the viewer easier.
    """

    @staticmethod
    def get_contours(
        instance_mask: np.ndarray, contours_level: float = None
    ) -> np.ndarray:
        """Find contours of objects in the instance mask. This function is used to identify the contours of the objects to prevent the problem of the merged
        objects in napari window (mask).

        :param instance_mask: The instance mask array.
        :type instance_mask: numpy.ndarray
        :param contours_level: Value along which to find contours in the array. See skimage.measure.find_contours for more.
        :type: None or float
        :return: A binary mask where the contours of all objects in the instance segmentation mask are one and the rest is background.
        :rtype: numpy.ndarray

        """
        instance_ids = Compute4Mask.get_unique_objects(
            instance_mask
        )  # get object instance labels ignoring background
        contour_mask = np.zeros_like(instance_mask)
        for instance_id in instance_ids:
            # get a binary mask only of object
            single_obj_mask = np.zeros_like(instance_mask)
            single_obj_mask[instance_mask == instance_id] = 1
            try:
                # compute contours for mask
                contours = find_contours(single_obj_mask, contours_level)
                # sometimes little dots appeas as additional contours so remove these
                if len(contours) > 1:
                    contour_sizes = [contour.shape[0] for contour in contours]
                    contour = contours[contour_sizes.index(max(contour_sizes))].astype(
                        int
                    )
                else:
                    contour = contours[0]
                # and draw onto contours mask
                rr, cc = polygon_perimeter(
                    contour[:, 0], contour[:, 1], contour_mask.shape
                )
                contour_mask[rr, cc] = instance_id
            except Exception as error:
                print("Could not create contour for instance id", instance_id, ". Error is :", error)
        return contour_mask

    @staticmethod
    def add_contour(labels_mask: np.ndarray, instance_mask: np.ndarray) -> np.ndarray:
        """Add contours of objects to the labels mask.

        :param labels_mask: The class mask array without the contour pixels annotated.
        :type labels_mask: numpy.ndarray
        :param instance_mask: The instance mask array.
        :type instance_mask: numpy.ndarray
        :return: The updated class mask including contours.
        :rtype: numpy.ndarray
        """
        labels_mask_with_contours = labels_mask.copy()
        instance_ids = Compute4Mask.get_unique_objects(instance_mask)

        for instance_id in instance_ids:
            
            # Binary mask for this instance
            instance_binary = (instance_mask == instance_id)
    
            # Label connected components within this instance
            single_instance_labeled = label(instance_binary, connectivity=1)
            num_components = single_instance_labeled.max()

            # If multiple components, there may be a case where instance was split
            for i in range(1, num_components + 1):
                component_mask = single_instance_labeled == i
                # get unique class ids where the object is present
                class_vals, counts = np.unique(labels_mask[component_mask], return_counts=True)
                # Skip if no class assigned
                if len(class_vals) == 0: continue
                # and take the class id which is most heavily represented
                class_id = class_vals[np.argmax(counts)]
                # ignore background=0 if dominant
                if class_id==0: continue
                # make sure instance mask and class mask match
                labels_mask_with_contours[component_mask] = class_id   

        return labels_mask_with_contours

    @staticmethod
    def compute_new_instance_mask(
        labels_mask: np.ndarray, instance_mask: np.ndarray
    ) -> np.ndarray:
        """Given an updated labels mask, update also the instance mask accordingly.
        So far the user can only remove an entire object in the labels mask view by
        setting the color of the object to the background.
        Therefore the instance mask can only change by entirely removing an object.

        :param labels_mask: The labels mask array, with changes made by the user.
        :type labels_mask: numpy.ndarray
        :param instance_mask: The existing instance mask, which needs to be updated.
        :type instance_mask: numpy.ndarray
        :return: The updated instance mask.
        :rtype: numpy.ndarray
        """
        '''
        instance_ids = Compute4Mask.get_unique_objects(instance_mask)
        for instance_id in instance_ids:
            unique_items_in_class_mask = list(
                np.unique(labels_mask[instance_mask == instance_id])
            )
            if (
                len(unique_items_in_class_mask) == 1
                and unique_items_in_class_mask[0] == 0
            ):
                instance_mask[instance_mask == instance_id] = 0
        return instance_mask
        '''
        # Create a copy to avoid modifying the original in place
        updated_instance_mask = instance_mask.copy()

        # Wherever the labels mask is set to background (0), clear the instance mask
        updated_instance_mask[labels_mask == 0] = 0

        return updated_instance_mask

    @staticmethod
    def compute_new_labels_mask(
        labels_mask: np.ndarray,
        instance_mask: np.ndarray,
        original_instance_mask: np.ndarray,
        old_instances: np.ndarray,
    ) -> np.ndarray:
        """Given the existing labels mask, the updated instance mask is used to update the labels mask.

        :param labels_mask: The existing labels mask, which needs to be updated.
        :type labels_mask: numpy.ndarray
        :param instance_mask: The instance mask array, with changes made by the user.
        :type instance_mask: numpy.ndarray
        :param original_instance_mask: The instance mask array, before the changes made by the user.
        :type original_instance_mask: numpy.ndarray
        :param old_instances: A list of the instance label ids in original_instance_mask.
        :type old_instances: list
        :return: The new labels mask, with updated changes according to those the user has made in the instance mask.
        :rtype: numpy.ndarray
        """
        new_labels_mask = np.zeros_like(labels_mask, dtype=np.int16)
        for instance_id in np.unique(instance_mask):
            where_instance = np.where(instance_mask == instance_id)
            # if the label is background skip
            if instance_id == 0:
                continue
            # if the label is a newly added object, add with the same id to the labels mask
            # this is an indication to the user that this object needs to be assigned a class
            elif instance_id not in old_instances:
                new_labels_mask[where_instance] = -1 #instance_id
            else:
                where_instance_orig = np.where(original_instance_mask == instance_id)
                # if the locations of the instance haven't changed, means object wasn't changed, do nothing
                num_classes = np.unique(labels_mask[where_instance])
                # if area was erased and object retains same class
                if len(num_classes) == 1:
                    new_labels_mask[where_instance] = num_classes[0]
                # area was added where there is background or other class
                else:
                    old_class_id, counts = np.unique(
                        labels_mask[where_instance_orig], return_counts=True
                    )
                    # assert len(old_class_id)==1
                    # old_class_id = old_class_id[0]
                    # and take the class id which is most heavily represented
                    old_class_id = old_class_id[np.argmax(counts)]
                    new_labels_mask[where_instance] = old_class_id

        return new_labels_mask

    @staticmethod
    def get_unique_objects(active_mask: np.ndarray) -> List:
        """Gets unique objects from the active mask.

        :param active_mask: The mask array.
        :type active_mask: numpy.ndarray
        :return: A list of unique object labels.
        :rtype: list
        """
        return list(np.unique(active_mask)[1:])

    @staticmethod
    def assert_connected_objects(mask: np.ndarray) -> tuple:
        """Before saving the final mask make sure the user has not mistakenly made an error during annotation,
        such that one instance id does not correspond to exactly one class id. Also checks whether for one instance id
        multiple classes exist.
        :param mask: The mask which we want to test.
        :type mask: numpy.ndarray
        :return:
            - A boolean which is True if there is more than one connected components corresponding to an instance id and Fale otherwise.
            - A list with all the instance ids for which more than one connected component was found.
        :rtype :
            - bool
            - list[int]
        """
        user_annot_error = False
        faulty_ids_annot = []
        if mask.ndim>2: instance_mask = mask[0]
        else: instance_mask = mask
        instance_ids = Compute4Mask.get_unique_objects(instance_mask)
        for instance_id in instance_ids:
            # check if there are more than one objects (connected components) with same instance_id
            if np.unique(label(instance_mask == instance_id)).shape[0] > 2:
                user_annot_error = True
                faulty_ids_annot.append(instance_id)
        return (
            user_annot_error,
            faulty_ids_annot,
        )
    
    @staticmethod
    def add_seg_layer(mask):
        """
        The input mask has only one channel representing the instance ids - add an additional channel which represents 
        the classes of the instances. These are originally all set to -1, meaning that the user should annotate them.
        :param mask: The mask with the instance ids
        :type mask: numpy.ndarray
        :return: The mask with an additional channel added representing the class mask
        """
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

        # Create class mask filled with -1
        class_mask = np.where(mask == 0, 0, -1)#.astype(np.int32)

        # Stack into shape (2, H, W)
        mask_with_classes = np.stack([mask, class_mask], axis=0)

        return mask_with_classes
    
    @staticmethod
    def assert_missing_classes(mask) -> bool:
        """
        Check if the class mask (second channel) contains any -1 values,
        which indicate unannotated regions.

        :param mask: The 2xHxW mask (first channel = instance ids, 
                                  second channel = class ids)
        :type mask: numpy.ndarray
        :return: True if any -1 values are found in the class mask, otherwise False
        """
        if mask.ndim != 3 or mask.shape[0] != 2:
            raise ValueError(f"Expected shape (2, H, W), got {mask.shape}")

        class_mask = mask[1]
        return np.any(class_mask == -1)
    
    @staticmethod
    def assert_num_classes(mask, num_classes) -> bool:
        """
        Check if the class mask (second channel) contains a number of classes larger than that specified by the user at rutime.

        :param mask: The 2xHxW mask (first channel = instance ids, 
                                  second channel = class ids)
        :type mask: numpy.ndarray
        :return: True if more classes found, otherwise False
        """
        class_ids = Compute4Mask.get_unique_objects(mask[1])
        if len(class_ids)>num_classes: return True
        else: return False

    
    @staticmethod
    def keep_largest_components_pair(mask, faulty_ids: list):
        """
        Keeps only the largest connected component for each label in faulty_ids
        in mask.

        :param mask: The mask which we want to test.
        :type mask: numpy.ndarray
        :param faulty_ids: List of label IDs for which to keep only the largest component.
        :type faulty_ids: List
        :return: Tuple of cleaned masks: (cleaned_mask, cleaned_class_mask)
        """
        if mask.ndim>2:
            cleaned_mask = mask[0].copy()
            cleaned_class_mask = mask[1].copy()
        else:
            cleaned_mask = mask.copy()
            cleaned_class_mask = np.zeros_like(mask)

        for label_id in faulty_ids:
            # binary mask for current label
            binary_mask = (cleaned_mask == label_id)

            if np.any(binary_mask):
                # label connected components
                labeled_array, num_features = labelnd(binary_mask)

                if num_features == 0:
                    continue

                # count pixels in each component
                component_sizes = np.bincount(labeled_array.ravel())
                component_sizes[0] = 0  # ignore background

                # largest component
                largest_component_label = component_sizes.argmax()

                # mask for pixels to remove
                remove_mask = (labeled_array != largest_component_label) & (labeled_array != 0)

                # set these pixels to 0 in both masks
                cleaned_mask[remove_mask] = 0
                cleaned_class_mask[remove_mask] = 0

                # Stack along new first axis
                stacked = np.stack([cleaned_mask, cleaned_class_mask], axis=0)

        if mask.ndim>2: return stacked
        else: return cleaned_mask