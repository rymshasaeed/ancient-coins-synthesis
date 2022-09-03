import cv2
import numpy as np
import scipy.ndimage as ndi

# Define a custom class to preprocess and segment the coin-images
class preprocess(object):
    def __init__(self):
        self.input_image = None
        self.local_range = None

        # Calculate the local image range footprint
        radius = 3
        self._footprint = np.zeros((2*radius+1, 2*radius+1), dtype=np.bool)
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                d_sq = dx*dx + dy*dy
                if d_sq > radius * radius:
                    continue
                self._footprint[dx + radius, dy + radius] = True

    def segment(self, image):
        self.input_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # compute the local range image
        self.local_range = ndi.maximum_filter(self.input_image, footprint=self._footprint)
        self.local_range -= ndi.minimum_filter(self.input_image, footprint=self._footprint)

        # normalize it
        self.local_range = self.local_range / float(np.amax(self.local_range))

        # find a threshold which gives the coin border
        best_threshold = 0
        best_contour = None
        best_form_factor = 0.0
        best_bin_im = None

        for threshold in np.arange(0.05, 0.65, 0.05):
            # Find contours in thresholded image
            contour_im = self.local_range >= threshold
            contours, _ = cv2.findContours(np.array(contour_im, dtype=np.uint8),
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_NONE)

            # Find maximum area contour
            areas = list(cv2.contourArea(c) for c in contours)
            max_index = np.argmax(areas)

            # Calculate the form factor
            contour = contours[max_index]
            area = areas[max_index]
            perim = cv2.arcLength(contour, closed=True)
            form_factor = 4.0 * np.pi * area / (perim * perim)

            # Reject contours with an area > 90% of the image to reject
            # contour covering entire image
            if area > 0.9 * np.product(self.local_range.shape):
                continue

            # Update best form factor
            if form_factor >= best_form_factor:
                best_threshold = threshold
                best_contour = contour
                best_form_factor = form_factor
                best_bin_im = contour_im

        # Store the extracted edge
        self.edge = np.reshape(best_contour, (len(best_contour), 2))
        self.edge_mask = best_bin_im.astype('float64')
        self.edge_threshold = best_threshold
        self.edge_form_factor = best_form_factor
