from wave_sim2d.wave_simulation import SceneObject
import cupy as cp
import numpy as np
import cv2


class StaticRefractiveIndex(SceneObject):
    """
    Implements a static refractive index field that overwrites the entire domain with a constant IOR value.
    Use this as base layer in your scene.
    """

    def __init__(self, refractive_index_field):
        """
        Creates a static refractive index field object
        :param refractive_index_field: The refractive index field, same size as the source.
                                       Note that values below 0.9 are clipped to prevent the simulation
                                       from becoming instable
        """
        shape = refractive_index_field.shape
        self.c = cp.ones((shape[0], shape[1]), dtype=cp.float32)
        self.c = 1.0/cp.clip(cp.array(refractive_index_field), 0.9, 10.0)

    def render(self, field: cp.ndarray, wave_speed_field: cp.ndarray, dampening_field: cp.ndarray):
        assert (wave_speed_field.shape == self.c.shape)
        wave_speed_field[:] = self.c

    def update_field(self, field: cp.ndarray, t):
        pass


class StaticRefractiveIndexPolygon(SceneObject):
    """
    Draws a static polygon with a given refractive index into the wave_speed_field using an
    anti-aliased mask and indexing. Caches the pixel coordinates and mask values.
    """

    def __init__(self, vertices, refractive_index):
        """
        Initializes the StaticRefractiveIndexPolygon.

        Args:
            vertices (list or np.ndarray): A list or array of (x, y) coordinates defining the polygon.
            refractive_index (float): The refractive index of the polygon. Values are clamped to [0.9, 10.0].
        """
        self.vertices = np.array(vertices, dtype=np.float32)
        self.refractive_index = min(max(refractive_index, 0.9), 10.0)
        self._cached_coords = None
        self._cached_mask_values = None
        self._cached_field_shape = (0, 0)

    def _create_polygon_data(self, field_shape):
        """
        Creates and caches the pixel coordinates and anti-aliased mask values for the polygon.

        Args:
            field_shape (tuple): The shape (rows, cols) of the simulation field.

        Returns:
            tuple: A tuple containing:
                - coords (tuple of cp.ndarray): (y_coordinates, x_coordinates) of the polygon pixels within the field.
                - mask_values (cp.ndarray): Corresponding anti-aliased mask values (0.0 to 1.0).
        """
        if self._cached_coords is not None and self._cached_field_shape == field_shape:
            return self._cached_coords, self._cached_mask_values

        rows, cols = field_shape

        # Find the bounding box of the polygon
        min_x = np.min(self.vertices[:, 0])
        max_x = np.max(self.vertices[:, 0])
        min_y = np.min(self.vertices[:, 1])
        max_y = np.max(self.vertices[:, 1])

        mask_width = int(np.ceil(max_x - min_x)) + 1
        mask_height = int(np.ceil(max_y - min_y)) + 1
        offset_x = int(np.floor(min_x))
        offset_y = int(np.floor(min_y))

        # Create the mask
        mask = np.zeros((mask_height, mask_width), dtype=np.float32)
        translated_vertices = self.vertices - [offset_x, offset_y]
        translated_vertices_cv = np.round(translated_vertices).astype(np.int32)
        cv2.fillPoly(mask, [translated_vertices_cv], 1.0, lineType=cv2.LINE_AA)

        # Get coordinates and mask values of non-black pixels
        coords_y, coords_x = np.where(mask > 0)
        mask_values = mask[coords_y, coords_x]

        # Adjust coordinates to the position in the main field
        global_coords_y = coords_y + offset_y
        global_coords_x = coords_x + offset_x

        # Perform out-of-bounds check here
        in_bounds = (global_coords_y >= 0) & (global_coords_y < rows) & \
                    (global_coords_x >= 0) & (global_coords_x < cols)

        valid_global_y = global_coords_y[in_bounds]
        valid_global_x = global_coords_x[in_bounds]
        valid_mask_values = mask_values[in_bounds]

        self._cached_coords = (cp.array(valid_global_y), cp.array(valid_global_x))
        self._cached_mask_values = cp.array(valid_mask_values, dtype=cp.float32)
        self._cached_field_shape = field_shape
        return self._cached_coords, self._cached_mask_values

    def render(self, field: cp.ndarray, wave_speed_field: cp.ndarray, dampening_field: cp.ndarray):
        coords, mask_values = self._create_polygon_data(wave_speed_field.shape)

        # Use advanced indexing to update the field and perform alpha blending
        bg_wave_speed = wave_speed_field[coords[0], coords[1]]
        wave_speed_field[coords[0], coords[1]] = (bg_wave_speed * (1.0 - mask_values) +
                                                  mask_values / self.refractive_index)

    def update_field(self, field: cp.ndarray, t):
        pass