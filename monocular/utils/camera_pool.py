import random
import torch


class CameraPool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_cameras = 0
            self.cameras = []

    def query(self, cameras):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return cameras
        return_cameras = []
        for camera in cameras:
            camera = torch.unsqueeze(camera.data, 0)
            if self.num_cameras < self.pool_size:  # if the buffer is not full; keep inserting current images to the buffer
                self.num_cameras = self.num_cameras + 1
                self.cameras.append(camera)
                return_cameras.append(camera)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.cameras[random_id].clone()
                    self.cameras[random_id] = camera
                    return_cameras.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_cameras.append(camera)
        return_cameras = torch.cat(return_cameras, 0)  # collect all the images and return
        return return_cameras
