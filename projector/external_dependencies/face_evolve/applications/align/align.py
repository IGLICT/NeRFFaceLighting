from PIL import Image
from detector import detect_faces
from align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np

def align_image(img: Image.Image, crop_size: int=112):
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale
    _, landmarks = detect_faces(img)
    if len(landmarks) == 0: # If the landmarks cannot be detected, the img will be discarded
        raise Exception(f"No face is detected.")
    facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
    img_warped = Image.fromarray(warped_face)
    return img_warped