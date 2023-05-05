device = "cuda"

from modules.crop import *
cropper = Cropper(device)

from modules.pose import *
pose_estimator = Poser(device)

def estimate_coeffs(image: Image.Image):
    try:
        keypoints = eg3d_detect_keypoints(image)
        pred_coeffs = cropper.get_deep3d_coeffs(image, keypoints)
        pose_data = pose_estimator.get_pose(pred_coeffs)
        image_cropped = cropper.final_crop(image, keypoints)
        return image_cropped, pose_data
    except Exception as e:
        # Cannot detect at least one valid face.
        return None, None

def convert_to_extrinsics(angle, trans):
    return torch.from_numpy(np.array(pose_estimator.convert(angle, trans), dtype=np.float32)).to(device).reshape(-1, 16)

if __name__ == "__main__":
    img = Image.open("/path/to/image") # Some Image
    img_cropped, pose_data = estimate_coeffs(img)
    assert img_cropped != None and pose_data != None
    # "angle" contains (pitch, yaw, roll), which are the pose of the image initially and can be freely modified.
    # "trans" contains the transition information, which in most cases does not need to be changed.
    extrinsics = convert_to_extrinsics(pose_data["angle"], pose_data["trans"])