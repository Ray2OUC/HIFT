import warnings
warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import argparse
import numpy as np
import torch
import torch.nn.functional as func
import sys
import cv2
from tqdm import tqdm
from HIFT_core import DeSTR

def SIFT_detect(img, nfeatures=1500, contrastThreshold=0.04):
    """ Compute SIFT feature points. """
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures,
                                       contrastThreshold=contrastThreshold)
    keypoints = sift.detect(img, None)
    keypoints = [[k.pt[1], k.pt[0], k.response] for k in keypoints]
    keypoints = np.array(keypoints)
    return keypoints

def keypoints_to_grid(keypoints, img_size):
    """
    Convert a tensor [N, 2] or batched tensor [B, N, 2] of N keypoints into
    a grid in [-1, 1]² that can be used in torch.nn.functional.interpolate.
    """
    n_points = keypoints.size()[-2]
    device = keypoints.device
    grid_points = keypoints.float() * 2. / torch.tensor(
        img_size, dtype=torch.float, device=device) - 1.
    grid_points = grid_points[..., [1, 0]].view(-1, n_points, 1, 2) #B*np*1*2
    return grid_points

def _adapt_weight_names(state_dict):
    """ Adapt the weight names when the training and testing are done
    with a different GPU configuration (with/without DataParallel). """
    train_parallel = list(state_dict.keys())[0][:7] == 'module.'
    test_parallel = torch.cuda.device_count() > 1
    new_state_dict = {}
    if train_parallel and (not test_parallel):
        # Need to remove 'module.' from all the variable names
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
    elif test_parallel and (not train_parallel):
        # Need to add 'module.' to all the variable names
        for k, v in state_dict.items():
            new_k = 'module.' + k
            new_state_dict[new_k] = v
    else:  # Nothing to do
        new_state_dict = state_dict
    return new_state_dict

def _match_state_dict(old_state_dict, new_state_dict):
    """ Return a new state dict that has exactly the same entries
            as old_state_dict and that is updated with the values of
            new_state_dict whose entries are shared with old_state_dict.
            This allows loading a pre-trained network. """
    return ({k: new_state_dict[k] if k in new_state_dict else v
             for (k, v) in old_state_dict.items()},
            old_state_dict.keys() == new_state_dict.keys())

def resize_and_crop(image, img_size):
    """ Resize an image to the given img_size by first rescaling it
        and then applying a central crop to fit the given dimension. """
    source_size = np.array(image.shape[:2], dtype=float)
    target_size = np.array(img_size, dtype=float)

    # Scale
    scale = np.amax(target_size / source_size)
    inter_size = np.round(source_size * scale).astype(int)
    image = cv2.resize(image, (inter_size[1], inter_size[0]))

    # Central crop
    pad = np.round((source_size * scale - target_size) / 2.).astype(int)
    image = image[pad[0]:(pad[0] + int(target_size[0])),
            pad[1]:(pad[1] + int(target_size[1])), :]

    return image

def export(images_list, num_keypoints, detection_thresh, extension, resize=False, h=480, w=640):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load my descriptor model
    checkpoint_path = '/home/ouc/hift.pth'
    descriptor = DeSTR()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    adapt_dict = _adapt_weight_names(checkpoint['model_state_dict'])
    net_dict = descriptor.state_dict()
    updated_state_dict, same_net = _match_state_dict(net_dict, adapt_dict)
    descriptor.load_state_dict(updated_state_dict)
    descriptor = descriptor.to(device)
    if same_net:
        print("Success in loading model!")
    descriptor.eval()

    # Parse the data, predict the features, and export them in an npz file
    with open(images_list, 'r') as f:
        image_files = f.readlines()
    image_files = [path.strip('\n') for path in image_files]

    for img_path in tqdm(image_files):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize:
            img = resize_and_crop(img, (h, w))
        img_size = img.shape
        if img_size[2] != 3:
            sys.exit('Export only available for RGB images.')
        cpu_gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        keypoints = SIFT_detect(cpu_gray_img, nfeatures=num_keypoints,
                                contrastThreshold=detection_thresh)
        scores = keypoints[:, 2]
        img_tensor = torch.tensor(cpu_gray_img, dtype=torch.float, device=device)
        img_tensor = img_tensor[None][None] / 255.
        grid_points = keypoints_to_grid(
            torch.tensor(keypoints[:, :2], dtype=torch.float, device=device),
            img_size[:2])
        keypoints = keypoints[:, [1, 0]]

        # Predict the corresponding descriptors
        #inputs = img
        with torch.no_grad():
            outputs = descriptor.forward(img_tensor)
            descs = {}
            descs['descriptors'] = outputs
            descriptors = []
            for k in descs.keys():
                desc = func.grid_sample(descs[k], grid_points).squeeze().cpu().numpy().transpose(1, 0)
                descriptors.append(desc)
            descriptors = np.stack(descriptors, axis=1) #n*1*128

            # Keep the best scores
            idxs = scores.argsort()[-num_keypoints:]
            with open(img_path + extension, 'wb') as output_file:
                np.savez(output_file, keypoints=keypoints[idxs],
                         descriptors=descriptors[idxs], scores=scores[idxs])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('images_list', type=str,
                         help='Path to a txt file containing the image paths.')
    parser.add_argument('--num_kp', type=int, default=1500,
                         help="Number of keypoints to use.")
    parser.add_argument('--detection_thresh', type=float, default=0.04,
                         help="Detection threshold for SuperPoint.")
    parser.add_argument('--resize', action='store_true', default=False,
                        help='Resize the images to a given dimension.')
    parser.add_argument('--h', type=int, default='480',
                        help='Image height.')
    parser.add_argument('--w', type=int, default='640',
                        help='Image width.')
    parser.add_argument('--extension', type=str, default=None,
                         help="Extension to add to each exported npz.")
    args = parser.parse_args()

    num_keypoints = args.num_kp

    keypoints_type = args.keypoints
    extension = args.extension if args.extension else 'dna_sift'
    extension = '.' + extension

    export(args.images_list, num_keypoints, args.detection_thresh, extension, args.resize, args.h, args.w)


