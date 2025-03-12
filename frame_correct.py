import cv2
import trackpy as tp
import numpy as np


def Frame_correct(frames,search_params,overide=[[],[]],steps=0,
                  output_crop=None,input_crop=None,):
    molecule_size, min_mass, max_mass, separation, min_size,max_ecc, adaptive_stop, search_range=search_params
    for i in range(steps):
        f = tp.batch(frames, molecule_size, minmass=min_mass, separation=separation,engine='python')
        t = tp.link(f, search_range=search_range, adaptive_stop=adaptive_stop,memory=7)
        if i==0:
            frames=main(frames,t,molecule_size,output_crop=output_crop,input_crop=input_crop,overide=overide)
        else:
            frames=main(frames,t,molecule_size,output_crop=output_crop,input_crop=input_crop)
    return frames

def crop_center(frame, crop_width, crop_height):
    """Crop the center region of an image."""
    h, w = frame.shape[:2]
    start_x = (w - crop_width) // 2
    start_y = (h - crop_height) // 2
    return frame[start_y:start_y+crop_height, start_x:start_x+crop_width]

def mask_from_points(t,frame,diameter,size):
    """Creates a mask so that the image alignment doesnt look at the particles
    t is df from trackpy.link"""
    mask = np.ones((size, size), dtype=np.uint8) * 255  # Use entire image
    t_=t[t['frame']==frame]
    x=list(t_['x'])
    y=list(t_['y'])
    for i in range(len(x)):
        cv2.circle(mask, (int(x[i]), int(y[i])), int(diameter/2), 0, thickness=-1)
    return mask
def align_frames(template, image,drift_estimate,mask):
    """
    Compute the translation aligning 'image' to 'template'.
    First try using ECC. If ECC fails (e.g. does not converge),
    fall back to phase correlation.
    """
    warp_mode = cv2.MOTION_TRANSLATION
    # Initialize the warp matrix as identity.
    warp_matrix = warp_matrix = np.array([[1, 0, drift_estimate[1]], #drifts returns y,x
                        [0, 1, drift_estimate[0]]], dtype=np.float32)
    # Set ECC termination criteria: maximum iterations and epsilon.
    number_of_iterations = 5000
    termination_eps = 1e-6
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)
    
    try:
        # Convert images to float32 (ECC works best with float images).
        template_f = template.astype(np.float32)
        image_f = image.astype(np.float32)
        # findTransformECC returns the enhanced correlation coefficient and warp matrix.
        cc, warp_matrix = cv2.findTransformECC(template_f, image_f, warp_matrix,
                                                 warp_mode, criteria,input_mask=mask)
    except cv2.error as e:
        # Fallback: use phase correlation to estimate translation.
        template_f = np.float32(template)
        image_f = np.float32(image)
        (shift_x, shift_y), response = cv2.phaseCorrelate(template_f, image_f)
        warp_matrix = np.array([[1, 0, shift_x],
                                [0, 1, shift_y]], dtype=np.float32)
    
    return warp_matrix

def main(frames,t,radius,output_crop=None,overide=[[],[]],input_crop=None,):
    #reader = SXMReader(SXM_PATH[0], channel="Z", correct="plane")  # Adjust parameters as needed
    #frames=reader
    #frames = [np.array(frame) for frame in reader]  # Convert to NumPy arrays
    drift_estimate=tp.compute_drift(t)
    camera_drifts=[]
    # Determine if the first frame is color.
    if frames[0].ndim == 2:
        is_color = False
    else:
        # Assume color if there are 3 or 4 channels.
        is_color = True

    cropped_frames = []
    if output_crop==None:
        output_crop=np.shape(frames)[1]
    # Crop each frame to its center region and standardize channel numbers.
    for frame in frames:
        if input_crop==None:
            cropped=frame
        else:
            cropped = crop_center(frame, input_crop, input_crop)
        # If we expect color but the frame is grayscale, convert to RGB.
        if is_color and cropped.ndim == 2:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
        # If the frame has an alpha channel, convert from RGBA to RGB.
        elif is_color and cropped.ndim == 3 and cropped.shape[2] == 4:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_RGBA2RGB)

        cropped_frames.append(cropped)
    
    # Create grayscale versions for alignment computation.
    if is_color:
        cropped_gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in cropped_frames]
    else:
        cropped_gray = cropped_frames

    aligned_frames = []         # To store the drift-corrected frames.
    cumulative_translation = np.array([0, 0], dtype=np.float32)
    
    # The first frame is used as the reference.
    image_shift=(output_crop-cropped.shape[1])/2

    M = np.array([[1, 0, image_shift],
                  [0, 1,image_shift]], dtype=np.float32)
    warped = cv2.warpAffine(cropped_frames[0], M, (output_crop, output_crop),
                        flags=cv2.INTER_LINEAR)
    aligned_frames.append(warped)

    prev_gray = cropped_gray[0]

    # Process each pair of consecutive frames.
    for i in range(1, len(cropped_frames)):
        curr_gray = cropped_gray[i]
        # Compute translation from previous frame to current frame.
        mask=mask_from_points(t,i,radius,output_crop)

        warp_matrix = align_frames(prev_gray, curr_gray,drift_estimate.loc[i],mask)
        if i in overide[0]:
            dx =overide[1][overide[0].index(i)][0]
            dy=overide[1][overide[0].index(i)][1]
        else:
            dy = warp_matrix[1, 2]
            dx = warp_matrix[0, 2]
        camera_drifts.append([dx,dy])
        cumulative_translation += np.array([dx, dy], dtype=np.float32)
        
        # Create a translation matrix to warp the current frame to align with the reference.
        M = np.array([[1, 0, -cumulative_translation[0]+image_shift],
                      [0, 1, -cumulative_translation[1]+image_shift]], dtype=np.float32)

        # Apply the translation warp.
        warped = cv2.warpAffine(cropped_frames[i], M, (output_crop, output_crop),
                                flags=cv2.INTER_LINEAR)
        aligned_frames.append(warped)

        # Update the reference (using the original gray version for this frame).
        prev_gray = curr_gray

    return(aligned_frames)


