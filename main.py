import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from part_swap import load_checkpoints
import torch
import torch.nn.functional as F
import matplotlib.patches as mpatches
from skimage import img_as_ubyte
from part_swap import make_video
import warnings

warnings.filterwarnings("ignore")

torch.cuda.empty_cache()

source_image = imageio.imread('files/pawel5.png')
target_video = imageio.mimread('files/mata_face_crop.mp4')

#Resize image and video to 256x256

source_image = resize(source_image, (512, 512))[..., :3]
target_video = [resize(frame, (512, 512))[..., :3] for frame in target_video]

def display(source, target, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(target)):
        cols = [source]
        cols.append(target[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani
    

# display(source_image, target_video).to_html5_video()



reconstruction_module, segmentation_module = load_checkpoints(config='config/vox-256-sem-5segments.yaml', 
                                               checkpoint='./vox-5segments.pth.tar',
                                               blend_scale=1)


def visualize_segmentation(image, network, supervised=False, hard=True, colormap='gist_rainbow'):
    with torch.no_grad():
        inp = torch.tensor(image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
        if supervised:
            inp = F.interpolate(inp, size=(512, 512))
            inp = (inp - network.mean) / network.std
            mask = torch.softmax(network(inp)[0], dim=1)
            mask = F.interpolate(mask, size=image.shape[:2])
        else:
            mask = network(inp)['segmentation']
            mask = F.interpolate(mask, size=image.shape[:2], mode='bilinear')
    
    if hard:
        mask = (torch.max(mask, dim=1, keepdim=True)[0] == mask).float()
    
    colormap = plt.get_cmap(colormap)
    num_segments = mask.shape[1]
    mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
    color_mask = 0
    patches = []
    for i in range(num_segments):
        if i != 0:
            color = np.array(colormap((i - 1) / (num_segments - 1)))[:3]
        else:
            color = np.array((0, 0, 0))
        patches.append(mpatches.Patch(color=color, label=str(i)))
        color_mask += mask[..., i:(i+1)] * color.reshape(1, 1, 3)
    
    fig, ax = plt.subplots(1, 2, figsize=(12,6))

    ax[0].imshow(color_mask)
    ax[1].imshow(0.3 * image + 0.7 * color_mask)
    ax[1].legend(handles=patches)
    ax[0].axis('off')
    ax[1].axis('off')

visualize_segmentation(source_image, segmentation_module, hard=True)
plt.show()

predictions = make_video(swap_index=[1,2,4,5], source_image = source_image, target_video = target_video,
                             segmentation_module=segmentation_module, reconstruction_module=reconstruction_module)
# display(source_image, target_video, predictions).to_html5_video()


# Saving result video
imageio.mimsave('./result.mp4', [img_as_ubyte(frame) for frame in predictions], fps=30)