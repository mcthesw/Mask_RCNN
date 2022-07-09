"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib

    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/nucleus/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
# CHANGED
'''VAL_IMAGE_IDS = [
    "0c2550a23b8a0f29a7575de8c61690d3c31bc897dd5ba66caec201d201a278c2",
    "92f31f591929a30e4309ab75185c96ff4314ce0a7ead2ed2c2171897ad1da0c7",
    "1e488c42eb1a54a3e8412b1f12cde530f950f238d71078f2ede6a85a02168e1f",
    "c901794d1a421d52e5734500c0a2a8ca84651fb93b19cec2f411855e70cae339",
    "8e507d58f4c27cd2a82bee79fe27b069befd62a46fdaed20970a95a2ba819c7b",
    "60cb718759bff13f81c4055a7679e81326f78b6a193a2d856546097c949b20ff",
    "da5f98f2b8a64eee735a398de48ed42cd31bf17a6063db46a9e0783ac13cd844",
    "9ebcfaf2322932d464f15b5662cae4d669b2d785b8299556d73fffcae8365d32",
    "1b44d22643830cd4f23c9deadb0bd499fb392fb2cd9526d81547d93077d983df",
    "97126a9791f0c1176e4563ad679a301dac27c59011f579e808bbd6e9f4cd1034",
    "e81c758e1ca177b0942ecad62cf8d321ffc315376135bcbed3df932a6e5b40c0",
    "f29fd9c52e04403cd2c7d43b6fe2479292e53b2f61969d25256d2d2aca7c6a81",
    "0ea221716cf13710214dcd331a61cea48308c3940df1d28cfc7fd817c83714e1",
    "3ab9cab6212fabd723a2c5a1949c2ded19980398b56e6080978e796f45cbbc90",
    "ebc18868864ad075548cc1784f4f9a237bb98335f9645ee727dac8332a3e3716",
    "bb61fc17daf8bdd4e16fdcf50137a8d7762bec486ede9249d92e511fcb693676",
    "e1bcb583985325d0ef5f3ef52957d0371c96d4af767b13e48102bca9d5351a9b",
    "947c0d94c8213ac7aaa41c4efc95d854246550298259cf1bb489654d0e969050",
    "cbca32daaae36a872a11da4eaff65d1068ff3f154eedc9d3fc0c214a4e5d32bd",
    "f4c4db3df4ff0de90f44b027fc2e28c16bf7e5c75ea75b0a9762bbb7ac86e7a3",
    "4193474b2f1c72f735b13633b219d9cabdd43c21d9c2bb4dfc4809f104ba4c06",
    "f73e37957c74f554be132986f38b6f1d75339f636dfe2b681a0cf3f88d2733af",
    "a4c44fc5f5bf213e2be6091ccaed49d8bf039d78f6fbd9c4d7b7428cfcb2eda4",
    "cab4875269f44a701c5e58190a1d2f6fcb577ea79d842522dcab20ccb39b7ad2",
    "8ecdb93582b2d5270457b36651b62776256ade3aaa2d7432ae65c14f07432d49",
]'''
VAL_IMAGE_IDS = ['00113211223105304_aug[1]_split[2]_patch[1507]_patch[1508]', '00113211223105242_aug[1]_split[1]_patch[1441]_patch[1442]', '00112211223103702_aug[4]_split[4]_patch[959]_patch[960]', '00115220318124445_aug[3]_split[2]_patch[1851]_patch[1852]', '00111211223102012_aug[2]_split[1]_patch[393]_patch[394]', '00115220318124642_aug[4]_split[6]_patch[3311]_patch[3312]', '00112211223103735_aug[3]_split[4]_patch[983]_patch[984]', '00115220318124556_aug[2]_split[5]_patch[2661]_patch[2662]', '00115220318124625_aug[1]_split[2]_patch[3027]_patch[3028]', '00113211223105406_aug[1]_split[3]_patch[1733]_patch[1734]', '00111211223101720_aug[4]_split[1]_patch[121]_patch[122]', '00115220318124458_aug[1]_split[5]_patch[1977]_patch[1978]', '00115220318124650_aug[1]_split[5]_patch[3369]_patch[3370]', '00111211223101720_aug[3]_split[2]_patch[115]_patch[116]', '00115220318124540_aug[2]_split[6]_patch[2519]_patch[2520]', '00113211223105304_aug[4]_split[4]_patch[1535]_patch[1536]', '00111211223101733_aug[2]_split[2]_patch[139]_patch[140]', '00111211223101947_aug[1]_split[1]_patch[321]_patch[322]', '00115220318124551_aug[3]_split[2]_patch[2619]_patch[2620]', '00111211223101709_aug[4]_split[4]_patch[95]_patch[96]', '00115220318124609_aug[4]_split[5]_patch[2829]_patch[2830]', '00115220318124654_aug[1]_split[2]_patch[3411]_patch[3412]', '00115220318124524_aug[4]_split[2]_patch[2343]_patch[2344]', '00115220318124612_aug[1]_split[6]_patch[2843]_patch[2844]', '00115220318124632_aug[1]_split[6]_patch[3131]_patch[3132]', '00115220318124535_aug[2]_split[3]_patch[2465]_patch[2466]', '00112211223104723_aug[3]_split[2]_patch[1139]_patch[1140]', '00112211223102936_aug[4]_split[3]_patch[701]_patch[702]', '00111211223101709_aug[3]_split[2]_patch[83]_patch[84]', '00112211223104804_aug[1]_split[4]_patch[1191]_patch[1192]', '00111211223103356_aug[1]_split[1]_patch[609]_patch[610]', '00112211223104530_aug[3]_split[4]_patch[1047]_patch[1048]', '00112211223104708_aug[2]_split[2]_patch[1099]_patch[1100]', '00112211223103735_aug[1]_split[1]_patch[961]_patch[962]', '00111211223101609_aug[3]_split[4]_patch[23]_patch[24]', '00115220318124615_aug[4]_split[2]_patch[2919]_patch[2920]', '00115220318124445_aug[4]_split[2]_patch[1863]_patch[1864]', '00113211223105317_aug[1]_split[1]_patch[1537]_patch[1538]', '00115220318124654_aug[3]_split[2]_patch[3435]_patch[3436]', '00115220318124628_aug[4]_split[3]_patch[3113]_patch[3114]', '00113211223105235_aug[2]_split[2]_patch[1419]_patch[1420]', '00113211223105235_aug[1]_split[1]_patch[1409]_patch[1410]', '00113211223105357_aug[3]_split[4]_patch[1687]_patch[1688]', '00113211223105228_aug[3]_split[4]_patch[1367]_patch[1368]', '00115220318124642_aug[2]_split[2]_patch[3279]_patch[3280]', '00113211223105248_aug[4]_split[4]_patch[1503]_patch[1504]', '00113211223105317_aug[3]_split[3]_patch[1557]_patch[1558]', '00115220318124625_aug[2]_split[1]_patch[3037]_patch[3038]', '00115220318124605_aug[2]_split[1]_patch[2749]_patch[2750]', '00112211223104804_aug[1]_split[2]_patch[1187]_patch[1188]', '00115220318124609_aug[4]_split[6]_patch[2831]_patch[2832]', '00115220318124645_aug[1]_split[1]_patch[3313]_patch[3314]', '00112211223103049_aug[4]_split[2]_patch[731]_patch[732]', '00115220318124605_aug[1]_split[3]_patch[2741]_patch[2742]', '00115220318124551_aug[1]_split[1]_patch[2593]_patch[2594]', '00115220318124527_aug[1]_split[3]_patch[2357]_patch[2358]', '00115220318124556_aug[4]_split[1]_patch[2677]_patch[2678]', '00111211223102032_aug[4]_split[3]_patch[477]_patch[478]', '00115220318124527_aug[4]_split[2]_patch[2391]_patch[2392]', '00112211223104842_aug[1]_split[3]_patch[1253]_patch[1254]', '00115220318124632_aug[2]_split[2]_patch[3135]_patch[3136]', '00115220318124527_aug[1]_split[2]_patch[2355]_patch[2356]', '00115220318124524_aug[4]_split[5]_patch[2349]_patch[2350]', '00111211223101733_aug[3]_split[2]_patch[147]_patch[148]', '00112211223103620_aug[4]_split[2]_patch[923]_patch[924]', '00111211223101748_aug[4]_split[4]_patch[191]_patch[192]', '00115220318124551_aug[2]_split[4]_patch[2611]_patch[2612]', '00113211223105419_aug[2]_split[3]_patch[1805]_patch[1806]', '00111211223102032_aug[1]_split[2]_patch[451]_patch[452]', '00112211223104723_aug[3]_split[1]_patch[1137]_patch[1138]', '00115220318124635_aug[2]_split[2]_patch[3183]_patch[3184]', '00112211223103049_aug[2]_split[4]_patch[719]_patch[720]', '00113211223105349_aug[3]_split[3]_patch[1653]_patch[1654]', '00115220318124551_aug[3]_split[5]_patch[2625]_patch[2626]', '00115220318124615_aug[1]_split[1]_patch[2881]_patch[2882]', '00111211223101733_aug[4]_split[4]_patch[159]_patch[160]', '00115220318124609_aug[2]_split[5]_patch[2805]_patch[2806]', '00115220318124625_aug[4]_split[4]_patch[3067]_patch[3068]', '00111211223101947_aug[1]_split[2]_patch[323]_patch[324]', '00112211223103152_aug[1]_split[1]_patch[801]_patch[802]', '00112211223103152_aug[1]_split[4]_patch[807]_patch[808]', '00115220318124551_aug[4]_split[4]_patch[2635]_patch[2636]', '00112211223104749_aug[2]_split[3]_patch[1165]_patch[1166]', '00111211223101609_aug[2]_split[3]_patch[13]_patch[14]', '00113211223105304_aug[3]_split[3]_patch[1525]_patch[1526]', '00112211223102936_aug[2]_split[2]_patch[683]_patch[684]', '00111211223101947_aug[1]_split[3]_patch[325]_patch[326]', '00115220318124657_aug[3]_split[1]_patch[3481]_patch[3482]', '00111211223101957_aug[2]_split[4]_patch[367]_patch[368]', '00115220318124514_aug[4]_split[1]_patch[2149]_patch[2150]', '00115220318124650_aug[4]_split[1]_patch[3397]_patch[3398]', '00112211223103114_aug[3]_split[3]_patch[757]_patch[758]', '00112211223102936_aug[2]_split[1]_patch[681]_patch[682]', '00115220318124531_aug[3]_split[6]_patch[2435]_patch[2436]', '00111211223101720_aug[1]_split[1]_patch[97]_patch[98]', '00115220318124535_aug[1]_split[4]_patch[2455]_patch[2456]', '00111211223101748_aug[4]_split[3]_patch[189]_patch[190]', '00115220318124551_aug[3]_split[3]_patch[2621]_patch[2622]', '00112211223103416_aug[1]_split[4]_patch[839]_patch[840]', '00113211223105406_aug[2]_split[1]_patch[1737]_patch[1738]', '00115220318124645_aug[4]_split[5]_patch[3357]_patch[3358]', '00111211223103356_aug[4]_split[2]_patch[635]_patch[636]', '00115220318124501_aug[2]_split[2]_patch[2031]_patch[2032]', '00115220318124518_aug[4]_split[1]_patch[2245]_patch[2246]', '00115220318124458_aug[2]_split[5]_patch[1989]_patch[1990]', '00115220318124535_aug[3]_split[3]_patch[2477]_patch[2478]', '00112211223104530_aug[1]_split[3]_patch[1029]_patch[1030]', '00111211223103254_aug[3]_split[1]_patch[529]_patch[530]', '00113211223105228_aug[4]_split[2]_patch[1371]_patch[1372]', '00113211223105228_aug[3]_split[3]_patch[1365]_patch[1366]', '00113211223105317_aug[4]_split[2]_patch[1563]_patch[1564]', '00111211223102032_aug[3]_split[2]_patch[467]_patch[468]', '00111211223103356_aug[2]_split[2]_patch[619]_patch[620]', '00113211223105327_aug[2]_split[1]_patch[1577]_patch[1578]', '00113211223105248_aug[3]_split[1]_patch[1489]_patch[1490]', '00115220318124621_aug[1]_split[3]_patch[2981]_patch[2982]', '00111211223102038_aug[1]_split[3]_patch[485]_patch[486]', '00115220318124520_aug[1]_split[1]_patch[2257]_patch[2258]', '00115220318124602_aug[2]_split[2]_patch[2703]_patch[2704]', '00115220318124514_aug[3]_split[4]_patch[2143]_patch[2144]', '00115220318124654_aug[3]_split[6]_patch[3443]_patch[3444]', '00115220318124657_aug[4]_split[3]_patch[3497]_patch[3498]', '00111211223101709_aug[4]_split[1]_patch[89]_patch[90]', '00115220318124458_aug[4]_split[1]_patch[2005]_patch[2006]', '00112211223103620_aug[2]_split[4]_patch[911]_patch[912]', '00111211223101817_aug[4]_split[1]_patch[217]_patch[218]', '00115220318124445_aug[3]_split[4]_patch[1855]_patch[1856]', '00112211223103620_aug[4]_split[1]_patch[921]_patch[922]', '00115220318124452_aug[3]_split[4]_patch[1903]_patch[1904]', '00115220318124540_aug[2]_split[1]_patch[2509]_patch[2510]', '00113211223105419_aug[4]_split[1]_patch[1817]_patch[1818]', '00111211223101720_aug[3]_split[3]_patch[117]_patch[118]', '00115220318124524_aug[3]_split[3]_patch[2333]_patch[2334]', '00113211223105349_aug[2]_split[1]_patch[1641]_patch[1642]', '00115220318124612_aug[1]_split[1]_patch[2833]_patch[2834]', '00111211223101748_aug[1]_split[2]_patch[163]_patch[164]', '00115220318124615_aug[4]_split[5]_patch[2925]_patch[2926]', '00115220318124642_aug[2]_split[6]_patch[3287]_patch[3288]', '00115220318124612_aug[2]_split[2]_patch[2847]_patch[2848]', '00112211223104530_aug[4]_split[4]_patch[1055]_patch[1056]', '00113211223105317_aug[4]_split[3]_patch[1565]_patch[1566]', '00115220318124635_aug[1]_split[4]_patch[3175]_patch[3176]', '00115220318124452_aug[1]_split[4]_patch[1879]_patch[1880]', '00111211223101720_aug[2]_split[3]_patch[109]_patch[110]', '00111211223101709_aug[1]_split[3]_patch[69]_patch[70]', '00115220318124628_aug[1]_split[6]_patch[3083]_patch[3084]', '00113211223105406_aug[2]_split[2]_patch[1739]_patch[1740]', '00111211223103254_aug[3]_split[3]_patch[533]_patch[534]', '00112211223103114_aug[1]_split[4]_patch[743]_patch[744]', '00115220318124650_aug[4]_split[6]_patch[3407]_patch[3408]', '00115220318124621_aug[4]_split[3]_patch[3017]_patch[3018]', '00113211223105401_aug[3]_split[3]_patch[1717]_patch[1718]', '00111211223102032_aug[2]_split[3]_patch[461]_patch[462]', '00112211223103434_aug[3]_split[1]_patch[881]_patch[882]', '00115220318124618_aug[1]_split[4]_patch[2935]_patch[2936]', '00115220318124511_aug[3]_split[4]_patch[2095]_patch[2096]', '00113211223105225_aug[1]_split[3]_patch[1317]_patch[1318]', '00112211223103434_aug[1]_split[2]_patch[867]_patch[868]', '00115220318124445_aug[1]_split[6]_patch[1835]_patch[1836]', '00112211223103434_aug[2]_split[3]_patch[877]_patch[878]', '00112211223104827_aug[4]_split[2]_patch[1243]_patch[1244]', '00115220318124609_aug[1]_split[6]_patch[2795]_patch[2796]', '00115220318124642_aug[2]_split[3]_patch[3281]_patch[3282]', '00115220318124645_aug[2]_split[1]_patch[3325]_patch[3326]', '00112211223103434_aug[4]_split[1]_patch[889]_patch[890]', '00112211223104723_aug[2]_split[2]_patch[1131]_patch[1132]', '00111211223102032_aug[4]_split[2]_patch[475]_patch[476]', '00115220318124455_aug[2]_split[1]_patch[1933]_patch[1934]', '00112211223104708_aug[1]_split[4]_patch[1095]_patch[1096]', '00111211223103318_aug[2]_split[3]_patch[589]_patch[590]', '00115220318124527_aug[1]_split[1]_patch[2353]_patch[2354]', '00111211223103254_aug[1]_split[1]_patch[513]_patch[514]', '00115220318124514_aug[1]_split[4]_patch[2119]_patch[2120]', '00115220318124535_aug[4]_split[2]_patch[2487]_patch[2488]', '00113211223105341_aug[3]_split[1]_patch[1617]_patch[1618]']


############################################################
#  Configurations
############################################################


class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    # CHANGED
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    # CHANGED
    NUM_CLASSES = 1 + 3  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    #IMAGE_MIN_SCALE = 2.0
    IMAGE_MIN_SCALE = 1.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class NucleusDataset(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        # CHANGED
        self.add_class("nucleus", 1, "l")
        self.add_class("nucleus", 2, "h")
        self.add_class("nucleus", 3, "n")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            image_ids = next(os.walk(dataset_dir))[1]
            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add images
        # CHANGED

        for image_id in image_ids:
            image_id: str
            '''
            if image_id.startswith("[l]"):
                tmp = "l"
            elif image_id.startswith("[h]"):
                tmp = "h"
            elif image_id.startswith("[n]"):
                tmp = "n"
            else:
                raise "unknown image id"
            '''

            self.add_image(
                #tmp,
                "nucleus",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = [] # mask图
        arr = [] # mask类别
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                # CHANGED
                if f.startswith("[l]"):
                    tmp = 1
                elif f.startswith("[h]"):
                    tmp = 2
                elif f.startswith("[n]"):
                    tmp = 3
                else:
                    print(f)
                    raise "Unknown mask class"
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
                arr.append(tmp)

        mask = np.stack(mask, axis=-1)
        arr = np.array(arr,dtype=np.int32)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        # return mask, np.ones([mask.shape[-1]], dtype=np.int32)
        return mask, arr

    '''原来
     def load_mask(self, image_id):
            """Generate instance masks for an image.
           Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
            """
            info = self.image_info[image_id]
            # Get mask directory from image path
            mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
    
            # Read mask files from .png image
            mask = []
            for f in next(os.walk(mask_dir))[2]:
                if f.endswith(".png"):
                    # TODO: 或许这里可以写个 f.startswith ('[h]')这样的函数？
                    m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                    mask.append(m)
            mask = np.stack(mask, axis=-1)
            # Return mask, and array of class IDs of each instance. Since we have
            # one class ID, we return an array of ones
            # return mask, np.ones([mask.shape[-1]], dtype=np.int32)
            # CHANGED
            if info["id"][:3] == "[l]":
                tmp = 1
            elif info["id"][:3] == "[h]":
                tmp = 2
            elif info["id"][:3] == "[n]":
                tmp = 3
            else:
                raise "Unknown mask class"
            arr = np.zeros([mask.shape[-1]], dtype=np.int32) + tmp  # 设置mask类别
            return mask, arr
    '''

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = NucleusDataset()
    dataset_train.load_nucleus(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleusDataset()
    dataset_val.load_nucleus(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='all')


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset()
    dataset.load_nucleus(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    APs = []
    for image_id in dataset.image_ids:
        _1, _2, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config,
                                   image_id)
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        print(f"{image_id}的AP是：{AP}")
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)
    print(f"mAP= {np.mean(APs)}")


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NucleusConfig()
    else:
        config = NucleusInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco" or "迁移源" in args.weights:
        print("迁移学习或重新训练")
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
