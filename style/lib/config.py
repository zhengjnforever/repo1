import numpy as np

class Config(object):
# Name the configurations. For example, 'COCO', 'Experiment 3', ...etc. 命名配置。
# Useful if your code needs to do things differently depending on which
# experiment is running.如果您的代码需要根据正在运行的实验进行不同的操作，则很有用。
    NAME = None
# Override in sub-classes在子类中重写

# NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.设置GPU的使用数量，若用CPU则设为1
    GPU_COUNT = 1

# Number of images to train with on each GPU. A 12GB GPU can typically 每个GPU上要训练的图像数
# handle 2 images of 1024x1024px.
# Adjust based on your GPU memory and image sizes. Use the highest根据GPU内存和图像大小进行调整。
# number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

# Number of training steps per epoch 每次迭代的训练步骤数
# This doesn't need to match the size of the training set. Tensorboard这不需要匹配训练集的大小。
# updates are saved at the end of each epoch, so setting this to a
# smaller number means getting more frequent TensorBoard updates.
# 这不需要匹配训练集的大小。tensorboard更新保存在每个历元的末尾，因此将其设置为较小的数字意味着获得更频繁的tensorboard更新。
# Validation stats are also calculated at each epoch end and they
# might take a while, so don't set this too small to avoid spending
# a lot of time on validation stats.
# 验证统计数据也会在每个历元结束时计算，它们可能需要一段时间，因此不要设置得太小，以免在验证统计数据上花费太多时间
    STEPS_PER_EPOCH = 600
#1000
# Number of validation steps to run at the end of every training epoch.在每个训练阶段结束时要运行的验证步骤数。
# A bigger number improves accuracy of validation stats, but slows更大的数字可以提高验证统计的准确性，但会减慢训练速度。
# down the training.
    VALIDATION_STEPS = 50
#50
# Backbone network architecture骨干网结构
# Supported values are: resnet50, resnet101.支持的值为：resnet50、resnet101。
# You can also provide a callable that should have the signature
# of model.resnet_graph. If you do so, you need to supply a callable
# to COMPUTE_BACKBONE_SHAPE as well您还可以提供一个应具有model.resnet_graph签名的可调用。如果这样做，则还需要提供一个可调用的计算主干形状
    BACKBONE = "resnet101"

# Only useful if you supply a callable to BACKBONE. Should compute
# the shape of each layer of the FPN Pyramid.
# See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

# The strides of each layer of the FPN Pyramid. These values
#FPN金字塔每一层的步幅
# are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

# Size of the fully-connected layers in the classification graph分类图中完全连通层的大小
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

# Size of the top-down layers used to build the feature pyramid用于构建特征棱锥体的自顶向下层的大小
    TOP_DOWN_PYRAMID_SIZE = 256

# Number of classification classes (including background)分类类别数（含背景）
    NUM_CLASSES = 1+13  # Override in sub-classes 在子类中重写

# Length of square anchor side in pixels正方形锚定边的长度（像素）
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

# Ratios of anchors at each cell (width/height)
#每个单元的锚栓比率（宽度 / 高度）
# 值1表示方锚，值0.5表示宽锚
# A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

# Anchor stride
# If 1 then anchors are created for each cell in the backbone feature map.
# If 2, then anchors are created for every other cell, and so on.
# 锚定跨距
# 如果为1，则为主干要素地图中的每个单元创建定位。
# 如果为2，则为每个其他单元格创建定位，依此类推。
    RPN_ANCHOR_STRIDE = 1

# Non-max suppression threshold to filter RPN proposals.
# You can increase this during training to generate more propsals.
# 过滤RPN建议的非最大抑制阈值。
# 你可以在训练中增加这个来产生更多的推进。
    RPN_NMS_THRESHOLD = 0.7

# How many anchors per image to use for RPN training每个图像要用于RPN训练的锚数
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    
# ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 6000

# ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

# If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    #如果启用，则将实例掩码调整为较小的大小以减少内存负载。使用高分辨率图像时推荐
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads用于训练分类器/掩模头的正roi百分比
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image在一个图像中使用的地面实况实例的最大数目
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.掩模RCNN使用lr=0.02，但在TensorFlow上，它会导致权重爆炸。可能是由于优化器实现的差异0.001
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization 权衰减正则化
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    # 为更精确的优化而损失权重。
    # 可用于R-CNN训练设置。
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    # 训练或冻结批处理规范化层

    # None：训练BN层。这是正常模式

    # False：冻结BN层。当使用小批量时很好

    # True：（不要用）。即使在预测时也将图层设置为训练模式
    TRAIN_BN = False  # Defaulting to False since batch size is often small # 默认为False，因为批大小通常很小

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        """Set values of computed attributes.设置计算属性的值。"""
        # Effective batch size 有效批量
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size 输入图片大小
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                self.IMAGE_CHANNEL_COUNT])

        # Image meta data length图像元数据长度
        # See compose_image_meta() for details有关详细信息，请参见compose_image_meta（）
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """Display Configuration values.显示配置值"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")