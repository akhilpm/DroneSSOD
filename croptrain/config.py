# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_ubteacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

    _C.SOLVER.IMG_PER_BATCH_LABEL = 1
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    _C.SOLVER.FACTOR_LIST = (1,)

    _C.DATASETS.TRAIN_LABEL = ("visdrone_2019_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("visdrone_2019_test",)
    _C.DATASETS.CROSS_DATASET = False
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.USE_SEMISUP = False
    _C.SEMISUPNET.AUG_CROPS_UNSUP = False
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"

    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 42  # random seed to read data
    _C.DATALOADER.USE_RANDOM_SPLIT = False
    _C.DATALOADER.SEED_PATH = "dataseed/visdrone_sup_10.0.txt"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True


def add_croptrainer_config(cfg):
    _C = cfg
    _C.CROPTRAIN = CN()
    _C.CROPTRAIN.USE_CROPS = False
    _C.CROPTRAIN.CLUSTER_THRESHOLD = 0.1
    _C.CROPTRAIN.CROPSIZE = (320, 476, 512, 640)
    _C.CROPTRAIN.MAX_CROPSIZE = 800
    _C.CROPTEST = CN()
    _C.CROPTEST.CLUS_THRESH = 0.3
    _C.CROPTEST.MAX_CLUSTER = 5
    _C.CROPTEST.CROPSIZE = 800
    _C.CROPTEST.DETECTIONS_PER_IMAGE = 800
    _C.MODEL.CUSTOM = CN()
    _C.MODEL.CUSTOM.FOCAL_LOSS_GAMMAS = []
    _C.MODEL.CUSTOM.FOCAL_LOSS_ALPHAS = []

    _C.MODEL.CUSTOM.CLS_WEIGHTS = []
    _C.MODEL.CUSTOM.REG_WEIGHTS = []


def add_fcos_config(cfg):
    _C = cfg
    _C.MODEL.FCOS = CN()
    _C.MODEL.FCOS.NORM = "GN"
    _C.MODEL.FCOS.NUM_CLASSES = 80
    _C.MODEL.FCOS.NUM_CONVS = 4
    _C.MODEL.FCOS.SCORE_THRESH_TEST = 0.01
    _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]


def add_querydet_config(cfg):
    _C = cfg
    _C.MODEL.QUERY = CN()
    _C.MODEL.QUERY.FEATURES_WHOLE_TRAIN = [2, 3, 4, 5]
    _C.MODEL.QUERY.FEATURES_VALUE_TRAIN = [0, 1]
    _C.MODEL.QUERY.Q_FEATURE_TRAIN = [2]
    _C.MODEL.QUERY.FEATURES_WHOLE_TEST = [2, 3, 4, 5]
    _C.MODEL.QUERY.FEATURES_VALUE_TEST = [0, 1]
    _C.MODEL.QUERY.Q_FEATURE_TEST = [2]
    _C.MODEL.QUERY.QUERY_LOSS_WEIGHT = []
    _C.MODEL.QUERY.QUERY_LOSS_GAMMA  = []
    _C.MODEL.QUERY.ENCODE_CENTER_DIS_COEFF = [1.]
    _C.MODEL.QUERY.ENCODE_SMALL_OBJ_SCALE = []
    _C.MODEL.QUERY.THRESHOLD = 0.12
    _C.MODEL.QUERY.CONTEXT = 2
    _C.MODEL.QUERY.QUERY_INFER = False
    #extra configs
    _C.MODEL.CUSTOM = CN()
    _C.MODEL.CUSTOM.FOCAL_LOSS_GAMMAS = []
    _C.MODEL.CUSTOM.FOCAL_LOSS_ALPHAS = []
    _C.MODEL.CUSTOM.CLS_WEIGHTS = []
    _C.MODEL.CUSTOM.REG_WEIGHTS = []    
    _C.MODEL.CUSTOM.USE_LOOP_MATCHER = False
    _C.MODEL.CUSTOM.GRADIENT_CHECKPOINT = False
    _C.MODEL.CUSTOM.CLEAR_CUDA_CACHE = False    
    # soft nms
    _C.MODEL.CUSTOM.USE_SOFT_NMS       = False
    _C.MODEL.CUSTOM.GIOU_LOSS          = False
    _C.MODEL.CUSTOM.SOFT_NMS_METHOD    = 'linear' # gaussian
    _C.MODEL.CUSTOM.SOFT_NMS_SIGMA     = 0.5
    _C.MODEL.CUSTOM.SOFT_NMS_THRESHOLD = 0.5
    _C.MODEL.CUSTOM.SOFT_NMS_PRUND     = 0.001
    _C.MODEL.CUSTOM.HEAD_BN = False