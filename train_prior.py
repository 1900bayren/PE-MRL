from utils.logger import setup_logger
from datasets import make_dataloader_prior
from model import make_model_prior
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
# from processor_prior import do_train
from processor_prior.processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from datetime import datetime  # ← 新增
# from timm.scheduler import create_scheduler
from config import cfg

def set_seed(seed):
    import torch, random, numpy as np

    # run_id = random.randint(0, 999999)
    # seed = seed + run_id

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = False  # 设置为 True 会锁定算法顺序，增加可复现性
    torch.backends.cudnn.benchmark = True       # True 会在不同输入大小时寻找最优算法，增加速度但带来随机性


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    # 读取配置（暂不 freeze，先生成时间戳子目录并写回 cfg.OUTPUT_DIR）
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # —— 这里创建时间戳子目录，并写回到 cfg.OUTPUT_DIR ——
    # 目录形如：<原OUTPUT_DIR>/<YYYYmmdd-HHMMSS>
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_output_dir = cfg.OUTPUT_DIR if len(str(cfg.OUTPUT_DIR)) > 0 else "./outputs"
    run_dir = os.path.join(base_output_dir, timestamp)

    # 修改 cfg.OUTPUT_DIR 指向子目录
    try:
        cfg.defrost()
    except Exception:
        pass
    cfg.OUTPUT_DIR = run_dir
    cfg.freeze()
    # —— 到此之后，下游所有使用 cfg.OUTPUT_DIR 的模块都会落到该子目录 ——

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    # 创建最终输出子目录
    output_dir = cfg.OUTPUT_DIR
    print(output_dir)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader_prior(cfg)

    model = make_model_prior(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )
