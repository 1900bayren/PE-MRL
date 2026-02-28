import os
import os.path as osp
import re
import random
from glob import glob

from .bases import BaseImageDataset
from config import cfg
from utils.seedg import seed_gen



class LLCM(BaseImageDataset):
    dataset_dir = 'LLCM'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(LLCM, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.idx_dir = osp.join(self.dataset_dir, 'idx')

        # 与 idkl 版一致的文件命名
        self.train_vis_file = osp.join(self.idx_dir, 'train_vis.txt')
        self.train_nir_file = osp.join(self.idx_dir, 'train_nir.txt')
        self.test_vis_file  = osp.join(self.idx_dir, 'test_vis.txt')
        self.test_nir_file  = osp.join(self.idx_dir, 'test_nir.txt')

        self.pid_begin = pid_begin   # 如需与其他数据集合并，可在 _generate_data 里 +pid_begin
        self._check_before_run()

        train, query, gallery = self._process_dir()

        if verbose:
            print("=> LLCM (MIP style) loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.idx_dir):
            raise RuntimeError("'{}' is not available".format(self.idx_dir))
        needed = [self.train_vis_file, self.train_nir_file, self.test_vis_file, self.test_nir_file]
        for p in needed:
            if not osp.exists(p):
                raise RuntimeError("'{}' is not available".format(p))

    def _read_file(self, file_path):
        """
        每行：<relative_path> <pid>
        例如：vis/0001/0001_xxx.jpg 1
        返回：[(abs_path, pid:int)]
        """
        index = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 1:
                    rel_path = parts[0]
                    pid = int(osp.basename(osp.dirname(rel_path)))
                else:
                    rel_path, pid = parts[0], int(parts[1])
                abs_path = osp.join(self.dataset_dir, rel_path)
                index.append((abs_path, pid))
        return index

    def _process_dir(self):
        train_vis = self._read_file(self.train_vis_file)
        train_nir = self._read_file(self.train_nir_file)
        test_vis  = self._read_file(self.test_vis_file)
        test_nir  = self._read_file(self.test_nir_file)

        # 训练：vis + nir 合并
        train = self._generate_data(visible_data=train_vis, thermal_data=train_nir, is_train=True)
        # 测试：query=nir，gallery=vis
        query = self._generate_data(thermal_data=test_nir, is_train=False)
        gallery = self._generate_data(visible_data=test_vis, is_train=False)

        return train, query, gallery

    def _generate_data(self, thermal_data=None, visible_data=None, is_train=False):
        """
        统一五元组：(image_path, pid, camid, 2, modality_id)
          - camid: nir→0, vis→1
          - modality_id: nir→0, vis→1
          - 第四位与 RegDB 的 MIP 保持一致，固定为 2
        """
        data = []
        if thermal_data:
            for img_path, pid in thermal_data:
                pid_out = pid + self.pid_begin  # 若不需要偏移，可改成 pid
                data.append((img_path, pid_out, 0, 2, 0))  # nir
        if visible_data:
            for img_path, pid in visible_data:
                pid_out = pid + self.pid_begin
                data.append((img_path, pid_out, 1, 2, 1))  # vis
        return data
