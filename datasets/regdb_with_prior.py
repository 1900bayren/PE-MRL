import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from data.priors import PriorLoaderDual

def _read_image(path):
    img = Image.open(path).convert('RGB')
    return img

class RegDBTrainWithPrior(Dataset):
    """
    接受 RegDB().train 列表，返回：
      img: Tensor
      pid: int
      camid: int
      prior_vec: Tensor(padded_dim,)
      prior_kind: int 0=IR,1=VIS
    """
    def __init__(self, regdb_obj, transform,
                 priors_root_ir: str, priors_root_vis: str,
                 prior_dim_ir: int = 53, prior_dim_vis: int = 52, strict_prior: bool = False):
        super().__init__()
        self.samples = regdb_obj.train  # [(img_path, pid, camid, view?, modality)]
        # RegDB 原始根目录下 Thermal/Visible 文件夹
        self.data_root_ir = os.path.join(regdb_obj.dataset_dir, 'Thermal')
        self.data_root_vis = os.path.join(regdb_obj.dataset_dir, 'Visible')
        self.transform = transform

        self.prior_loader = PriorLoaderDual(
            data_root_ir=self.data_root_ir, priors_root_ir=priors_root_ir,
            data_root_vis=self.data_root_vis, priors_root_vis=priors_root_vis,
            prior_dim_ir=prior_dim_ir, prior_dim_vis=prior_dim_vis, strict=strict_prior
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, pid, camid, _, modality = self.samples[index]  # modality: 0=IR,1=VIS
        img = _read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        prior_vec, prior_kind = self.prior_loader.load(img_path, modality)
        return img, pid, camid, prior_vec, prior_kind
