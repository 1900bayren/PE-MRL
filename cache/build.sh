
# /root/code/IDKL-main/IDKL/configs/dataset/SYSU-MM01/cam3
# /root/code/IDKL-main/IDKL/configs/dataset/RegDB/Thermal
# /root/code/IDKL-main/IDKL/configs/dataset/LLCM/test_nir
# /root/code/IDKL-main/IDKL/configs/dataset/LLCM/nir
## 对训练集 IR 图像生成先验
python /root/code/AMaP/cache/build_physical_priors_amap_full.py \
  --data-root /root/code/IDKL-main/IDKL/configs/dataset/LLCM/nir \
  --priors-root /root/code/AMaP/cache/LLCM/nir \
  --modality ir

# /root/code/IDKL-main/IDKL/configs/dataset/LLCM/vis
# /root/code/IDKL-main/IDKL/configs/dataset/RegDB/Visible
# 对训练集 RGB 图像生成先验
#python /root/code/AMaP/cache/build_physical_priors_amap_full.py \
#  --data-root /root/code/IDKL-main/IDKL/configs/dataset/RegDB/Visible \
#  --priors-root /root/code/AMaP/cache/RegDB/Visible \
#  --modality rgb
