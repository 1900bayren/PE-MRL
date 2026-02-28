# ===================== 模块2：多参数联合优化实验（新增核心，三参数均取不等值）=====================
echo -e "\n==== 开始：多参数联合优化实验 ===="
# 联合实验候选值（围绕单参数最优值）
alpha_candidates=(0.8 1.0 1.2)   # MARP权重
beta_candidates=(0.4 0.5 0.6)    # PGMP权重
gamma_candidates=(0.8 0.9 1.0)  # ADG权重
# 三重循环遍历所有组合（3×3×3=27组）
for alpha in "${alpha_candidates[@]}"; do
  for beta in "${beta_candidates[@]}"; do
    for gamma in "${gamma_candidates[@]}"; do
      echo "==== 联合实验：α=$alpha, β=$beta, γ=$gamma ===="
      # 输出目录：joint_α_β_γ，清晰区分联合实验
      python train_adr.py --config_file configs/RegDB_P/RegDB_adr.yml MODEL.DEVICE_ID "('0')" OUTPUT_DIR "logs_P/joint_${alpha}_${beta}_${gamma}" SOLVER.BASE_LR 0.01 SOLVER.EVAL_PERIOD 1 INPUT.AUG 2 \
      INPUT.SIZE_TRAIN [256,128] INPUT.SIZE_TEST [256,128] MODEL.MSEL True MODEL.MSEL_EPOCH 20 DATASETS.SAMPLER 'modal' MODEL.MSEL_MODAL True DATALOADER.NUM_INSTANCE 8 \
      MODEL.USE_PROMPT True MODEL.NUM_TOKEN 16 MODEL.PROMPT_SCALE 20.0 MODEL.PROMPT_SHIFT 0.0 MODEL.USE_INS_PROMPT True MODEL.USE_INS_PROMPT_GEN True MODEL.NUM_INS_PMT_TOKEN 16 \
      SOLVER.WEIGHT_IPLR 15.0 SOLVER.WEIGHT_MPLR 5.0 MODEL.IPIL False SOLVER.MAX_EPOCHS 120 TEST.EVAL_EPOCH 10 SOLVER.SCHEDULER 'cosine-refine' SOLVER.MIN_INDEX 0.00001 SOLVER.COSINE_EPOCHS 40 SOLVER.SEED 3 \
      SOLVER.WEIGHT_MOD $alpha SOLVER.WEIGHT_PHYS $beta SOLVER.WEIGHT_ADR_PROXY $gamma
    done
  done
done

echo -e "\n==== 所有实验完成：单参数实验日志在logs_P/single_*，联合实验日志在logs_P/joint_* ===="