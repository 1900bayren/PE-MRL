### PE-MRL: Physics-enhanced Modality Representation Learning for Visible-infrared Person Re-identification
This is the official implementation of our paper "PE-MRL: Physics-enhanced Modality Representation Learning for Visible-infrared Person Re-identification", which has been submitted to Neurocomputing (March 2026).
# Environment Setup
Python Version: 3.10
Hardware: 2 × NVIDIA RTX 5090 GPUs
Dependencies: Install all required packages via:
bash
pip install -r requirements.txt

# Model Weights
The pre-trained model weights can be downloaded from the following Google Drive link:
https://drive.google.com/file/d/1dYalQd-JrpoDRKbmXO_7Q7RKXqSgES7v/view?usp=sharing

The needed ViT pretrain model when you try to train your own model:
https://drive.google.com/file/d/1-6-4YiqzemUVQ_Ha1w-Cn9VUlOzu6SZW/view?usp=drive_link
https://drive.google.com/file/d/1HcmL5QjAF4cSukwB3DvkWVI8UqtKkT00/view?usp=drive_link

# Training
To train the PE-MRL model, run:
bash
bash train.sh
# Evaluate on LLCM
bash eval_llcm.sh

# Evaluate on RegDB
bash eval_regdb.sh

# Evaluate on SYSU-MM01
bash eval_sysu.sh
📁 Project Structure
plaintext
.
├── model/          # Model definitions
├── processor/      # Data processing modules
├── processor_adr/  # ADR-based data processing
├── processor_prior/ # Prior-based data processing
├── solver/         # Optimization and training logic
├── utils/          # Utility functions
├── .gitattributes
├── LICENSE
├── README.md
├── adr.sh
├── eval_llcm.sh
├── eval_regdb.sh
├── eval_sysu.sh
├── grad.py
├── grad_cam.py
├── multi_para.sh
├── para.sh
├── regdb_p.sh
├── requirements.txt
├── run_for.sh
├── test.py
├── train.py
├── train.sh
├── train_adr.py
├── train_prior.py
└── vis.py
📝 Citation
If you find this work useful for your research, please consider citing our paper:
bibtex
@article{PE-MRL2026,
  title={PE-MRL: Physics-enhanced Modality Representation Learning for Visible-infrared Person Re-identification},
  author={Tianmao Cui, Ze Tao, Jian Zhang, Shichao Zhang and et al.},
  journal={Neurocomputing},
  year={2026},
  note={Submitted}
}
