# DiffTORI Reproduction (PushT imitation Learning Task)

This project is a minimal reproduction of **DiffTORI: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning**, focused on the `PushT` task using the `gym_pusht` environment.

The original paper proposes a hybrid framework combining CVAE-based policy priors with differentiable trajectory optimization. This repo replicates core components for experimentation and understanding.

## 🛠️ Installation

```bash
pip install git+https://github.com/huggingface/lerobot.git
pip install gym_pusht
# pip install theseus-ai
cd theseus
pip install -e .
pip install torch numpy matplotlib
```
🚀 Usage
Clone the repository:

```bash
git clone https://github.com/LawrenceLinn/difftoi.git
cd difftoi
```

Train on the PushT task with imitation learning:

```bash
python train.py 
```

Evaluate the trained model:

```bash
# python eval.py
```

📄 Citation
Please cite the original paper if you use this work:
```bib
@article{wan2025difftori,
  title={DiffTORI: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning},
  author={Wan, Weikang and Wang, Ziyu and Wang, Yufei and Erickson, Zackory and Held, David},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={109430--109459},
  year={2025}
}
```