# Gauge-Reading-MM

This repository accompanies our paper submission:

**"Language as a Supervisor: A Text-Driven Visual Learning Approach for General Purpose Meter Reading"**

---

## 📌 Code Status

The code is currently being cleaned and organized.  

We have uploaded:

- ✅ Fine-tuning code (`src/`)
- ✅ Training and evaluation dataset (`dataset/`)
- ✅ Configuration (`src/training_config.py`)
- 🛠️ Inference and evaluation scripts will be added soon

---

## 🔄 Updates

- **[2025-07]** Initial release with fine-tuning pipeline and dataset  
- **Next:** Inference demo and model checkpoints

---

## 🛠️ Fine-tuning Usage

We provide scripts for constructing the dataset and training the model from scratch.

### 📁 Project Structure

```bash
Gauge-Reading-MM/
│
├── src/
│   ├── build_dataset.py       # Script to preprocess/construct training data
│   ├── finetuning.py          # Fine-tuning entry point
│   ├── training_config.py     # Training parameters and config
│
├── dataset/
│   ├── train/                 # Training images
│   └── eval/                  # Evaluation set
│       ├── images/            # Test/eval images
│       └── labels.json        # Ground truth labels
│
├── model/                     # Saved model weights (e.g., best_model.pth)
└── save/                      # Optional directory for logs, checkpoints, etc.