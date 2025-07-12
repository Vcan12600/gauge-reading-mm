---
# Gauge-Reading-MM

This repository accompanies our paper submission:

**"Language as a Supervisor: A Text-Driven Visual Learning Approach for General Purpose Meter Reading"**

---

## 📌 Code Status

The code is currently being cleaned and organized.

We have uploaded:

* ✅ Fine-tuning code (`src/`)
* ✅ Training and evaluation dataset (`dataset/`)
* ✅ Configuration (`src/training_config.py`)
* ✅ Inference script (`src/inference.py`)
* 🛠️ Evaluation utilities will be added soon

---

## 🔄 Updates

* **\[2025-07]** Initial release with fine-tuning pipeline and dataset
* **\[2025-07]** Added inference script and model checkpoint
* **Next:** Inference demo notebook and evaluation utilities

---

## 🛠️ Fine-tuning Usage

We provide scripts for constructing the dataset and training the model from scratch.

### 📁 Project Structure

* **Gauge-Reading-MM/**

  * **src/**

    * `build_dataset.py` – Script to preprocess/construct training data
    * `finetuning.py` – Fine-tuning entry point
    * `training_config.py` – Training parameters and config
    * `inference.py` – 🔍 Inference script
  * **dataset/**

    * `train/` – Training images
    * `eval/` – Evaluation set

      * `images/` – Test/eval images
      * `labels.json` – Ground truth labels
  * **model/** – Saved model weights (e.g., `best_model.pth`)
  * **save/** – Optional directory for logs, checkpoints, etc.


---

## 🧪 Inference

You can run inference using the provided script:

```bash
python src/inference.py --image path/to/image.jpg --model_path model/model.safetensors --prompt reading
````

Available arguments:

* `--image`: Path to the input image
* `--model_path`: Path to the trained model checkpoint (`.safetensors` or `.pth`)
* `--prompt`: Inference mode prompt, must be one of:

  * `norm`: Read normalized meter value
  * `reading`: Read real-world meter value



### 📥 Model Weights

Model checkpoints can be downloaded from the following link:

* **Download URL:** [https://pan.baidu.com/s/16Dtp1GJ4r0SJC9OLj2g8Zg?pwd=85gv](https://pan.baidu.com/s/16Dtp1GJ4r0SJC9OLj2g8Zg?pwd=85gv)
* **(Extraction Code):** `85gv`

After downloading, place the `.pth` file into the `model/` directory before running inference.

---

## 📬 Contact

If you have any questions, feedback, or issues, feel free to:

* Open an [issue](https://github.com/Vcan12600/gauge-reading-mm/issues)
* Submit a pull request
* Contact us via email at **[vacanth0126@gmail.com](mailto:vacanth0126@gmail.com)**

We appreciate your interest and contributions!

```
```
