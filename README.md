# Art LoRA: Personalized Avatar Generation with Stable Diffusion

**Art LoRA** is a modular Python project for fine-tuning a Stable Diffusion v1.5 model using LoRA (Low-Rank Adaptation). It enables high-fidelity generation of personalized AI avatars from a small set of custom images.



## Project Structure

```
art-lora/
├── main.py # Entry point: preprocessing, training, or demo
├── notebooks/
│ └── art_lora_colab.ipynb # Colab-ready notebook copy
├── data/
│ ├── original_photos/ # Place your raw images here
│ └── processed_dataset/ # Preprocessed 512x512 images
├── model/ # Trained LoRA weights (or place pretrained model here)
├── logs/ # Runtime logs
├── src/
│ ├── config/ # Configuration classes
│ ├── data/ # Preprocessing scripts
│ ├── training/ # LoRA fine-tuning workflow
│ ├── utils/ # Logger, helpers, etc.
│ └── web/ # Gradio demo
└── requirements.txt # Python dependencies
```

---

## Features

- **Custom LoRA training:** Fine-tune Stable Diffusion v1.5 with a few images.
- **Automated preprocessing:** OpenCV face detection & square cropping.
- **Interactive demo:** Gradio web app for generating avatars from a trigger word.
- **Cross-platform aware:** Automatically adjusts for Apple Silicon systems.


## Colab Support

A fully working Google Colab notebook is available [here](https://colab.research.google.com/drive/1DdEjOhAQDVlQwS6f37KUtHMEhdHiqHSL?usp=sharing).

A copy of this notebook is also included in `notebooks/art_lora_colab.ipynb`.


## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/art-lora.git
cd art-lora
```

2. Create a virtual environment:

```
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

Place your raw images in `data/original_photos/`.

The preprocessed images will be saved to `data/processed_dataset/`.

The trained LoRA weights will be saved to `model/` or you can place an existing model into this folder to skip training.

`--prompt_name` is the unique trigger word used in prompts for your custom concept.

The main.py script allows running preprocessing, training, or demo via command-line arguments:

#### 1. Preprocess images
```
python main.py --stage preprocess
```
#### 2. Train the LoRA model
```
python main.py --stage train
```
#### 3. Launch the interactive demo
```
python main.py --stage demo
```
*(Optional: you can add `--prompt_name your_person_name` to use a custom trigger name in the demo.)*

## Apple Silicon Support

The training pipeline automatically detects Apple Silicon and disables CUDA-only features such as 8-bit Adam optimization (bitsandbytes). For CUDA-enabled GPUs, these optimizations remain active.

## Logging

All runtime logs are saved in the `logs/` directory, including:

- Preprocessing details

- Training progress

- Demo generation steps

- Contributing

Feel free to open issues or pull requests to improve the pipeline, add new features, or optimize performance.

