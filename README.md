# explain-llm-captum
This repository is for the course "Explainability of LLMs" at University of Osnbarück. It contains demo code for a presentation on "Using captum to Explain Generative Language Models".
## **Table of Contents**
- [Overview](#overview)
- [QuickStart](#quickstart)
- [Attribution Methods](#different-attribution-methods)
- [Structure of this Repository](#structure-of-this-repository)
- [References](#references)
---
## Overview
This repository is demonstarting how to use [Captum](https://github.com/meta-pytorch/captum) [[Miglani et al., 2023]](#references) for explaining Generative Language Models.
Captum is a PyTorch library for model interpretability that provides tools to analyze and understand how LLMs make predictions. It allows you to attribute the output of a model to its input features, helping answer questions like “Which tokens or words in this text most influenced the model’s prediction?”


## Different Attribution Methods
In Captum there are two main different ways of calculating Attribution.
Perturbation based methods and Gradient based Methods.

Captum includes several attribution methods such as Feature Ablation, Integrated Gradients, (both gradient-based), Shapley Value Sampling, and Kernel SHAP (both perturbation based), which can be used to evaluate token-level contributions and better understand the internal reasoning of generative models.

Per default, Captum only supports perturbation based methods for LLMAttribution, as explored in `pert_captum.ipynb`. In this notebook we use the `distilgpt2` model from huggingface, to have an easy-working example of the capabilities of the Captum library.

In the other Jupyter Notebook, `grad_captum.ipynb`, we created our own, feed-forward text classification model. Its architecture allows for gradient based methods to be evaluated. We also provided a pre-trained model (`bow_text_classifier.pt`), so that you don't have to do the training.

## QuickStart
If you do not want to run this demo locally, you can also look at it on Google Colab:
- [Perturbation-Based](https://colab.research.google.com/drive/1-T0mja-TGL2h_v4_bXKhyY_fYDd8JdEj?usp=sharing).
- [Gradient-Based]   (not yet uploaded)
Otherwise, please follow the next steps:

<!-- Fist we need to install [Git](#git) to be able to clone this repository.
Then decide, whether you want to set up your virtual environment with [venv](#venv) (built into Python) or [Conda](#conda) (a package and environment manager from Anaconda/Miniconda).

### Install Git
<a name="git"></a>
Download and install Git:

- Visit the [official Git website](https://git-scm.com/) to download the latest version of Git.
- Follow the installation instructions for your operating system.

### Clone the Git Repository

- Open a terminal or command prompt.
- Go to the directory where you want to store everything regarding the course:
```bash
cd <directory_name>
```
- Clone the Git repository:
```bash
git clone https://github.com/MeinChef/explain-llm-captum
```
- Change into the cloned repository:
```bash
cd explain-llm-captum
``` -->

### Set Up a Virtual Environment (venv)
<a name="venv"></a>

Download and install Python:
- Visit the [official Python website](https://www.python.org/) to download the latest version of Python.
- During installation, make sure to check the option that adds Python to your system's PATH.

- Create a virtual environment:
```bash 
python -m venv venv
```
- Activate the virtual environment:

--> On Windows:
```bash
.\venv\Scripts\activate
```
--> On Unix or MacOS:
```bash
source venv/bin/activate
```
- Install required packages (to make use of GPU acceleration, use `requirements-cuda_rocm.txt` instead of `requirements.txt`)
```bash
pip install -r requirements.txt
```

### Set Up a Virtual Environment (conda)
<a name="conda"></a>
- Create a virtual environment:
- Open your terminal (Command Prompt on Windows, Terminal on macOS/Linux).
<!-- 2. Navigate to the directory where you saved the environment.yml file. (This should be YOUR_PATH/UDL-Reinforcement-Learning/) -->
- Execute the following command to create the environment:

```bash 
conda create -m venv python=3.14
```
- Activate the virtual environment:
--> On Windows, Unix and MacOS:
```bash
conda activate venv
```
- Install required packages (to make use of GPU acceleration, use `requirements-cuda_rocm.txt` instead of `requirements.txt`)
```bash
pip install -r requirements.txt
```

## Structure of this Repository

```
├── .gitignore
├── grad_captum.ipynb
├── pert_captum.ipynb
├── README.md
├── requirements.txt
├── requirements-cuda.txt
```

## References
<a name="references"></a>
[1] Miglani, V., Yang, A., Markosyan, A. H., Garcia-Olano, D., & Kokhlikyan, N. (2023, December). Using Captum to Explain Generative Language Models [arXiv:2312.05491[cs]]. https://doi.org/10.48550/arXiv.2312.05491
