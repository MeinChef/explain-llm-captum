# explain-llm-captum
This repository is for the course "Explainability of LLMs" at University of Osnbarück. It contains demo code for a presentation on "Using captum to Explain Generative Language Models".
## **Table of Contents**
- [Overview](#overview)
- [QuickStart](#quickstart)
- [Structure of this Repository](#structure-of-this-repository)
- [References](#references)
---
## Overview
This repository is demonstarting how to use [Captum](https://github.com/meta-pytorch/captum) [[1]](#references) for explaining Generative Language Models.
Captum is a 
## QuickStart
If you do not want to run this demo locally, you can also look at it on [google collab](https://colab.research.google.com/drive/1-T0mja-TGL2h_v4_bXKhyY_fYDd8JdEj?usp=sharing).
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
├── main.ipynb
├── README.md
├── requirements.txt
├── requirements-cuda.txt
```

## References
<a name="references"></a>
[1] Miglani, V., Yang, A., Markosyan, A. H., Garcia-Olano, D., & Kokhlikyan, N. (2023, December). Using Captum to Explain Generative Language Models [arXiv:2312.05491[cs]]. https://doi.org/10.48550/arXiv.2312.05491
