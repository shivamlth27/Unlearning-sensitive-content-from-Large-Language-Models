# Unlearning Sensitive Content from Large Language Models

A framework for selectively unlearning sensitive or unwanted information from pre-trained language models while preserving their general knowledge and capabilities.

## Overview

This project implements a novel approach to machine unlearning in Large Language Models (LLMs), focusing on the "TOFU" (Task of Unlearning) scenario. It provides a systematic way to remove specific knowledge or content from pre-trained models while maintaining their overall performance on other tasks.

## Features

- **Selective Unlearning**: Target specific content or knowledge for removal
- **Knowledge Retention**: Maintain model performance on non-targeted content
- **Interactive Dashboard**: Web interface for comparing base and unlearned models
- **Configurable Training**: Flexible configuration system for unlearning parameters
- **Evaluation Metrics**: Comprehensive evaluation of unlearning effectiveness

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shivamlth27/Unlearning-sensitive-content-from-Large-Language-Models.git
cd Unlearning-sensitive-content-from-Large-Language-Models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── app.py                 # Flask web application
├── unlearn.py            # Main unlearning script
├── train.py             # Training script
├── configs/             # Configuration files
│   ├── data_config.yaml
│   └── model_config.yaml
├── utils/               # Utility modules
│   ├── data_manager.py
│   ├── model_loader.py
│   ├── unlearning_trainer.py
│   └── evaluation.py
├── templates/           # Web interface templates
│   ├── index.html
│   └── dashboard.html
└── data/               # Data directory
    ├── synthetic_author_data.json
    ├── forget_set.json
    └── retain_set.json
```

## Usage

1. **Configure the Model**:
   Edit `configs/model_config.yaml` to set your desired model parameters.

2. **Prepare Data**:
   - Place data in the `data/` directory
   - Use the provided data manager to prepare forget and retain sets

3. **Run Unlearning**:
```bash
python unlearn.py
```

4. **Launch Web Interface**:
```bash
python app.py
```

5. **Access Dashboard**:
   Open browser and navigate to `http://localhost:5000`

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `data_config.yaml`: Data loading and preprocessing settings
- `model_config.yaml`: Model architecture and training parameters

## Evaluation

The framework provides several metrics to evaluate the effectiveness of unlearning:

- Perplexity scores for forget and retain sets
- Knowledge retention ratio
- Response comparison between base and unlearned models

<p align="center">
  <img src="https://github.com/user-attachments/assets/c3670db2-fe23-455e-b1ac-c797b01b4bbf" alt="Screenshot 1" width="30%" height="300" hspace="30" />
  <img src="https://github.com/user-attachments/assets/6ade2cdf-f22e-46ff-a595-0dbbfbf2fd2e" alt="Screenshot 2" width="35%" height="300" />
</p>

<!-- <p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/e/ec/DeepSeek_logo.svg" alt="DeepSeek Logo" width="100" style="margin-right: 60px;" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg" alt="ChatGPT Logo" width="100" />
</p> -->

<!-- Table with black border -->
<!--
<table align="center" style="border: 2px solid black; border-collapse: collapse;">
  <tr>
    <td align="center" style="border: 1px solid black;">
      <img src="https://github.com/user-attachments/assets/c3670db2-fe23-455e-b1ac-c797b01b4bbf" alt="Screenshot 1" width="300" height="300" />
    </td>
    <td align="center" style="border: 1px solid black;">
      <img src="https://github.com/user-attachments/assets/6ade2cdf-f22e-46ff-a595-0dbbfbf2fd2e" alt="Screenshot 2" width="300" height="300" />
    </td>
  </tr>
  <tr>
    <td align="center" style="border: 1px solid black;">
      <img src="https://upload.wikimedia.org/wikipedia/commons/e/ec/DeepSeek_logo.svg" alt="DeepSeek Logo" width="100" />
    </td>
    <td align="center" style="border: 1px solid black;">
      <img src="https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg" alt="ChatGPT Logo" width="100" />
    </td>
  </tr>
</table>
-->

