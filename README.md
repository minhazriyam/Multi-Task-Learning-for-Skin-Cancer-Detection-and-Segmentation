# Skin Cancer Multi-Task Learning Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)

This repository contains the implementation of a multi-task deep learning model for simultaneous skin cancer segmentation and classification using the HAM10000 dataset.
## Project Overview

Skin cancer is a prevalent global health concern, requiring early detection and localization for effective treatment. This project presents a multi-task learning (MTL) approach that leverages a shared ResNet18 encoder to perform:
- **Segmentation**: Identifying lesion boundaries in dermatoscopic images using a U-Net-style decoder.
- **Classification**: Diagnosing seven skin cancer types (MEL, NV, BCC, AKIEC, BKL, DF, VASC).

The model is trained on the HAM10000 dataset, achieving a Dice score of ~0.91 for segmentation and a classification accuracy of ~70% on the validation set.

## Features

- **Multi-Task Architecture**: Combines segmentation and classification using a shared ResNet18 encoder.
- **Custom Loss Function**: Balances Binary Cross Entropy (segmentation) and Cross Entropy (classification) losses.
- **Evaluation Metrics**:
  - Segmentation: Dice Score (~0.91)
  - Classification: Accuracy (~70%), Confusion Matrix
- **Visualization**: Includes functions to visualize predicted masks and classification results.
- **Efficient Training**: Utilizes PyTorch DataLoaders, Adam optimizer, and GPU acceleration.

## Dataset

The model is trained and evaluated on the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T), which contains:
- 10015 dermatoscopic images with corresponding segmentation masks.
- Labels for seven skin cancer classes: Melanoma (MEL), Melanocytic Nevus (NV), Basal Cell Carcinoma (BCC), Actinic Keratosis (AKIEC), Benign Keratosis (BKL), Dermatofibroma (DF), and Vascular Lesion (VASC).

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- tqdm

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/minhazriyam/Multi-Task-Learning-for-Skin-Cancer-Detection-and-Segmentation.git
   cd skin-cancer-multitask-learning
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the HAM10000 dataset and place it in the `data/` directory. Ensure the dataset is structured as follows:
   ```
   data/
   ├── images/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── masks/
   │   ├── image1_mask.png
   │   ├── image2_mask.png
   │   └── ...
   └── GroundTruth.csv
   ```

## Usage

1. **Run the Notebook**:
   The main implementation is provided in `Skin_Cancer_Multitask_Project.ipynb`. Open it in Jupyter Notebook or Google Colab:
   ```bash
   jupyter notebook Skin_Cancer_Multitask_Project.ipynb
   ```

2. **Training**:
   - Configure hyperparameters (e.g., learning rate, epochs, batch size) in the notebook.
   - Run the training cells to train the model. The model will save checkpoints in the `checkpoints/` directory.

3. **Evaluation**:
   - Use the provided evaluation cells to compute Dice scores and classification accuracy on the validation set.
   - Visualize results using the visualization functions.

4. **Inference**:
   - Load a trained model checkpoint and run inference on new images using the inference cells.

## Model Architecture

The multi-task model consists of:
- **Shared Encoder**: Pretrained ResNet18 to extract robust features.
- **Segmentation Decoder**: U-Net-style architecture for pixel-wise mask prediction.
- **Classification Head**: Fully connected layers for seven-class prediction.
- **Loss Function**: Weighted combination of Binary Cross Entropy (segmentation) and Cross Entropy (classification).

## Results

After training for 5 epochs:
- **Segmentation**: Dice Score ~0.91
- **Classification**: Accuracy ~70%

Learning curves show stable convergence, with segmentation performance improving steadily and classification achieving high accuracy for common classes. Visualizations confirm accurate lesion boundary detection and reliable predictions for most classes, with challenges noted for rare classes (e.g., DF, VASC) due to class imbalance.

## Discussion

### Benefits of Multi-Task Learning
- **Efficiency**: Shared encoder reduces computational and memory overhead.
- **Performance**: Complementary tasks enhance feature learning, leading to robust representations.
- **Convergence**: Faster convergence compared to single-task models.

### Challenges
- **Data Synchronization**: Handled differences in segmentation and classification data.
- **Loss Balancing**: Adjusted weights to prevent one task from dominating.
- **Class Imbalance**: Rare classes (e.g., DF, VASC) showed lower performance, suggesting the need for data augmentation.

### Future Work
- **External Validation**: Test on diverse datasets to ensure generalizability.
- **Explainability**: Implement Grad-CAM for visual explanations.
- **Uncertainty Estimation**: Use Bayesian methods to highlight low-confidence predictions.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


