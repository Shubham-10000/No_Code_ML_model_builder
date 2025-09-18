# 🚀 No-Code ML Builder

A Streamlit-based application to build, train, and evaluate machine learning models without writing code.  
Simply upload your dataset, preprocess it, choose algorithms, and get instant results.

## Features
- 📂 Upload and manage CSV files
- 🧹 Preprocess data (null handling, encoding, etc.)
- 🤖 Train ML models (classification & regression)
- 🎯 Select the best model based on evaluation metrics
- 💾 Save trained models for reuse

## How to use
1. Upload your dataset
2. Preprocess and clean data
3. Train multiple ML models
4. Select the best-performing model
5. Download and reuse saved models

## Run locally
```bash
pip install -r requirements.txt  
streamlit run app.py
# ML Project

This project is a Streamlit application designed for machine learning tasks, including file management, data preprocessing, model training, and model selection. Below is an overview of the project's structure and functionalities.

## Project Structure

```
ML_Project
├── app.py                # Main entry point of the Streamlit application
├── file_manage.py        # Handles file management functionalities
├── preprocessing.py      # Contains functions for data preprocessing
├── model_training.py     # Responsible for training and evaluating machine learning models
├── model_selection.py     # Allows users to select a specific model and determine problem type
├── uploads               # Directory for storing uploaded CSV files
├── preprocessed_data     # Directory for preprocessed CSV files
├── mapping               # Directory for JSON files with label encoding mappings
├── saved_models          # Directory for saving trained models
└── README.md             # Documentation for the project
```

## Installation

To run this project, ensure you have Python installed on your machine. You can install the required packages using pip:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn joblib
```

## Usage

1. **Run the Application**: Navigate to the project directory and run the following command:

   ```bash
   streamlit run app.py
   ```

2. **File Management**: Use the "File Management" option in the sidebar to upload CSV files and manage them.

3. **Data Preprocessing**: Select the "Preprocessing Data" option to preprocess your uploaded CSV files. This includes viewing data, analyzing null values, and encoding categorical variables.

4. **Model Training**: Choose the "Model Training" option to train and evaluate various machine learning models on your preprocessed data.

5. **Model Selection**: Use the "Model Selection" option to select a specific model and determine whether the problem is a classification or regression problem.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.



--
