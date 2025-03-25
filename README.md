# Project_04: Diabetes Prediction

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Colaborators](#colaborators)
- [Data](#data)

## Introduction
This project aims to provide a practical example of how to build and apply various machine learning models. Specifically, this project focuses on using machine learning to predict whether or not someone might have diabetes.

## Models
- **Logistic Regression**: A statistical model that in its basic form uses a logistic function to model a binary dependent variable.
- **Random Forest**: An ensemble learning method for classification, regression, and other tasks that operates by constructing a multitude of decision trees.
- **Neural Network**: A series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.
- **Gradient Boosting**: A machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.

## Installation
To install and run this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/CamdenBeck/Project_04.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Project_04
    ```
3. Create a new environment:
    ```sh
    conda create --name {environment_name} python=3.8
    ```
4. Activate the environment:
    ```sh
    conda activate {environment_name}
    ```
5. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To train the models yourself, examine the code in the notebook, make edits as needed, and run the code in [Google Colab](https://colab.research.google.com/).
**Warning:** This will take a long time.

To try out the pretrained models, follow these steps:

1. Run the [main.py](main.py) file:
   ```sh
   python main.py
   ```
2. In the dropdown menu, select the desired pretrained model.
3. Input the correct values into the text boxes.

**Disclaimer:** 

These are machine learning models and not at all meant to be used for a medical diagnosis. They are only meant for educational purposes to showcase machine learning and should not be used as a substitute for professional medical advice. Always consult a healthcare professional for any medical concerns.

## Colaborators
- [Camden Beck](https://github.com/CamdenBeck)
- [Alex Gerwer](https://github.com/AlexGerwer)
- [Sylvester Gold](https://github.com/Sylvesterg95)
- [Sarah Gutierrez](https://github.com/SarahGR22)

## Data
The dataset used in this project is in the public domain. The objective is to predict based on health incidators whether a patient has diabetes.

**Reference:**
Diabetes Health Indicators Dataset. https://www.kaggle.com/uciml/pima-indians-diabetes-database. Accessed 19 March. 2025.