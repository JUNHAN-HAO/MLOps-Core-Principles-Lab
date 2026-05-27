# 🚀 MLOps-Core-Principles-Lab

**A practical Machine Learning Operations (MLOps) laboratory focusing on Reproducibility, Continuous Integration (CI), and Open Source Contribution.**

---

## 📌 Overview
Building a machine learning model in a Jupyter Notebook is only the first step of the data science lifecycle. In the real world, models must be tracked, tested, and deployed reliably. This repository serves as the baseline for our MLOps algorithm comparison lab. 

Currently, this repository contains a baseline training script (`train.py`) that successfully trains a model but violates core MLOps principles: it lacks traceability (we don't know how well the model performed after training) and it lacks automated testing (Continuous Integration). 

Your mission is to upgrade this pipeline, compare classical ML against Deep Learning, and submit your improvements back to this repository using the industry-standard **Fork and Pull Request** workflow.

---

## 🎯 Learning Objectives
By completing this lab, you will learn how to:
1. **Engineer and Compare Models:** Evaluate Classical ML (e.g., Logistic Regression, SVM, XGBoost) against a 4-layer Deep Learning Neural Network.
2. **Implement Traceability:** Programmatically log model metrics (Accuracy, F1-Score) to JSON files for experiment tracking.
3. **Enforce Continuous Integration (CI):** Write automated Python tests that block poorly performing models from reaching "production."
4. **Master Open Source Collaboration:** Use Git commands to clone, commit, push, and open a Pull Request.

---

## 🛠️ The Git Workflow (Why Fork and PR?)
In traditional software engineering, developers rarely push code directly to the `main` branch. Doing so risks overwriting a colleague's work or breaking the production system. Instead, we use the **Fork and Pull Request (PR)** model.

* **Forking:** Creates an exact, personal copy of this repository under your own GitHub account. You can experiment safely here without affecting the main class repository.
* **Pull Request (PR):** Once your code is ready, you "request" that the repository administrator (the Professor) "pull" your upgraded code into the main repository. This allows for code review and automated testing before the code is merged.

---

## 🚀 Step-by-Step Instructions & Commands

### Step 1: Fork the Repository
1. Navigate to the top right of this page.
2. Click the **Fork** button.
3. Select your personal GitHub account as the destination. You now have your own copy of this lab.

### Step 2: Clone the Repository to your Environment
Open Google Colab (or your local terminal) and set up your Git credentials. 

```bash
# Set your Git identity
git config --global user.name "Your Name"
git config --global user.email "your.email@xu.edu"

# Clone your forked repository (Replace YOUR_USERNAME and use your Personal Access Token)
git clone [https://YOUR_TOKEN@github.com/YOUR_USERNAME/MLOps-Core-Principles-Lab.git](https://YOUR_TOKEN@github.com/YOUR_USERNAME/MLOps-Core-Principles-Lab.git)

# Move into the project directory
cd MLOps-Core-Principles-Lab
