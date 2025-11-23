## MLOps-RainPrediction ☔️

This project demonstrates the creation of an end-to-end **MLOps (Machine Learning Operations)** system for rain prediction. It focuses on building robust, automated pipelines for continuous model training, monitoring, and deployment.

-----

## 🎯 Key Features

  * **Automation & Orchestration:** Uses **Apache Airflow** to manage and schedule both the training and monitoring pipelines.
  * **Model & Experiment Tracking:** Leverages **MLflow** for tracking experiments, managing model versions, and maintaining the Model Registry.
  * **Data Drift Monitoring:** Implements **Data Drift** detection to automatically assess the necessity for model retraining.
  * **Data Version Control:** Utilizes **DVC (Data Version Control)** to manage and version datasets, ensuring reproducibility.
  * **Deployment:** The predictions are served via a web application built using **Streamlit**.

-----

## 🏗️ System Architecture

The MLOps system is designed to run on **Google Cloud Platform (GCP)** and consists of three main cooperating components: the Monitoring Pipeline, the Training Pipeline, and the Model Management system.

### 1\. Tools and Core Technologies

| Tool/Platform | Role |
| :--- | :--- |
| **Apache Airflow** | Orchestration for controlling the sequence and scheduling of the pipelines |
| **MLflow** | Model Tracking, Model Registry, and Experiment Management for the model lifecycle |
| **Streamlit** | Web Application Framework for serving the deployed production model |
| **DVC (Data Version Control)** | Used for managing dataset versions to ensure reproducibility |
| **Evidently AI** (Implied) | Used within the Monitoring Pipeline to generate Data Drift Reports |
| **Google Cloud Platform (GCP)** | Cloud infrastructure hosting all services |

### 2\. Pipelines Flow

<img width="873" height="680" alt="Mlops drawio (2)" src="https://github.com/user-attachments/assets/2f7d6aa6-b074-427e-aa35-58f6b4336870" />

#### ⚙️ Monitoring Pipeline (Airflow)

The purpose of this pipeline is to continuously check the integrity and relevance of the data flowing into the system and decide if a retraining is necessary:

1.  **Fetch Data From API:** Pulls the latest data for prediction.
2.  **Preprocessing & Feature Selection:** Cleans and prepares the data.
3.  **Data Drift Evidently Report:** Generates a Data Drift report and compares it against baselines logged in **MLflow Data Drift Experiments**.
4.  **Decide Retrain Model:** Checks the specified condition (e.g., drift threshold).
5.  **If True:** If retraining is required (Condition = true), it triggers the **Training Pipeline**.

#### 🧠 Training Pipeline (Airflow)

This pipeline is triggered when the model needs to be updated:

1.  **Train Model:** Trains a new model using the latest dataset.
2.  **Save and Promote Best Model:** Logs the best-performing model (Best Model) and its artifacts to **MLflow Model Experiments** and promotes it to the **MLflow Model Registry**.
3.  **Upload Dataset to DVC:** Manages the versioning of the dataset used for the latest training run.

#### 🚀 Deployment & Serving

The model that is approved and designated as the **Production Model** in the **MLflow Model Registry** is loaded and utilized by the **Streamlit** application, allowing users to access real-time predictions.

