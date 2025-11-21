# prefect_rain_pipeline.py
from prefect import task, flow
import subprocess
import os
import sys

# ----------------------------
# Task 1: Fetch repo from GitHub
# ----------------------------
GITHUB_REPO = "https://github.com/RattipongMark/MLOps-RainPrediction.git"
LOCAL_DIR = "/tmp/MLOps-RainPrediction"

@task
def fetch_from_github():
    if os.path.exists(LOCAL_DIR):
        subprocess.run(["git", "-C", LOCAL_DIR, "pull"], check=True)
    else:
        subprocess.run(["git", "clone", GITHUB_REPO, LOCAL_DIR], check=True)
    print(f"Repository fetched to {LOCAL_DIR}")

# ----------------------------
# Task 2: Run training
# ----------------------------
@task
def run_training():
    if LOCAL_DIR not in sys.path:
        sys.path.insert(0, LOCAL_DIR)

    from script.train_and_log_models import run_training

    os.chdir(LOCAL_DIR)
    run_training()
    print("Training completed!")

# ----------------------------
# Flow definition
# ----------------------------
@flow(name="Rain Model Pipeline")
def rain_pipeline():
    fetch_from_github()
    run_training()

if __name__ == "__main__":
    rain_pipeline()
