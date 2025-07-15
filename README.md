## Adult Income Prediction & Fairness Analysis

**Note to Study.com graders**

The data is being retrieved from openml directly. A CSV need not be included in the project. If you'd prefer that I do all of my coding in the jupyter notebook cells then I can accomodate that. Just let me know.

This repository contains a complete machine learning pipeline for the UCI Adult Income dataset, with a focus on bias detection and fairness evaluation. It is fully containerized using Docker, so you can run everything locally without installing Python or dependencies directly on your machine.

### Prerequisites

- **Docker** (or an alternative like Colima)
- **Make** (optional, for shortcut commands)

### Getting Started

1. **Clone the repository**

2. **Build and launch the container**

   If you have **Docker Compose** support in your Docker CLI:

   ```bash
   make up       # builds the image and starts the container
   ```

   Or manually:

   ```bash
   docker compose build --no-cache
   docker compose up
   ```

3. **Open JupyterLab** In your browser, navigate to:

   ```
   http://localhost:8888
   ```

   You will see the project files and the `notebooks/01-explore.ipynb` notebook.

### Project Structure

```
├── Dockerfile           # Defines the container environment
├── environment.yml      # Conda spec for reproducible deps
├── Makefile             # `make up` shortcut for docker compose
├── data/                # Raw & processed CSV data
│   ├── raw/
│   └── processed/
├── notebooks/           # Jupyter notebooks for EDA, modeling, and fairness analysis
│   └── 01-explore.ipynb
├── src/                 # Python modules for data prep and modeling
└── reports/figures/     # Generated plots (confusion matrices, fairness bar charts)
```

### Workflow Overview

1. **Data ingestion**: `src/data.py` fetches, cleans, and splits the Adult dataset (60/20/20).
2. **Modeling**: `src/modeling.py` defines preprocessing pipelines and trains multiple models (Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting).
3. **Evaluation**:
   - Overall metrics (accuracy, precision, recall, F1)
   - Confusion matrices and ROC curves
   - Subgroup precision & recall by race and sex for bias detection
4. **Fairness analysis**: Visualize disparities and, if desired, apply fairness-aware thresholding or re-weighting.

### Tips

- All dependencies are installed inside the container—no local Python setup needed.
- To re-run the entire data pipeline or modeling scripts from a shell inside the container:
  ```bash
  docker compose exec notebook python src/data.py
  docker compose exec notebook python src/modeling.py
  ```
- You can customize the notebook (`notebooks/01-explore.ipynb`) to experiment with additional models or fairness criteria.

