# AutoMLClustering

This project implements an automated machine learning pipeline for clustering tasks, supporting various data preprocessing, error correction, clustering methods, and result analysis.

## Prerequisites

- Python 3.9 or above
- Linux-based system (tested on Ubuntu)
- Required packages (installed via `config.sh`)

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:tianchanghrbcn/AutoMLClustering.git
   cd /root/AutoMLClustering
   ```

2. Run the configuration script to set up the virtual environment and install dependencies:

   ```bash
   bash config.sh
   ```

3. Activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```

## Running the Project

### Step 1: Data Preprocessing
Navigate to the `src/pipeline/train` directory and run the preprocessing script:

```bash
cd src/pipeline/train
python pre-processing.py
```

### Step 2: Start the Training Pipeline
Run the training pipeline in the background using `nohup` to ensure it continues running even if the session is disconnected:

```bash
nohup python train_pipeline.py > output_training.log 2>&1 &
```

- Ensure that `PYTHONPATH` is correctly set up to avoid `ModuleNotFoundError`.
- The output and logs will be saved to the `output_training.log` file.

### Step 3: Start the Classifier
After completing the training pipeline, run the classifier in the background using the same `nohup` mechanism:

```bash
nohup python classifier.py > output_classifier.log 2>&1 &
```

- Ensure that the classifier script is in the correct directory and has all the required configurations.
- Logs and outputs for the classifier will be stored in the `output_classifier.log` file.

### Notes:
1. **Check Running Processes**:
   To ensure both scripts are running:
   ```bash
   ps aux | grep python
   ```

2. **Monitor Logs**:
   View the logs in real-time with:
   ```bash
   tail -f output_training.log
   tail -f output_classifier.log
   ```

3. **Stop Processes**:
   To stop the running processes, find their process IDs (PIDs) using `ps aux` and terminate them:
   ```bash
   kill <PID>
   ```

## Project Directory Structure
```plaintext
AutoMLClustering/
├── config.sh                # Configuration script to set up the environment
├── dataset/                 # Contains datasets for training and testing
│   ├── train/               # Training datasets
│   │   ├── beers            # Beers dataset
│   │   ├── flights          # Flights dataset
│   │   ├── hospital         # Hospital dataset
│   │   ├── rayyan           # Rayyan dataset
│   │   └── ...              # Other datasets (if applicable)
│   └── test/                # Placeholder for testing datasets
├── LICENSE                  # Project license (e.g., MIT License)
├── README.md                # Project documentation
├── reference/               # References and supporting documentation
├── requirements.txt         # Python dependencies
├── results/                 # Directory to store results (e.g., logs, outputs)
├── src/                     # Source code for the project
│   ├── cleaning/            # Data cleaning modules
│   │   ├── baran            # Baran cleaning algorithm
│   │   ├── mode             # Mode cleaning algorithm
│   │   └── ...              # Other cleaning algorithms (if applicable)
│   ├── clustering/          # Clustering methods
│   │   ├── AP               # Affinity Propagation clustering
│   │   ├── DBSCAN           # DBSCAN clustering
│   │   ├── GMM              # Gaussian Mixture Model clustering
│   │   ├── HC               # Hierarchical clustering
│   │   ├── KMEANS           # K-Means clustering
│   │   ├── OPTICS           # OPTICS clustering
│   │   └── ...              # Other clustering algorithms (if applicable)
│   ├── pipeline/            # Pipeline implementation
│   │   ├── train/           # Training pipeline
│   │   │   ├── pre-processing.py       # Preprocessing script
│   │   │   ├── train_pipeline.py       # Main training pipeline script
│   │   │   ├── classifier.py           # Classification logic
│   │   │   ├── classifier_preparation.py # Classifier data preparation
│   │   │   ├── cluster_methods.py      # Clustering method utilities
│   │   │   ├── clustered_analysis.py   # Clustering result analysis
│   │   │   └── error_correction.py     # Error correction module
│   └── --legacy/            # Legacy or deprecated code
├── venv/                    # Virtual environment directory (created by config.sh)
```

## Logs and Outputs

- Logs are saved in `output.log`.
- Intermediate results are stored in the `results` directory.

## Notes

- Ensure all paths in the scripts are correctly configured relative to the project root.
- If you encounter permission issues, ensure you have the required privileges or use `sudo` cautiously.

## Contributing

1. Fork the repository.
2. Create your feature branch: `git checkout -b feature/YourFeature`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/YourFeature`.
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

