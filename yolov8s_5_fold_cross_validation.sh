#!/path/to/your/python
from ultralytics import YOLO
from pathlib import Path

weights_path = "yolov8s.pt"

# List of dataset YAML files for each fold
ds_yamls = [
    Path('split_1/split_1_dataset.yaml'),
    Path('split_2/split_2_dataset.yaml'),
    Path('split_3/split_3_dataset.yaml'),
    Path('split_4/split_4_dataset.yaml'),
    Path('split_5/split_5_dataset.yaml')
]

results = {} # Dictionary to store training results for each fold

# Define your additional training arguments
batch = 8            # Batch size: controls GPU VRAM usage and training speed
project = "name_of_your_project" # Project name for organizing results
epochs = 40         # Number of training epochs per fold
val = True           # Enable validation during training
num_workers = 10     # Number of CPU workers for data loading
optimizer = 'SGD'
weight_decay = 0.001

# Loop through each dataset YAML for 5-fold cross-validation
for k, dataset_yaml in enumerate(ds_yamls):
    # Re-initialize the model with original weights for each fold
    # This ensures each fold starts from the same baseline and prevents data leakage
    model = YOLO(weights_path, task="detect")

    # Start training for the current fold
    results[k] = model.train(
        data=dataset_yaml,       # Dataset YAML for the current fold
        epochs=epochs,           # Number of epochs
        batch=batch,             # Batch size
        project=project,         # Project name
        name=f"fold_{k + 1}",    # Unique name for the current fold's run
        val=val,                 # Enable validation
        workers=num_workers,      # Number of CPU workers for data loading
        optimizer=optimizer,
        weight_decay=weight_decay
    )