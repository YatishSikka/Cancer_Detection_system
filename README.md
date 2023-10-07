# Cancer_Detection_system
## Overview

The Cancer Detection System is a deep learning project designed to identify the presence of brain tumors in MRI scan images. This repository contains the necessary code and instructions to set up and use the system effectively.

## Setup

1. **Clone this repository** to your local machine:

   ```bash
   git clone https://github.com/your-username/cancer-detection-system.git

2. Navigate to the project directory:

   ```bash
   cd cancer-detection-system
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
4. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
   - On macOS and Linux:
    ```bash
    source venv/bin/activate
  
5. Install the project dependencies using the requirements.txt file:
   ```bash
   pip install -r requirements.txt
6. Download the brain MRI scans dataset from [Kaggle Brain MRI Images dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) .
7. Enter the root directory path (ROOT_DIR) as well as the test image path (PATH) in the .env file in your root directory.

## Usage

Once you have set up the environment, you can run the Cancer Detection System using the following command:
```bash
python main.py
```
The system will process the MRI scan images located at the path specified in the .env file and provide results on whether tumors are detected.
