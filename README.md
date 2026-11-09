# Titanic Survival Prediction

This repository uses logistic regression in Python and R to predict which passengers survived the Titanic disaster. The analysis is implemented twice: once in Python, and once in R. Each implementation runs in its own Docker container.

The scripts print data exploration, feature preparation, training accuracy, test predictions, and validation accuracy.

## Requirements

- Git
- Docker Desktop (installed + running)
- A Kaggle account to download the required dataset

Python, R, and their packages are installed inside the containers. You do not need to install them locally.

The commands below are for macOS or Linux using a Bash or Zsh terminal.

Windows alternatives are included in steps 2–4. On Windows, use Docker Desktop with Linux containers. The clone and build commands work in Bash, Zsh, PowerShell, and Command Prompt. Use the run command for your terminal. Internet access is needed for downloads and image builds.

## Repository structure

```text
titanic-disaster/
├── README.md
├── .gitignore
└── src/
    ├── data/
    │   ├── train.csv
    │   └── test.csv
    ├── python/
    │   ├── Dockerfile
    │   ├── main.py
    │   └── requirements.txt
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── main.R
```

## 1. Clone the repository

```bash
git clone https://github.com/seanc327/titanic-disaster.git
cd titanic-disaster
```

Run all commands below from this repository folder.

## 2. Download the data
### MacOS/Linux:
1. Go to [Titanic dataset on Kaggle](https://www.kaggle.com/competitions/titanic/data).
2. Sign in, join competition (if needed), then download and extract the dataset.
3. Create the data folder:

```bash
mkdir -p src/data
```

4. Copy `train.csv` and `test.csv` into `src/data`.

### If using terminal instead (MacOS/Linux):
```bash
mkdir -p src/data
unzip ~/Downloads/titanic.zip train.csv test.csv -d src/data
```

### For downloading with the Kaggle CLI:

The CLI option requires local Python and pip, plus authentication configured using the [official Kaggle CLI instructions](https://github.com/Kaggle/kaggle-cli). Accept the competition rules before downloading. Use only one download method.

```bash
python3 -m pip install kaggle
mkdir -p src/data
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip train.csv test.csv -d src/data
```

### For Windows PowerShell:

```powershell
py -m pip install kaggle
New-Item -ItemType Directory -Force -Path .\src\data | Out-Null
kaggle competitions download -c titanic -p src/data
Expand-Archive -Path .\src\data\titanic.zip -DestinationPath .\src\data -Force
```

### Expected folder structure:

```text
└── src/
    ├── data/
    │   ├── train.csv
    │   └── test.csv
```

The dataset is excluded from Git using `.gitignore` and is not included in this repository. 

Note: `gender_submission.csv` does not contain any actual test labels, and is not required.

## 3. Run the Python implementation

### MacOS / Linux
Build image:

```bash
docker build -t titanic-python ./src/python
```

Run container:

```bash
docker run --rm -v "$(pwd)/src/data:/app/data:ro" titanic-python
```

### Windows Powershell:

First build the image using the build command above, then run:

```powershell
docker run --rm -v "${PWD}/src/data:/app/data:ro" titanic-python
```

### Windows Command Prompt:

First build the image using the build command above, then run:

```cmd
docker run --rm -v "%cd%\src\data:/app/data:ro" titanic-python
```


## 4. Run the R implementation

### MacOS / Linux
Build image:

```bash
docker build -t titanic-r ./src/r
```

Run container:

```bash
docker run --rm -v "$(pwd)/src/data:/app/data:ro" titanic-r
```

### Windows Powershell:

First build the image using the build command above, then run:

```powershell
docker run --rm -v "${PWD}/src/data:/app/data:ro" titanic-r
```

### Windows Command Prompt:

First build the image using the build command above, then run:

```cmd
docker run --rm -v "%cd%\src\data:/app/data:ro" titanic-r
```

The first build may take several minutes while Docker downloads the image and installs packages.

For both containers:

- `-v` makes the local `src/data` folder available at `/app/data`.
- `:ro` makes the dataset read-only inside the container.
- `--rm` removes the container after the script finishes.
- Results are printed in the terminal. No prediction CSV is saved.

Note: if you change a script, rebuild its image before running it again.

## Data preparation and models

The following order is used in both implementations:

1. Inspect the data and report missing values and survival counts.
2. Add `FamilySize = SibSp + Parch + 1`.
3. Use `Pclass`, `Sex`, `Age`, `FamilySize`, and `Fare` as features.
4. Encode sex as male = 0 and female = 1 (binary).
5. Fill missing ages using the training median.
6. Fill the missing test fare using the training median fare.
7. Fit logistic regression model to predict `Survived`.
8. Predict survival for all 418 test passengers.

The Python model uses StandardScaler and scikit-learn's logistic regression. The R model uses an unpenalized logistic regression through `glm()`.

## Accuracy and expected output

Data dimensions before feature preparation (for both Python and R): 

- Training: 891 rows x 12 columns
- Test: 418 rows x 11 columns

Observed results:

| Result | Python | R |
| --- | --- | --- |
| Training accuracy | 80.02% | 80.02% |
| Validation passengers | 179 | 179 |
| Validation accuracy | 81.56% | 82.12% |
| Test predictions | 418 | 418 |

Both runs predicted 260 test passengers did not survive and 158 survived.

## Why test accuracy is unavailable

Kaggle's `test.csv` does not include `Survived` labels. Therefore, accuracy cannot be calculated directly for those 418 predictions.

Instead, each implementation trains a separate model using a stratified 80/20 split of the labeled `train.csv` data:

- 712 passengers for fitting
- 179 passengers for validation
- Random seed of 42

Missing-value replacements are learned from the fitting split only. Python also learns its scaling from that split only.

The reported validation accuracy is not accuracy on Kaggle's `test.csv`. R and Python generate different random splits even with the same seed, and their model implementations differ.

## Troubleshooting

### Docker not connecting?

Open Docker Desktop and wait for it to finish starting, then retry.

### CSV file not found?

Confirm that both CSV files are directly inside `src/data`, with these exact names:

```text
train.csv
test.csv
```

Run the Docker commands from the repository root so `$(pwd)/src/data` points to the correct folder.

### Input/output error or unexpectedly empty data on macOS/Linux?

During development, Docker intermittently failed to read the CSV files. Reading them locally before rerunning the container resolved the issue in those instances:

```bash
cat src/data/train.csv src/data/test.csv > /dev/null
```

Then retry the Docker run commands.

Note: This is a workaround, not a normal setup requirement. The underlying cause was not confirmed. If the issue persists, try keeping the repository and data in a fully local folder outside cloud-synced Desktop or Documents folders, and check Docker Desktop's file-sharing settings.