# Vessel Shaft-Power Assessment

This project uses vessel sensor data to study shaft-power measurement reliability and estimate shaft power from other onboard measurements.

The work includes data preprocessing, sensor reliability checks, machine learning model comparison, training, evaluation and investigation of unusual sensor readings.

## Project Structure

The main code is in `src/vessel_power/`.

| File | Description |
|---|---|
| `preprocessing.py` | Reads sensor CSV files and prepares minute-level data |
| `reliability.py` | Checks sensor readings, selects usable labels and defines prediction conditions |
| `model_comparison.py` | Compares seven machine learning models |
| `modeling.py` | Contains the selected model, updating methods and evaluation metrics |
| `pipeline.py` | Runs preprocessing, training and evaluation |
| `change_analysis.py` | Investigates changes in sensor measurements |
| `plotting.py` | Creates graphs from the results |

Other files and folders:

- `run.py` - Main script for running the project
- `verify.py` - Checks the saved results and runs tests
- `config/` - Sensor information and model settings
- `results/` - Model results, figures and supporting files
- `tests/` - Automated tests
- `requirements.txt` - Required Python libraries

## Data Preprocessing

The dataset contains 13 monthly sensor files covering January 2024 to January 2025.

The preprocessing code:

1. Reads the original sensor measurements.
2. Checks for duplicate records and conflicting values.
3. Groups readings into completed one-minute intervals.
4. Calculates the mean, count, minimum and maximum where needed.
5. Identifies missing or insufficient sensor readings.
6. Prepares the data for reliability checks and model training.

Missing sensor values are not filled with estimated values.

The reliability checks are used to identify which recorded shaft-power measurements can be retained for training and evaluation. Measurements with unresolved issues are excluded from the labelled evaluation.

## Model Inputs

Seven sensor measurements are used in the model workflow:

- Main engine RPM
- Turbocharger RPM
- Fuel load
- Scavenging-air pressure
- Speed through water (STW)
- Speed over ground (SOG)
- Turning rate

The selected model estimates shaft power using absolute engine RPM:

    Estimated Power = a × |Engine RPM|³

The fitted coefficient is:

    a = 0.007814723394

The other sensor inputs help determine the operating condition and whether a prediction can be made.

A prediction is not issued when required input readings are insufficient, outside the accepted limits or do not satisfy the supported operating conditions.

## Model Training and Comparison

Seven models were compared:

1. Training-mean baseline
2. Speed-cubic model
3. RPM-cubic model
4. Ridge regression (alpha = 1)
5. Ridge regression (alpha = 100)
6. Histogram Gradient Boosting (15 leaves)
7. Histogram Gradient Boosting (31 leaves)

The RPM-cubic model was selected from the historical model comparison.

The final model was trained using 180,673 eligible minute-level observations.

Three prediction methods were evaluated:

- **Static RPM cubic:** Uses the fitted coefficient without updating it.
- **Historical offset:** Adjusts the static prediction using recent recorded prediction errors.
- **Rolling RPM cubic:** Updates the cubic coefficient using recent eligible observations.

The updating methods use a seven-day history with at least 100 eligible labels covering three days. A 72-hour delay in label availability is assumed.
The historical model comparison originally used 180,675 training labels. After the final reliability screening, two labels were excluded, leaving 180,673 for final model fitting. The RPM-cubic model remained the selected method.

When sufficient historical labels are unavailable, the methods use their static-model fallback.

## Data Splitting

The data was divided chronologically:

| Period | Purpose |
|---|---|
| January-May 2024 | Training |
| June-July 2024 | Model selection |
| August 2024 | Calibration diagnostics |
| September-December 2024 | Evaluation |

January 2025 was included in the data inspection but not in the labelled shaft-power evaluation.

## Model Results

The final evaluation includes 101,998 eligible active-operation minutes.

| Method | MAE (kW) | RMSE (kW) | R² |
|---|---:|---:|---:|
| Static RPM Cubic | 736.23 | 764.49 | 0.439 |
| Historical Offset | 310.98 | 420.76 | 0.830 |
| Rolling RPM Cubic | 270.93 | 368.75 | 0.869 |

The rolling RPM-cubic method achieved the lowest overall error. However, its performance varied across months, and it was not the best method in every month.

These results were calculated against selected recorded-power measurements. Independently calibrated physical shaft-power values were not available.

Detailed results are saved in `results/metrics.csv`.

## Running the Project

Place the authorized monthly CSV files inside `data/raw/`:

- `1.csv` through `12.csv`
- `Jan25.csv`

Run the main preprocessing, training and evaluation pipeline:

```bash
python run.py --output results/reproduction --processed data/processed/reproduction.pkl
```

The script reads the raw files, prepares the measurements, fits the model, calculates predictions and saves the results.

Use a new output folder for each run to avoid overwriting existing results.

To verify the original saved analysis:

```bash
python verify.py --results results --processed data/processed/measurements.pkl
```

Full verification requires the original input files, saved prediction artifacts, Noon Reports and reference PDFs used in the project.

Additional analysis commands:

```bash
python run.py compare
python run.py changes
```

These commands reproduce the model comparison and sensor-change investigation. They require the original processed dataset and saved final artifacts.

## Results and Outputs

Important output files include:

- `metrics.csv` - Model evaluation metrics
- `coverage.csv` - Prediction coverage and abstention information
- `label_counts.csv` - Training and evaluation label counts
- `model.json` - Selected model coefficient and training details
- `reliability_summary.csv` - Summary of sensor reliability checks
- `figures/monthly_errors.png` - Monthly prediction error comparison
- `figures/daily_power.png` - Daily power comparison graph

## Limitations

The available data does not provide independent physical shaft-power calibration.

The rolling updates also assume that reliable recorded-power labels become available after 72 hours. This assumption has not been verified for live operation.

The model results should therefore be treated as a retrospective evaluation against screened recorded measurements, rather than confirmed physical-power accuracy.

## Data Availability

The original vessel sensor files and some processed datasets are not included in the public repository.
Access to the authorized data is required to reproduce the complete analysis.
