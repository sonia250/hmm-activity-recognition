# Human Activity Recognition with a Hidden Markov Model

Formative 2 – Hidden Markov Models | African Leadership University | October 2026
Author: **[Your full name]** (individual work)

This project recognises four human activities (**standing, walking, jumping, still**) from smartphone accelerometer and gyroscope data. It uses a Hidden Markov Model (HMM) written **from scratch with numpy**, trained with **Baum-Welch** and decoded with **Viterbi**.

**Use case:** low-cost tracking of children's physical activity with an ordinary smartphone (a proof of concept recorded by one adult).

## Repository contents

```
.
├── README.md                    this file
├── notebook.ipynb               full code: features, HMM, training, evaluation, plots
├── prepare_data.py              turns raw Sensor Logger zips into the clean CSV files
├── data/
│   ├── train/                   53 labelled CSV files (8 s each)
│   └── test/                    8 labelled CSV files (unseen data, 8 s each)
└── Formative2_HMM_Report.docx   short report (4 pages)
```

## Data

- **Device and app:** iPhone XR, Sensor Logger (accelerometer = linear acceleration, gyroscope).
- **Sampling rate:** set to 50 Hz (measured 50.4 Hz), resampled to an exact 50 Hz grid.
- **Activities:** standing (phone in hand at waist level), walking, jumping, still (phone flat on a table).
- **File naming:** `<activity>_<split>_<recording>_<chunk>.csv`, for example `walking_train_1_03.csv`.
- **Columns:** `time_s, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z` (time in seconds, starting at 0).
- **Preprocessing:** 1 s trimmed from the start and end of each recording, then cut into 8 s files.

| Activity | Training files | Training seconds | Test files |
|---|---|---|---|
| Standing | 13 | 103.7 | 2 |
| Walking | 13 | 103.1 | 2 |
| Jumping | 14 | 108.4 | 2 |
| Still | 13 | 103.4 | 2 |
| **Total** | **53** | | **8** |

### Videos of data collection (one 5-second video per activity)

- Standing: [paste link]
- Walking: [paste link]
- Jumping: [paste link]
- Still: [paste link]

## Method

1. **Windows:** 2 s (100 samples), 50% overlap.
2. **Features (14 per window):** for accelerometer and gyroscope magnitude: mean, standard deviation, RMS, signal magnitude area, axis correlation (time domain), dominant frequency and spectral energy (FFT). Size features are log-scaled, then z-scored with the training mean and standard deviation.
3. **HMM:** 4 hidden states (activities), Gaussian emissions with diagonal covariance, transition matrix A and initial probabilities learned from data.
4. **Training:** Baum-Welch with a convergence check (relative change of the log-likelihood below 1e-4); it converged after 6 iterations.
5. **Decoding:** Viterbi.

## Results (unseen test data, 56 windows)

| Activity | Samples | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| Standing | 14 | 1.000 | 1.000 | 1.000 |
| Walking | 14 | 1.000 | 1.000 | 1.000 |
| Jumping | 14 | 1.000 | 1.000 | 1.000 |
| Still | 14 | 1.000 | 1.000 | 1.000 |

Training accuracy was 96.7% (all errors were still windows labelled as standing). The test set is small and comes from the same session, phone and person as the training data, so the result does not show that the model works for other people or phones. See the report for the full discussion.

## How to run

**In Google Colab (recommended)**

1. Open `notebook.ipynb` in Colab.
2. In the first cell, download the repository so the `data/` folder is available:

```python
!git clone https://github.com/<your-username>/<repo-name>.git
%cd <repo-name>
```

3. Run the remaining cells in order (Runtime > Run all).

**On your computer**

```bash
pip install numpy pandas matplotlib jupyter
jupyter notebook notebook.ipynb
```

Keep `notebook.ipynb` in the same folder as `data/`.

## Requirements

Python 3, numpy, pandas, matplotlib. No HMM library is used: the HMM, Baum-Welch and Viterbi are implemented in the notebook.
