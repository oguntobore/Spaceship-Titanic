# Spaceship Titanic: Predicting Who Was Transported

A machine learning solution to the Kaggle [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition. The task is to predict whether each passenger was transported to another dimension, using records recovered from the ship's computer. The metric is classification accuracy.

The focus of this project is **tested data cleaning and honest validation**, not chasing the last tenth of a point on the leaderboard.

## Results

| Model | 5-fold grouped CV accuracy |
|---|---|
| HistGradientBoosting (selected) | 0.810 ± 0.016 |
| Random Forest | 0.801 ± 0.016 |

HistGradientBoosting fold accuracies: 0.780, 0.827, 0.816, 0.821, 0.806.

Validation uses `StratifiedGroupKFold` (5 folds, seed 42), so members of one travel group never appear in both a training fold and its validation fold.

**Kaggle public leaderboard score:** `[add your score here]`

## Data

- `train.csv`: 8,693 labelled passengers. `test.csv`: 4,277 passengers to predict.
- Every feature column has roughly 2% missing values (181 to 217 missing per column), so how the gaps are filled matters.

## Approach

### 1. Validation
Passengers travelling together are often related and behave alike. Ordinary k-fold would let a passenger's group-mates sit in the training set while the passenger sits in validation, which inflates scores. The group ID comes from `PassengerId`, and folds are built around it.

### 2. Missing values: patterns were measured before being used
For each column I checked whether a rule beat a plain guess before using it.

| Column | Evidence from the data | Fill used |
|---|---|---|
| HomePlanet | Group-mates share a planet 100% of the time; decks A, B, C, T are 100% Europa and deck G is 100% Earth | Deck/destination agreement rule, then group value, then deck rules, then `Unknown` |
| Side | Group-mates share a side 100% of the time | Group value, then mode |
| Deck | Group-mates match 75.8% (random pair: 23.0%) | Group value, then mode |
| CabinNum | Group-mates match 75.9%; cabin number tracks group number (correlation 0.98 to 1.00 on every deck except T) | Group value, then interpolation by group number, then median |
| CryoSleep | Anyone who spent money cannot be in cryosleep | Spending rules, then `False` |
| Spending columns | Cryosleep passengers cannot spend | 0 |
| Age | Mean age differs by HomePlanet (average error 10.97 vs 11.42 years for the overall mean) | HomePlanet mean |
| VIP | Under 15% VIP in every deck and planet | `False` |
| Destination | No grouping beat the baseline (69.3% to 69.6% vs 69.5%) | Deck T rule, then group mode, then overall mode |

### 3. Ideas that did not help
- **Travel group to predict CryoSleep:** 53.6% accuracy where the rule applied, *worse* than always guessing "not in cryosleep" (64.2%). Companions do not copy each other's cryosleep choice.
- **Deck to predict CryoSleep:** rates range from about 20% (decks D, E, F) to about 55% (decks B, G), too weak to be a reliable rule.
- **HomePlanet, Deck or CryoSleep to predict Destination:** no gain over the mode, because TRAPPIST-1e dominates every group.
- **Deck + destination agreement for HomePlanet:** 76.9% accurate where the two agree, slightly *below* deck alone (78.8%). The exact group rule and the deck rules do the real work.
- **Dropping rows with missing values:** would discard about 20% of the training data and is impossible for the test set.

### 4. Features
Log-transformed spending, total spend, a "spent nothing" flag, age bins, group size, a solo-traveller flag, group spending, and a deck+side combination. After one-hot encoding the model uses 52 numeric columns.

### 5. Models
Random Forest and HistGradientBoosting, compared with grouped cross-validation. The higher mean accuracy is selected and retrained on all labelled passengers to produce the submission.

## What I learned
- Grouping the validation folds changes how far the scores can be trusted.
- Measuring a rule before using it exposed several fills that looked sensible but were no better than guessing.
- Fold-to-fold spread is about 1.6 points, so small differences between models should not be over-interpreted.

## Limitations and next steps
- Cross-validation used one seed, and the models use mostly default hyperparameters. Tuning and multiple seeds are the obvious next step.
- Fill values (modes, means, medians) are computed on the full training set before cross-validation, a small source of optimism.
- The deck/destination agreement step runs before the exact group fill and tested below deck alone, so it is a candidate to remove or reorder.

## Repository contents

```
spaceship-titanic-predicting-who-was-transported.ipynb   Analysis, cleaning, modelling
submission.csv                                           Predictions for the test set
requirements.txt                                         Python dependencies
```

## How to run

1. Download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/competitions/spaceship-titanic/data). The data is not included in this repository.
2. The notebook was written on Kaggle and reads from `/kaggle/input/competitions/spaceship-titanic/`. To run it locally, change the two `pd.read_csv` paths to `train.csv` and `test.csv`.
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Open the notebook and choose **Restart and Run All**.

## Acknowledgements

Competition and data by Addison Howard, Ashley Chow, and Ryan Holbrook, *Spaceship Titanic*, Kaggle, 2022.
