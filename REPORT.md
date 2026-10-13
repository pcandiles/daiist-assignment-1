# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Paula María Candiles Puente
- **Student ID**: 18670
- **Email**: pcandiles.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## 1. Dataset

### 1.1 Where it comes from?

The dataset is the **Inside Airbnb listings** data for **Madrid, Comunidad de Madrid, Spain**, in the **17 September 2026 release**. It comes from Inside Airbnb, an independent, mission-driven project that publishes data about Airbnb's impact on residential communities. 

Its data is sourced from publicly available information on the Airbnb site, which Inside Airbnb analyzes, cleanses and aggregates, so it is not data provided by Airbnb itself. 

### 1.2 The CSV

I downloaded the detailed listings file (`listings.csv.gz`) from https://insideairbnb.com/get-the-data/. 

- Each row **represents one Airbnb listing** (a property or room offered for rent), not a booking and not a host, and a host can own many listings.
- The **target is the nightly price listed**, which is an asking price, not necessarily the price guests actually paid. 
- The **raw file has 23,109 rows and 90 columns**, although these numbers will change as I filter and clean the data. 

### 1.3 Why this dataset?

I picked it because I wanted to move away from the usual retail, supermarket and banking datasets and work on a problem connected to pricing, investment and finance. Pricing a new Airbnb listing is a **real decision with money at stake**: too high and the property stays empty, too low and revenue is lost.

## 2. Business / real-life framing

### 2.1 The hypothetical scenario

A new host in Madrid is about to publish their first Airbnb listing and has to **choose a nightly price**. They know nothing about the market, and the decision is risky: *if the price is too high the property stays empty, and if it is too low they lose revenue every night*. 

*What does the model do?* My model works as a **market benchmark**. It learns what other hosts in Madrid charge for similar listings and, from the characteristics of the new property (type, size, location, amenities),recommends a starting nightly price.

This scenario shaped how I built the pipeline:

- **Target definition.** The target is the **nightly price**, but the listed price is only what a host asks, not what guests actually pay. To get closer to what the market accepts, **I only kept listings with at least one review** **in the last 12 months** since a review shows that someone really booked it. The downside is that the model learns nothing about listings that never get booked. I also removed the top 1% of prices (above about €792 per night) so a few extreme listings would not dominate the model, and I trained on log(price) because prices are strongly right-skewed.
- **Split.** I used a **random 70/15/15 train/validation/test split**. The dataset is a single snapshot (scraped on 17 September 2026), so there is no timeline to split on, and a random split does not leak the future into training. 

### 2.2 Metrics driving the decision

This is a regression problem, so there is no classification threshold to set. The decision the host takes is the recommended price, so the metric that drives it is **MAE** (mean absolute error, in € per night). It reads directly as "the recommendation is typically off by X euros", which is what a host needs to judge how far to trust it. 

I also report **RMSE**, which penalizes large mistakes more, and **R²**, which shows how much of the price variation the model explains. All three are compared against a naive baseline that always predicts the median training price (about €155 per night, MAE of about €79 on the test set).

## 3. Data preparation & feature engineering

### 3.1 Cleaning and data-quality decisions

- **Personal data.** I dropped the host columns with personal information
  (name, "about" text, profile URLs and pictures, location), following the data
  minimization principle. They also add noise and no pricing information.
- **Listings without a price.** 4,324 of the 23,109 listings (18.7%) have no
  price, so they cannot be used as training targets. They look like inactive
  listings: 75.6% have zero availability for the year (vs. 12.1% of priced
  listings) and their median number of reviews is 4 (vs. 32).
- **Demand filter.** Among the priced listings I kept only those with at least
  one review in the last 12 months (14,766 of 18,785). The idea is that the
  model should learn prices that guests actually accept, not only asking
  prices. Listings with no recent reviews have a much more extreme price
  distribution, so this also removes a lot of noise. The trade-off is that the
  model says nothing about listings that do not get booked, so its
  recommendations are for the part of the market that has demand.
- **Missing and impossible values.** I removed duplicate or missing IDs, dropped
  columns that were more than 90% empty, and removed listings with coordinates
  outside Madrid or with zero capacity. Missing numeric values were filled with
  the median and missing categories were labelled "Unknown".
- **Outliers.** Prices go up to about €6,000 per night. I removed the top 1%
  (above €792 per night, 148 listings) because a few extreme prices would
  dominate the squared-error loss of a linear model.

### 3.2 Feature engineering

I started from a practical set of listing and host columns and built these
features:

| Feature | How it is built | Why |
|---|---|---|
| `bathrooms_num` | number extracted from the text ("1.5 baths"; "half-bath" = 0.5) | more bathrooms usually means a bigger, more expensive place |
| `amenity_count` | number of items in the amenities list | more amenities is a signal of a higher-end listing |
| `guests_per_bedroom` | `accommodates` / bedrooms (at least 1) | how crowded the listing is; separates shared from spacious |
| `distance_to_sol_km` | haversine distance to Puerta del Sol | location is the main price driver in Madrid, and distance to the centre is a smooth variable (neighbourhood dummies are not) |
| `*_total_months` | host/user tenure in years x 12 + months | one number for tenure |
| `review_score_mean` | mean of the review sub-scores | summary of review quality |
| 0/1 flags | superhost, verified identity, instant bookable, etc. (-1 = unknown) | the models need numbers |

### 3.3 Target and leakage

The target is trained as `log1p(price)`. The raw price is strongly right-skewed
(skewness 1.70), and the log version is almost symmetric (-0.30). Predictions
are converted back to euros per night to report the results. I also excluded
`estimated_revenue_l365d` and `estimated_occupancy_l365d` from the features,
because they are computed from the price itself and would leak the target.

### 3.4 Collinearity

A linear model cannot separate the effect of two features that carry the same
information. I measured it with the variance inflation factor (VIF) on the
numeric training features, and removed features one at a time, always the one
with the highest VIF, until all were at or below 10. Five were removed:
`hosts_time_as_user_years`, `hosts_time_as_host_years`, `review_scores_rating`,
`review_score_mean` and `availability_60`. The first four are exact
combinations of other columns (infinite VIF), and `availability_60` overlaps
with the 30- and 90-day availability (VIF 25.3). After this, the highest VIF was
4.55.

### 3.5 Encoding and scaling

The final set has 33 features: 4 categorical, 25 numeric and 4 binary. The 4
categorical features (neighbourhood, district, property type, room type) are
one-hot encoded. Categories with fewer than 30 training listings are grouped
into one "infrequent" column, which reduces 126 neighbourhoods to about 66
columns and avoids unreliable coefficients. Numeric features are standardized.
This matters most for the manual PyTorch loop, because gradient descent
converges slowly when features are on very different scales. Everything is in
one scikit-learn pipeline fitted on the training set only, and the validation
and test sets are only transformed, which prevents leakage. The result is 132
input columns, and the fitted pipeline is saved so the Gradio app applies the
same transformation.

## 4. Modeling: three implementations, one model

### 4.1 Model Choice

Since the target is a continuous nightly price, we use **linear regression (with L2 regularisation, i.e. Ridge)** rather than logistic regression, which is meant for categorical outcomes. 

The target is modelled as log1p(price) to reduce its right skew (1.70 → −0.30), and the top 1% of prices (> €792) were removed before splitting into 70/15/15 train/validation/test sets.

### 4.2 Results against the naive baseline
The baseline predicts the training median for every listing (log1p = 5.05, about €155/night)

All three implementations improve substantially on it: 
- Scikit-learn Ridge reaches an **R² of 0.607 (RMSE €86.1, MAE €53.2)**
- The manual PyTorch loop **0.527 (RMSE €90.8, MAE €57.1)**
- The nn.Linear/Adam model **0.516 (RMSE €89.8, MAE €57.4).**

All three models beat the baseline clearly.

### 4.3 Do they agree between eachother?
The three implementations fit the same linear model, so they largely agree, but they do not match exactly.

**Scikit-learn solves the least-squares problem in closed form**, whereas the two PyTorch versions use gradient descent for a fixed 500 epochs and do not fully converge (their validation loss remains above Ridge’s, and the best learning rate sits at the edge of the grid searched). 

The small gap, about 0.04 in log-RMSE, therefore reflects optimisation rather than any difference in model. The two PyTorch variants are statistically indistinguishable from one another.

### 4.3 What was tuned, and how?
Hyperparameters were tuned on the validation set only. The test set was kept for the final comparison, which is the correct protocol. **Only one hyperparameter per model was tuned.**

- **Ridge**: I tried 5 values and picked α = 1. It made almost no difference, because I have plenty of data. (α ∈ {0.01, 0.1, 1, 10, 100}).
- **PyTorch**: I tried 4 values, and both methods preferred the biggest, 0.1. (learning rate over {0.001, 0.01, 0.05, 0.1} for 300 epochs each)

## 5. Limitations & next steps

### 5.1 Limitations
I found the following limitations on the work:
1. **The model is too simple**. It treats every feature on its own, but in real life features work together. A central location matters more for a whole flat than for a room.
2. **Two of the models didn’t finish learning**. The PyTorch models were stopped too early, so they are a bit less accurate than scikit-learn.
3. **Amenities were just counted**. A pool and a hairdryer counted the same, even though a pool affects the price far more.
4. **The model doesn’t cover everything**. For example very expensive listings (over €792 per night) were removed and brand-new listings with no recent reviews were left out.
5. **Euro predictions are slightly too low**. We predicted on a log scale and converted back, which causes a small bias.
6. **Preparing data mistakes** Some calculations used all the data before it was split, so the test set leaked a little into training.

### 5.2 Next Steps
1. **Train the PyTorch models longer and with bigger step**s, until they match scikit-learn.
2. **Use amenities properly**, with a yes/no column for important ones like pool, air conditioning, lift and terrace.
3. **Cover the whole market: test on expensive listings too**, and build a second model for new listings without reviews.
4. **Fix the data preparation**. Do all calculations on the training data only, and find out why some values were missing.
5. **Test more than once**, using several data splits and data from different months.
6. **Correct the euro conversion**, or report percentage errors instead.

## 6. Generative AI use disclosure

Generative AI (Claude Code) was used to assist with coding, and refining content. All AI-assisted content was reviewed and edited for accuracy and consistency with my own understanding.

