# Defense Study Guide — Customer Churn Prediction
**Joel Suhner | Fundamentals of Python & Applications in Data Science**

---

## 1. Problem & Data

### What the task is

You are solving a **binary classification** problem: given a customer's behavioural and contractual
data, predict whether they will churn (`churned = 1`) or stay (`churned = 0`). The target variable
is `churned`. Everything else in the dataset is a feature used to make that prediction.

### What each feature represents

| Feature | What it measures | Direction of churn risk |
|---|---|---|
| `tenure_months` | How long the customer has been subscribed (1–60 months) | Longer → lower risk (loyalty builds up) |
| `monthly_fee_chf` | What they pay per month (CHF 8–69) | No strong direct effect; feeds into value perception |
| `usage_hours_last_30d` | How much they actively use the service (0–70 h) | Lower usage → higher risk |
| `support_tickets_last_6m` | Number of complaints/issues raised (0–12) | More tickets → higher frustration → higher risk |
| `payment_delay_days_avg` | Average days late on payments (0–35 days) | More delays → financial friction → higher risk |
| `contract_type` | Monthly vs. annual subscription | Monthly → much higher risk (no commitment lock-in) |
| `plan_type` | Basic, standard, or premium tier | Basic → higher risk (lower perceived value) |
| `auto_renew_enabled` | Whether automatic renewal is switched on (0/1) | Off → higher risk (they have to actively choose to stay) |
| `email_engagement_score` | How much they interact with emails (0–10) | Lower → higher risk (already mentally disengaged) |
| `days_since_last_login` | Days since they last used the service | More days → higher risk |

### Class balance and why it matters

- **Churned:** 256 customers (21.3%)
- **Retained:** 944 customers (78.7%)
- **Ratio:** 3.7:1 (retained to churned)

This matters because a model that **always predicts "retained"** would score **78.7% accuracy** while
catching exactly **zero churners**. That model is useless for the business. Class imbalance means
accuracy is a deceptive metric — you need recall, precision, F1, and ROC-AUC to tell the real story.

> **Defensible alternative:** If the imbalance were more severe (e.g., 10:1 or 20:1), you might also
> use SMOTE to synthetically oversample the minority class. At 3.7:1 the `class_weight='balanced'`
> adjustment is standard practice and avoids the risk of overfitting to synthetic points.

---

## 2. Feature Engineering

### Every transformation you performed

**Encoding of categorical variables:**

1. **`is_monthly_contract`** — binary flag: 1 if `contract_type == "monthly"`, else 0.
   *Reason:* Logistic regression and distance-based models cannot use strings. Binary encoding is
   correct here because there are only two categories with no natural order.

2. **`plan_tier`** — ordinal mapping: `basic=0`, `standard=1`, `premium=2`.
   *Reason:* Plan type has a natural hierarchy (premium > standard > basic), so ordinal encoding
   preserves that meaningful order. Using 0/1/2 tells the model these values rank, which one-hot
   encoding would not.

**Composite interaction features:**

3. **`disengagement_risk`** = `(1 - usage/70) × (1 - email_score/10)`
   *Reason:* A customer who is both low-usage AND ignores emails is signalling disengagement on two
   independent channels. Multiplying them captures the combined effect — a customer with low usage
   but high email engagement is less at risk than one who scores low on both simultaneously.
   The `/70` and `/10` normalise to [0, 1] using the hard clipping bounds from data generation
   — this avoids data leakage because the bounds are logical caps, not learned from training data.

4. **`friction_score`** = `support_tickets × payment_delay_days_avg`
   *Reason:* Someone who raises many tickets is frustrated; someone who delays payments has financial
   friction. Someone doing both simultaneously is experiencing compounded stress. The product encodes
   that these signals amplify each other: 4 tickets × 10 delay days = 40 (much riskier than 4 × 1 = 4).

5. **`value_per_chf`** = `usage_hours / (monthly_fee + 1e-9)`
   *Reason:* Customers who get little usage relative to what they pay are more likely to perceive
   the subscription as poor value and cancel. The `1e-9` prevents division by zero.

6. **`inactivity_ratio`** = `days_since_last_login / (usage_hours + 1)`
   *Reason:* Combines two inactivity signals — someone who hasn't logged in for 60 days AND uses
   the service very little is doubly inactive. Dividing by `(usage + 1)` amplifies the signal for
   dormant customers and suppresses it for active ones.

7. **`loyalty_commitment`** = `tenure_months × auto_renew_enabled`
   *Reason:* Auto-renew alone says "I haven't actively opted out." Tenure alone says "I've been here
   a while." Together they say "I've been here a long time AND I trust the service enough to auto-renew"
   — the strongest retention signal. For customers with `auto_renew=0`, this is 0 regardless of tenure.

### Why combinations can be more informative than isolated variables

Individual features give you partial information. The data-generating logic builds in explicit
interaction terms (e.g., monthly contract AND inactive for 30+ days gets an extra +0.75 in log-odds
of churn). A single variable cannot capture this — `is_monthly_contract=1` alone doesn't predict churn
as strongly as `is_monthly_contract=1 AND days_since_last_login=60`. The engineered features bake
these combinations directly into the feature matrix, so even a simple linear model can exploit them.

### Encoding choices

| Feature | Choice | Why this, not the alternative |
|---|---|---|
| `contract_type` | Binary (0/1) | Only 2 categories; one-hot would create a redundant second column |
| `plan_type` | Ordinal (0/1/2) | Natural ordering exists; one-hot would lose that ordering and add an extra column |

> **Alternative to defend:** One-hot encoding for `plan_type` is also defensible — it makes no
> assumption about equal spacing between tiers (basic→standard might not be the same "jump" as
> standard→premium). Ordinal encoding assumes equidistant steps. In practice, for tree-based models
> this distinction doesn't matter because trees can learn any step pattern. For logistic regression,
> ordinal encoding forces the assumption of linearity in tier level.

### Scaling choices

| Model | Scaling needed? | Why |
|---|---|---|
| Logistic Regression | **Yes** — `StandardScaler` used | LR uses gradient descent and regularisation (L2 via `C`). Features on different scales mean the penalty term treats large-scale features unfairly. Scaling puts all features on the same footing. |
| Random Forest | **No** | Trees split on thresholds, not distances or dot products. A feature scaled by 1000 creates the same splits as the same feature unscaled — the threshold just moves proportionally. |
| XGBoost | **No** | Same reason as Random Forest — gradient boosting uses tree splits. |

> **Key leakage check:** You fit `StandardScaler` on `X_train` only (`scaler.fit_transform(X_train)`)
> and then apply it to `X_test` (`scaler.transform(X_test)`). This is correct. Fitting on the full
> dataset before splitting would leak test set statistics (mean, std) into training — a data leakage
> error.

---

## 3. Preprocessing

### Train/test split strategy

```
X_train: 960 rows (80%)  |  X_test: 240 rows (20%)
Train churn rate: 21.4%  |  Test churn rate: 21.2%
random_state=42, stratify=y
```

- **80/20 split:** Common default. On 1200 rows this gives 240 test samples — enough to get reliable
  estimates on ~51 actual churners in the test set. A larger test set would give more reliable
  estimates but leave less data for training.
- **`stratify=y`:** Without stratification, a random split could by chance put 25% of churners in
  train and 17% in test, or vice versa. Stratification guarantees both splits mirror the original
  21.3% churn rate (21.4% train, 21.2% test — almost identical). This is essential when the minority
  class is small.
- **`random_state=42`:** Makes the exact same split reproducible. Anyone re-running the notebook
  gets identical numbers.

> **Alternative:** You could use 70/30 (more test data, more reliable estimates) or 85/15 (more
> training data, potentially better model). 80/20 is the standard starting point for datasets of
> this size.

### How class imbalance was handled

Three approaches, each model-specific:

1. **Logistic Regression:** `class_weight='balanced'` — sklearn automatically computes sample weights
   as `n_samples / (n_classes × class_count)`. Churner samples get weight 3.7× higher than retained,
   so errors on churners penalise the loss function more heavily.
2. **Random Forest:** Same `class_weight='balanced'` mechanism applied during tree splitting.
3. **XGBoost:** `scale_pos_weight = 755/205 = 3.68` — equivalent mechanism for gradient boosting
   (XGBoost doesn't use `class_weight` — using that parameter would silently be ignored, which you
   caught and documented in your AI usage section).

> **Alternative not used — SMOTE:** Synthetic Minority Oversampling creates new artificial churner
> rows. Tradeoff: can improve recall further, but risks overfitting to synthetic points and is harder
> to explain. `class_weight` achieves similar results without introducing fake data.

### Leakage risks and how they were avoided

| Potential leakage point | What you did | Why it matters |
|---|---|---|
| Scaler fit | `fit_transform` on train only, `transform` on test | If fitted on full data, test set mean/std leak into training |
| Feature normalisation bounds (`/70`, `/10`) | Used hardcoded clip bounds, not computed from data | Computing max from data and dividing would encode test set information |
| CV scoring | `cross_validate` on `X_train` only, not the full dataset | If test set rows were included in CV folds, the test ROC-AUC would be inflated |

---

## 4. Models

### Model 1: Logistic Regression

Logistic Regression works by finding a weighted sum of all features (a linear combination) that best
separates churners from retained customers. Imagine drawing a flat decision boundary — a line in 2D,
a hyperplane in 15D — that puts most churners on one side and most retained customers on the other.
The `sigmoid` function then squashes this weighted sum into a probability between 0 and 1. The model
learns which features to weight heavily by minimising a loss function (log-loss) on the training data.
It is the simplest possible classifier that outputs calibrated probabilities, and it serves as the
interpretable baseline that every more complex model should aim to beat.

**Key hyperparameters:**
- `C=0.1` — Inverse of regularisation strength. Smaller C = stronger L2 penalty = coefficients are
  pulled toward zero = less overfitting. C=0.1 is conservative, suitable for a small dataset where
  overfitting is a real risk.
- `class_weight='balanced'` — Described above.
- `solver='lbfgs'` — Optimisation algorithm. The default for sklearn LR; works well for small-to-medium
  datasets. Alternatives include `liblinear` (faster for very small data) or `saga` (needed for L1).
- `max_iter=1000` — Number of steps the solver takes. Increased from default 100 to ensure convergence.

### Model 2: Random Forest

A Random Forest builds hundreds of decision trees (here: 300), where each tree is trained on a random
bootstrap sample of the training rows, and at each split only a random subset of features is considered.
Individual trees overfit badly, but when you average their predictions, the errors cancel out — a
technique called bagging. The "random" part (both in data sampling and feature selection at each split)
ensures the trees are different enough from each other that their average is more accurate than any
single tree. The model naturally handles non-linear relationships and feature interactions because each
tree can make complex sequences of if-else splits.

**Key hyperparameters:**
- `n_estimators=300` — 300 trees. More trees → lower variance, diminishing returns after ~200.
- `max_depth=12` — Maximum depth of each tree. Without this limit, trees would grow until every
  training row is perfectly classified (overfitting). Limiting depth forces generalisation.
- `min_samples_leaf=4` — Each leaf (terminal node) must contain at least 4 training samples. Prevents
  tiny, overfitted leaves.
- `class_weight='balanced'` — Same as LR.

### Model 3: XGBoost

XGBoost is a gradient-boosted ensemble. Unlike Random Forest where trees are built in parallel
independently, gradient boosting builds trees sequentially, where each new tree tries to correct the
errors of all previous trees. Think of it as an iterative process: the first tree makes rough
predictions, the second tree learns the residuals (errors) of the first, the third learns the residuals
of the first two, and so on. The "gradient" part means it uses calculus (gradients of the loss function)
to decide which direction each new tree should push predictions. This sequential correction usually
makes XGBoost more accurate than Random Forest on tabular data, at the cost of more tuning.

**Key hyperparameters:**
- `n_estimators=300` — Number of sequential trees (boosting rounds).
- `learning_rate=0.05` — How much each tree's contribution is scaled. Smaller = more conservative,
  less overfitting, but needs more trees.
- `max_depth=5` — Shallower trees than RF (5 vs 12) because XGBoost trees are "weak learners"
  that are intentionally kept simple.
- `subsample=0.8` — Each tree trains on 80% of rows (like RF bootstrapping).
- `colsample_bytree=0.8` — Each tree uses 80% of features at each split.
- `scale_pos_weight=3.68` — Imbalance correction (equivalent to `class_weight='balanced'`).

### Why these three models are appropriate

- **LR** is the correct interpretable baseline. Because the data was generated from a logistic
  function, LR is actually the theoretically optimal model class here. It also forces you to think
  about linearity assumptions.
- **RF** tests whether non-linear interactions improve on LR. It is robust, doesn't need scaling,
  and supports SHAP TreeExplainer exactly.
- **XGBoost** is the state-of-the-art for tabular classification and serves as the performance
  ceiling. That LR beat it on ROC-AUC is itself an interesting finding (see Section 5).

---

## 5. Evaluation

### Why accuracy alone is insufficient

With 78.7% retained customers, a model that predicts "retained" for every customer gets 78.7% accuracy
and catches exactly 0 churners (recall = 0). This model is completely useless for business but looks
"good" by accuracy. You need metrics that specifically measure performance on the minority class.

### Metrics used and what they measure

| Metric | Formula (intuition) | Why it matters here |
|---|---|---|
| **Accuracy** | Correct / Total | Reported but not the primary metric — misleading under imbalance |
| **Recall** | TP / (TP + FN) | What fraction of actual churners did we catch? Missing a churner costs CHF 408/year |
| **Precision** | TP / (TP + FP) | Of customers we flagged as churners, how many actually are? False alarms waste retention budget |
| **F1** | Harmonic mean of precision & recall | Single number balancing both; punishes extreme imbalances |
| **ROC-AUC** | Area under ROC curve | Threshold-independent; 0.5 = random, 1.0 = perfect. Good for ranking customers by risk |
| **PR-AUC** | Area under precision-recall curve | More sensitive than ROC-AUC under class imbalance; harder to game |
| **Brier Score** | Mean squared error of probabilities | Measures calibration quality; lower is better |

### Actual results — test set

```
                     ROC-AUC  PR-AUC      F1  Recall  Precision  Accuracy   Brier
Logistic Regression   0.8613  0.6643  0.5986  0.8627     0.4583    0.7542  0.1720
Random Forest         0.8259  0.5106  0.5400  0.5294     0.5510    0.8083  0.1414
XGBoost               0.8247  0.4978  0.5660  0.5882     0.5455    0.8083  0.1416
```

**Cross-validation (5-fold, on train set) — more reliable estimate of generalisation:**
```
                     ROC-AUC           PR-AUC               F1
Logistic Regression  0.7930 ± 0.0204  0.5792 ± 0.0425  0.5324 ± 0.0278
Random Forest        0.7636 ± 0.0208  0.5362 ± 0.0315  0.4843 ± 0.0435
XGBoost              0.7368 ± 0.0266  0.4906 ± 0.0161  0.4743 ± 0.0338
```

**Key observation:** The single test set results are notably higher than CV for all models (LR:
0.8613 vs 0.7930). CV over 5 folds is more trustworthy than a single split. The ranking is the same
— LR is best on both — but the actual numbers should be quoted from CV for a rigorous claim.

**Why LR beat RF and XGBoost (important for defence):** The data was literally generated by a
logistic function. LR is the theoretically correct model for logistic data — it has enough capacity
to learn the true relationship without introducing unnecessary variance from overfitting to noise. RF
and XGBoost have more flexibility (complexity) than needed, and on a small dataset (1200 rows, 15
features) that extra complexity doesn't help.

### Confusion matrix in business terms

**Approximate confusion matrices (test set, 51 actual churners, 189 retained):**

```
Logistic Regression:    TP ≈ 44   FN ≈ 7    (catches 86% of churners but 52 false alarms)
                        FP ≈ 52   TN ≈ 137

Random Forest:          TP ≈ 27   FN ≈ 24   (more balanced but misses half the churners)
                        FP ≈ 22   TN ≈ 167

XGBoost:                TP ≈ 30   FN ≈ 21   (slightly better recall than RF)
                        FP ≈ 25   TN ≈ 164
```

**Business cost of each error type:**

| Error | What it means | Business cost |
|---|---|---|
| False Negative (FN) | Churner predicted as retained → no intervention | Lost annual revenue ≈ CHF 33.67 × 12 = **~CHF 408/customer** |
| False Positive (FP) | Retained customer predicted as churner → unnecessary offer | Cost of retention offer ≈ CHF 33.67 (one month free) = **~CHF 34/customer** |

The cost ratio is **~12:1** (missing a churner costs 12× more than a wasted offer). This is why
you prefer a model with high recall even at the cost of lower precision — better to send 52 wasted
offers (LR) than miss 24 churners (RF).

### Business cost threshold analysis

Your cost function analysis found the optimal threshold (the probability above which you classify
as "churn") for RF and XGBoost that minimises total cost (FN × 408 + FP × 34). At 0.5 threshold
the models are too conservative; lowering the threshold catches more churners at the cost of more
false alarms, but the 12:1 cost ratio means the trade-off is worth it.

---

## 6. SHAP Interpretation

### What SHAP values are

SHAP (SHapley Additive exPlanations) values answer the question: **"How much did each feature
contribute to pushing this specific prediction away from the average prediction?"**

The key idea: the model's base value (expected output across all training data) is 0.50 (the RF
predicts 50% churn probability on average). For any individual customer, each feature gets a SHAP
value that represents how much it increased or decreased the prediction from that baseline. The SHAP
values for all features sum to exactly `prediction − base_value`. This is mathematically guaranteed.

**SHAP vs. simple feature importance (e.g., Gini importance):**

| | SHAP | Gini/Permutation Importance |
|---|---|---|
| Per-prediction | Yes — each customer gets their own SHAP values | No — single global number |
| Direction | Yes — positive = pushes toward churn, negative = toward retention | No — only magnitude |
| Correlated features | Handles by distributing credit fairly (Shapley value theory) | Permutation: can be unstable; Gini: biased toward high-cardinality features |
| For trees | TreeExplainer = exact, fast | Both available but less information |

### Why TreeExplainer specifically

`shap.TreeExplainer` traverses the actual tree structure to compute exact Shapley values for every
customer — no approximation, no sampling. It runs in milliseconds on tree-based models. This is
the gold standard for RF and XGBoost interpretation.

### Which features drive churn predictions

Based on the beeswarm plot and mean |SHAP| bar chart:

**Top features driving churn (in order of global importance):**

1. **`inactivity_ratio`** (engineered) — the single strongest predictor. High inactivity ratio
   (long absence, low usage) strongly pushes toward churn. This makes intuitive sense: customers
   who barely use the service and haven't logged in recently have mentally already left.

2. **`loyalty_commitment`** (engineered) — tenure × auto_renew. High values (long-tenure customers
   with auto-renew on) strongly push toward retention. Customers with zero auto-renew have zero
   loyalty_commitment regardless of tenure — the interaction captures this correctly.

3. **`is_monthly_contract`** — monthly contract customers are at substantially higher risk. This
   mirrors the +0.55 logit weight built into the data and the additional +0.75 interaction for
   inactive monthly customers.

4. **`usage_hours_last_30d`** — high usage pushes toward retention (negative SHAP). Customers who
   actively use the service find value in it.

5. **`disengagement_risk` / `friction_score`** — the dual-channel disengagement signal and the
   frustration + payment friction signal both contribute meaningfully.

**Reading the beeswarm plot:**
- Each dot = one customer in the SHAP sample (200 customers)
- Colour: red = high feature value, blue = low feature value
- X-axis: SHAP value (right = pushes toward churn, left = toward retention)
- Example: red dots for `is_monthly_contract` on the right side = monthly contract customers are
  predicted as more likely to churn

### Connection to data-generating logic

The data was generated by a logit function with these known weights:

| Signal | Logit weight | Matches SHAP? |
|---|---|---|
| auto_renew_enabled | −0.65 (strongest negative) | Yes — captured by `loyalty_commitment` as top feature |
| contract monthly | +0.55 + +0.75 interaction | Yes — `is_monthly_contract` is top 3 |
| plan basic | +0.35 | Yes — `plan_tier` in mid-range importance |
| payment_delay | +0.085 | Yes — `friction_score` captures this |
| email_engagement | −0.12 | Yes — `disengagement_risk` captures this |
| usage_hours | −0.07 | Yes — directly and via `inactivity_ratio` |

The engineered features successfully extracted the interaction signals that the raw features alone
could not represent as linear terms.

---

## 7. Likely Defense Questions

### [likely] — Focus your prep here

**Q1 [likely]: Why did you stratify the train/test split?**
Without stratification, random splitting on a small minority class (21%) could accidentally put
too few or too many churners in one split. Stratification guarantees both sets have the same ~21%
churn rate, so evaluation metrics are comparable to the full dataset's distribution.

**Q2 [likely]: Why is accuracy not the right metric for this problem?**
Because 78.7% of customers are retained, a model that always predicts "retained" scores 78.7%
accuracy while catching zero churners. Accuracy doesn't distinguish between types of errors, so it
is misleading when classes are imbalanced. Recall and ROC-AUC measure performance specifically on
the minority (churn) class.

**Q3 [likely]: What is ROC-AUC and what does 0.86 mean for your best model?**
ROC-AUC is the area under the Receiver Operating Characteristic curve, which plots True Positive
Rate against False Positive Rate at every possible classification threshold. AUC = 0.5 is random
guessing; AUC = 1.0 is perfect. AUC = 0.8613 means the LR model correctly ranks a randomly chosen
churner above a randomly chosen retained customer 86.1% of the time. It measures ranking quality
independently of which threshold you choose.

**Q4 [likely]: Why does Random Forest not need feature scaling but Logistic Regression does?**
Logistic Regression uses L2 regularisation and gradient descent, which penalise and update features
based on their magnitude. If `tenure_months` ranges 1–60 and `email_score` ranges 0–10, the
regularisation penalty treats them unequally. Scaling puts both on the same scale (mean=0, std=1)
so every feature is penalised equally. Decision trees (RF, XGBoost) make binary splits on thresholds
— the scale of a feature doesn't affect where the split is, only the threshold value changes
proportionally. The tree's structure and accuracy are identical with or without scaling.

**Q5 [likely]: What is `class_weight='balanced'` doing?**
It reweights training samples inversely proportional to class frequency. In your case, churner
samples are weighted 3.7× higher than retained samples. This means the model's loss function is
penalised 3.7× more heavily for misclassifying a churner than for misclassifying a retained
customer. The effect is similar to oversampling the minority class without actually duplicating rows.

**Q6 [likely]: Why did Logistic Regression outperform Random Forest and XGBoost?**
The dataset was generated by a logistic function — a linear combination of features passed through
a sigmoid. Logistic Regression is the theoretically correct model for logistic data. RF and XGBoost
have far more capacity (they can model complex non-linear interactions) but on 1200 rows this extra
complexity introduces variance without reducing bias. A simpler model that matches the data's true
structure beats a complex model that overfits to noise.

**Q7 [likely]: What is a SHAP value and what does a positive SHAP value mean?**
A SHAP value tells you how much a specific feature contributed to pushing one individual customer's
predicted churn probability away from the average prediction (base value ≈ 0.50). A positive SHAP
value means that feature pushed the prediction toward higher churn probability. A negative SHAP
value means it pushed toward lower probability (i.e., toward retention). The sum of all SHAP values
plus the base value equals the model's final probability for that customer.

**Q8 [likely]: What is your engineered feature `friction_score` and why is it better than just using
the two raw features separately?**
`friction_score = support_tickets × payment_delay_days_avg`. A customer with 4 tickets but 0 delay
may just be actively engaged (not at risk), while a customer with 1 ticket but 30 days delay has
different risk. A customer with 4 tickets AND 30 days delay is experiencing compounded frustration
and financial friction. The multiplication captures the joint effect: the two features amplify each
other, which the model can't learn from two separate linear terms in logistic regression.

**Q9 [likely]: Why did you use ordinal encoding for `plan_type` instead of one-hot encoding?**
Plan type has a natural ranking: basic < standard < premium. Ordinal encoding (0, 1, 2) preserves
this order and tells the model "premium customers are more committed than basic customers, not just
different." One-hot encoding would create three binary columns where no order is implied. For tree
models the choice barely matters (they can learn any pattern), but for logistic regression ordinal
encoding captures the monotonic relationship more naturally with a single coefficient.

**Q10 [likely]: Your test ROC-AUC (0.8613) is higher than your CV ROC-AUC (0.7930). Which should
you trust?**
Cross-validation (0.7930 ± 0.0204). CV evaluates the model on 5 different held-out folds, giving
5 independent estimates of generalisation. The single test set gives just one estimate on one
specific partition. The test set happened to be a favourable partition. CV's average over 5 folds
is a less noisy, more reliable estimate of true generalisation performance. The ranking (LR > RF >
XGBoost) is consistent across both.

---

### [stretch] — Know the concept, not every detail

**Q11 [stretch]: What is PR-AUC and when is it preferred over ROC-AUC?**
PR-AUC is the area under the Precision-Recall curve, which plots Precision vs. Recall at every
threshold. The no-skill baseline for PR-AUC equals the positive class rate (~0.21), while ROC-AUC's
baseline is always 0.5 regardless of imbalance. This means PR-AUC is a stricter metric under
imbalance — a model that barely beats the baseline looks good on ROC-AUC but not on PR-AUC.
LR scores 0.6643 PR-AUC vs. the 0.21 baseline, which is a meaningful improvement.

**Q12 [stretch]: Why might SHAP and permutation importance disagree?**
Permutation importance measures how much model accuracy drops when you randomly shuffle a feature's
values. If two features are highly correlated (e.g., `disengagement_risk` and `usage_hours_last_30d`
both capture usage), permuting one leaves the other intact — the model barely suffers because it
shifts to the correlated feature. So both features get low permutation importance. SHAP distributes
credit fairly between correlated features using game-theoretic Shapley values, so it better reflects
each feature's true marginal contribution. In your case, SHAP would split credit between
`usage_hours` and `disengagement_risk` proportionally.

**Q13 [stretch]: What is the Brier score and why does Random Forest have a better Brier score than
Logistic Regression despite lower ROC-AUC?**
The Brier score is the mean squared error of predicted probabilities (lower = better). RF (0.1414)
beats LR (0.1720) here because LR with `class_weight='balanced'` shifts its effective threshold
aggressively toward predicting churn — this boosts recall (0.8627) but means its probabilities are
less well-calibrated. RF is more conservative in its probability estimates, making them closer to
actual frequencies. This is visible in the calibration reliability diagram.

**Q14 [stretch]: What is calibration and why does it matter for business decisions?**
A calibrated model is one where "predicted 70% churn probability" means 70% of those customers
actually churned. RF tends to be overconfident (probabilities clustered near 0.5), and LR with
class weighting is overconfident in the high-probability range. For business decisions — especially
computing revenue at risk (P(churn) × annual_fee) — you need well-calibrated probabilities, not
just a good ranking. Miscalibrated probabilities give wrong revenue-at-risk estimates.

**Q15 [stretch]: What are the main limitations of your model that would apply in a real deployment?**
(1) Synthetic data: generated from a clean logit function. Real churn data is noisier and the
feature-target relationship is messier — real performance would likely be lower. (2) No time
dimension: churn is a temporal process (customers churn after sustained frustration), but this
dataset is a single cross-sectional snapshot. A proper model would use time-series features or
survival analysis. (3) Possible feature leakage: `support_tickets_last_6m` and `payment_delay`
could partially reflect post-decision behaviour (customers who have decided to leave may stop paying
on time), which would inflate model performance. (4) No hyperparameter tuning: RandomizedSearchCV
would likely improve performance further.

---

## 8. Defense-Specific Prep

### 2-minute spoken walkthrough script

> "I built a binary classifier to predict which subscription customers will churn before they cancel.
> The dataset has 1200 customers and a 21% churn rate, which creates a class imbalance problem —
> so accuracy alone is misleading.
>
> I started with 10 raw features — things like tenure, usage hours, payment delays, and contract type.
> I then engineered 7 additional features that capture interaction effects. For example, `friction_score`
> multiplies support tickets by payment delay days, because a customer who complains a lot AND delays
> payments is at compounded risk — more than either signal alone. I also created `inactivity_ratio`,
> which turned out to be the single most important predictor.
>
> I trained three models: Logistic Regression as an interpretable baseline, Random Forest as a
> non-linear ensemble, and XGBoost as the performance ceiling. Surprisingly, Logistic Regression
> achieved the best ROC-AUC of 0.86 — which makes sense because the data itself was generated by a
> logistic function. All three models used class weighting to correct for the 3.7:1 class imbalance.
>
> I evaluated using ROC-AUC, PR-AUC, and F1 rather than accuracy, and validated with 5-fold
> cross-validation. I then applied SHAP TreeExplainer to the Random Forest to understand which
> features drive predictions. The top driver is inactivity_ratio, followed by loyalty_commitment
> and monthly contract status — which all connect directly to the known churn signals in the data.
>
> Finally, I quantified the business impact: the optimal classification threshold that minimises
> cost comes from treating a missed churner as 12 times more expensive than a false alarm — which
> favours a model with high recall even at lower precision."

### The 3 numbers you must know cold

| Number | Value | Context |
|---|---|---|
| **Best model ROC-AUC** | **0.8613** (LR, test) — but quote **0.7930 ± 0.02** (CV) for rigour | "Our best model achieves 0.86 ROC-AUC, or 0.79 in cross-validation" |
| **Class balance** | **21.3% churned / 78.7% retained (3.7:1 ratio)** | Every imbalance question starts here |
| **Top SHAP feature** | **`inactivity_ratio`** — days since last login divided by usage hours | "The single strongest churn predictor is inactivity: customers who haven't logged in and barely use the service" |

### The 3 decisions most likely to be challenged in Q&A (ranked)

**#1 — Metric choice (highest probability):**
"Why ROC-AUC and not accuracy?" Be ready with the "always predict retained = 78.7% accuracy but
zero recall" argument. Also be ready to explain why you report PR-AUC alongside ROC-AUC (it's
more sensitive under imbalance). If pushed: "For this business, I actually care most about recall
because missing a churner costs 12× more than a false alarm."

**#2 — LR beating RF and XGBoost (very likely given the surprising result):**
"Why does the simplest model win?" Have the answer ready: the data was generated by a logistic
function, so LR is the correct model class. More complex models introduce variance without adding
useful bias reduction on this small synthetic dataset. Also note: CV results confirm LR is
consistently best (0.7930 vs 0.7636 vs 0.7368), not just lucky on one test split.

**#3 — Feature engineering choices (likely):**
"Why multiply features rather than just include them both?" Use friction_score as your clearest
example: 4 tickets × 1 delay day = 4 (mildly concerning), 4 tickets × 25 delay days = 100 (high
risk). The product captures that these signals amplify each other. A linear model can only add them,
missing the compounding effect. Be ready to say "I could have used a polynomial feature transformer
instead, but domain-motivated features are more interpretable and defensible."
