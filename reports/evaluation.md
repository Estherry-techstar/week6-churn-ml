# Evaluation Report — Customer Churn Prediction

**Dataset:** IBM Telco Customer Churn — 7,043 customers, 19 features, 26.5% churn
**Model:** Logistic regression (26 features after one-hot encoding)
**Held-out ROC AUC:** 0.8424 | **5-fold CV AUC:** 0.8451 ± 0.0133
**Shipped decision threshold:** 0.25

---

## 1. Which metric matters most, and why

**Recall on the churn class**, with ROC AUC used separately for comparing models.

### Accuracy is disqualified, and I can prove it

A `DummyClassifier` that predicts "no customer will ever churn" scores **73.46%
accuracy** on the test set. Its confusion matrix:

```
              predicted stay   predicted churn
actually stay          1035                 0
actually churn          374                 0
```

It identifies **zero** of the 374 customers who actually left. Recall 0.00. A
metric that awards 73% to a model with no predictive content is measuring class
balance, not skill. Since my real model scores 80.5% accuracy, the honest
comparison is seven points above a model that does nothing — not "80% correct."

### The costs of the two errors are not symmetric

- A **false positive** means sending a retention offer to a customer who was
  staying anyway. Cost: roughly one discount, ~$50.
- A **false negative** means a customer leaves with no intervention attempted.
  At ~$64/month average revenue, that is roughly **$770** of annual revenue
  lost, before replacement acquisition cost.

The ratio is on the order of **15:1**. When one error costs fifteen times the
other, the metric should track the expensive one. Recall measures exactly the
error that hurts: of everyone who left, how many did we see coming?

### Why not F1, and why AUC does a different job

**F1** is the harmonic mean of precision and recall, which weights the two
errors equally. That directly contradicts the 15:1 cost ratio. F1 is also nearly
flat across the entire useful threshold range in this model (0.602 at 0.50,
0.617 at 0.30), so it cannot discriminate between operating points that differ
enormously in business terms.

**ROC AUC** is threshold-independent — it measures how well the model *ranks*
customers by risk. At 0.8424, a randomly chosen churner is scored above a
randomly chosen non-churner about 84% of the time. That makes AUC the right tool
for comparing candidate models, but it cannot tell you where to draw the
decision line. Ranking quality and operating point are separate questions.

### Choosing the threshold

The 0.5 default is an arbitrary convention, not a property of the model. Sweeping
it across the held-out set:

| Threshold | Precision | Recall | F1 | Flagged | Missed |
|---|---|---|---|---|---|
| 0.20 | 0.470 | 0.858 | 0.607 | 683 | 53 |
| 0.25 | 0.498 | 0.807 | 0.616 | 607 | 72 |
| 0.30 | 0.521 | 0.757 | 0.617 | 543 | 91 |
| 0.40 | 0.572 | 0.671 | 0.617 | 439 | 123 |
| 0.50 | 0.656 | 0.556 | 0.602 | 317 | 166 |
| 0.60 | 0.715 | 0.396 | 0.509 | 207 | 226 |
| 0.70 | 0.745 | 0.187 | 0.299 | 94 | 304 |

Moving from 0.50 to 0.25 raises recall from 0.556 to 0.807 — from missing 166
churners to missing 72. The same model, no retraining, just a different cut
point.

I modelled the cost directly, assuming a $50 offer, $770 customer value, and a
**30% save rate** (not every contacted churner is retained). The minimum sits at
threshold 0.15 at $247,697, against $287,980 for doing nothing — about **$40,000
saved**.

I did not ship 0.15. At that threshold the model flags 779 of 1,409 customers —
**55% of the customer base** — which no retention team can act on, and blanket
discounting trains customers to expect discounts. The cost curve is nearly flat
between 0.15 and 0.30 (under $2,000 apart), so I chose **0.25** on operational
grounds: inside the optimal region, flagging a workable 43%.

**One methodological note.** My first cost model assumed every contacted churner
was saved, and it recommended flagging everyone — a degenerate answer. Adding a
realistic save rate fixed it. When an optimiser recommends something absurd, the
assumption is wrong, not the world.

### Shipped performance (threshold 0.25)

```
              predicted stay   predicted churn
actually stay           730               305
actually churn           72               302
```

Precision 0.498 · Recall 0.808 · F1 0.616 · AUC 0.8424

---

## 2. Where the model is weak

**Precision is only 0.50.** Half of the 607 flagged customers were not going to
leave. This is a deliberate trade, not an oversight — at 15:1 cost asymmetry,
305 wasted offers (~$15,000) is cheaper than the revenue behind the churners
they buy. But it means the retention team must be resourced for an alert list
that is wrong half the time, and they should be told that explicitly.

**72 churners are still missed** out of 374 — 19% of departures happen with no
warning. Recall of 0.81 is not 1.0, and the remaining misses are likely
customers who left for reasons absent from the data entirely: a competitor
offer, a house move, a single bad support experience.

**The API validates fields individually but not in combination.** While testing,
I submitted a customer with `tenure: 100` and `TotalCharges: 0` — subscribed for
eight years, never billed. Every field passed its own `Literal` or range check,
the request returned `200 OK`, and the model confidently returned 0.0034. The
input was physically impossible and the model extrapolated into a region of
feature space it never saw in training. A confident number on nonsense input is
the most dangerous failure mode an ML API has, because nothing about the
response signals that anything is wrong.

**The data is a static snapshot with no time dimension.** There is no complaint
history, no usage trend, no support-ticket volume, no record of a recent price
increase. I am predicting from a customer's *state*, not their *trajectory* —
and churn is fundamentally a process that unfolds over time. This is almost
certainly the binding constraint on performance.

**Probabilities are not calibrated-checked.** I treat the output as a
probability and set thresholds on it, but I never verified that customers scored
at 0.30 actually churn ~30% of the time. Logistic regression is usually
reasonably calibrated, but "usually" is not "verified," and the entire cost
model rests on it.

---

## 3. What I would improve with more time or data

### The evidence says: more data, not more model

I tested four model families and three engineered features. Cross-validated AUC:

| Configuration | CV AUC |
|---|---|
| Logistic regression (raw features) | 0.8451 ± 0.0133 |
| Logistic regression (+ engineered features) | 0.8474 ± 0.0116 |
| Logistic regression (class_weight balanced) | 0.8449 ± 0.0134 |
| Random forest (min_samples_leaf=5) | 0.8442 ± 0.0114 |
| Hist gradient boosting | 0.8333 ± 0.0108 |
| Decision tree (max_depth=5) | 0.8319 ± 0.0117 |
| Decision tree (unconstrained) | 0.6600 ± 0.0137 |

**Nothing beat the linear baseline.** My engineered features — average spend per
month of tenure, current-to-historical price ratio, service count, new-customer
flag — gained **+0.0023 AUC against an error bar of ±0.0133**. That is a fifth
of the noise floor. It would be dishonest to report it as an improvement, and
the five-fold cross-validation exists precisely so that I can tell a real gain
from a random one.

The reasonable conclusion is that the signal in this dataset is largely linear
and concentrated in a few columns — contract type, tenure, monthly charges.
Logistic regression captures linear structure completely. Gradient boosting's
advantage is finding complex interactions, and there evidently are not many here
worth finding. Additional model capacity cannot extract signal that is not
present. **More informative features would help; a more powerful model will
not.**

### On overfitting

The unconstrained decision tree is the clearest illustration in this project:

| Model | Train AUC | Test AUC | Gap |
|---|---|---|---|
| Decision tree (unconstrained) | 1.0000 | 0.6613 | +0.3387 |
| Decision tree (max_depth=5) | 0.8525 | 0.8344 | +0.0181 |
| Logistic regression | 0.8493 | 0.8424 | +0.0068 |

A perfect training score of 1.0000 is not success — the tree grew until each
leaf held a single customer, memorising all 5,634 training rows. On unseen
customers it collapsed to 0.6613, far *worse* than the simple linear model.
Capping depth at 5 cut the gap from +0.34 to +0.02.

My shipped model has a train/test gap of **+0.0068**, so it is not overfitting.
If anything it may be slightly underfitting — but the experiments above show
that extra capacity does not convert into test performance here.

### Concrete next steps, in priority order

1. **Behavioural time-series data.** Monthly usage trend, support contacts,
   complaint history, billing disputes, recent plan or price changes. A customer
   whose usage halved last month is a signal this dataset cannot express, and I
   expect this to matter far more than any modelling change.
2. **Cross-field validation in the API.** Reject internally inconsistent
   customers (`tenure > 0` with `TotalCharges == 0`, `TotalCharges` wildly
   inconsistent with `tenure × MonthlyCharges`) rather than returning a
   confident prediction on impossible input.
3. **Validate the cost assumptions with the business.** The $770 value and 30%
   save rate are my estimates, and the threshold recommendation is only as good
   as they are. The retention team's actual campaign conversion rate would
   replace the guess.
4. **Check calibration**, with a reliability curve, before anyone treats the
   probabilities as literal probabilities.
5. **Monitor for drift in production.** Churn drivers change with pricing and
   competition; a model trained on this snapshot will decay, and the metric to
   watch is realised recall against actual churn outcomes, not training scores.
6. **Then revisit resampling and class weighting** for the imbalance —
   `class_weight="balanced"` did not help here (0.8449 vs 0.8451), which is
   itself evidence that threshold tuning was the more effective lever.

---

## Summary

The model is a deliberately simple one, chosen because more complex alternatives
demonstrably did not beat it. It ranks churn risk well (AUC 0.8424) and, at a
threshold chosen on cost and operational grounds rather than convention, catches
four out of five departing customers. Its main weakness is precision — half its
alerts are wrong — which is an accepted consequence of an asymmetric cost
structure, not a defect. The clearest path to a better model is better data
about customer behaviour over time, not a better algorithm.