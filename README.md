# Probabilistic Modeling for Customer Prioritization in a Banking Marketing Campaign

## 1. Project Overview

This project develops a complete Data Science pipeline to predict whether a customer will subscribe to a **term deposit** after a banking marketing campaign.

The goal is not only to build a binary classifier, but to create a model capable of providing calibrated probabilities of subscription, allowing the calculation of the uncertainty associated with each prediction.

The final solution is designed as a decision-support tool for marketing teams, helping them prioritize customers with higher expected conversion probability.

Also, we can provide the global and local feature importance values.

Section `11` of this README file shows the business impact of using the model.

The dataset used is available at https://www.kaggle.com/datasets/abdelazizsami/bank-marketing/data

I also suggest reading:

    [Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. 
    In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.

Available at: 

[pdf] http://hdl.handle.net/1822/14838

[bib] http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt

---

## 2. Business Problem

Marketing campaigns have operational costs and limited contact capacity. If customers are selected randomly, the sales team may spend effort contacting clients with low likelihood of conversion.

The business question is:

> Which customers are more likely to subscribe to the term deposit?

The target variable is:

- `y = yes`: the customer subscribed to the product;
- `y = no`: the customer did not subscribe.

The model is intended to support campaign prioritization by ranking customers according to their calibrated probability of subscription. **Venn-Abers** was used to calibrate the probabilities and calculate the uncertainty interval associated with each prediction.

---

## 3. Understanding Classification Errors

Since this is a business decision problem, it was important to define the cost of classification errors.

### False Positive

The model predicts that a customer is likely to subscribe, but the customer does not subscribe.

In this context, this error may be risky because the business could incorrectly assume that this customer is already highly likely to convert and reduce commercial effort toward them.

### False Negative

The model predicts that a customer is unlikely to subscribe, but the customer would actually subscribe.

This may lead to lower campaign efficiency, but it was considered less critical than incorrectly treating a non-converting customer as a strong opportunity.

Therefore, the project gave special attention to reducing false positives, prioritizing metrics such as:

- Precision;
- F0.5-score (which is the f-beta score with 0.5 beta);
- PR-AUC.

---

## 4. Key Data Decisions

### 4.1 Removing `duration`

The variable `duration` represents the duration of the last contact with the customer.

Although it has strong predictive power, it is only known after the call has already happened. Since the model is intended to support decisions before the campaign action, using this variable would create **data leakage**.

For this reason, `duration` was removed from the modeling stage.

---

### 4.2 Handling `unknown` Values

Some categorical variables contained `unknown` values, especially:

- `job`;
- `education`;
- `contact`;
- `poutcome`.

These values were kept as separate categories instead of being removed or imputed.

The reasoning is that missing categorical information may itself carry predictive signal. For example, most `unknown` values in `poutcome` were associated with customers who had not been contacted in previous campaigns and can be considered a category by itself.

---

### 4.3 Handling `pdays = -1`

The variable `pdays` represents the number of days since the customer was last contacted in a previous campaign.

The value `-1` means that the customer had not been contacted before. Therefore, treating `-1` as a regular numerical value would be misleading.

Two variables were created to replace `pdays`:

- `was_previously_contacted`: indicates whether the customer had been contacted before;
- `pdays_since_previous_contact`: number of days since the previous contact, set to `0` when no previous contact existed.

The original `pdays` variable was not used in the final model.

---

### 4.4 Transforming `day`

The variable `day` represents the day of the month when the contact occurred.

Since the 30th day of the month should not be interpreted as “three times larger” than the 10th day, it was grouped into periods:

- beginning of the month;
- middle of the month;
- end of the month.

The original `day` variable was then removed.

---

## 5. Model Evaluation Strategy

The dataset is imbalanced, with only approximately **11.7%** positive cases. Therefore, `accuracy` was not used as the main evaluation metric.

A model that always predicts the majority class would achieve high accuracy but would not identify any potential subscribers.

The following metrics were prioritized:

- **Precision**: to control false positives;
- **Recall**: to monitor how many actual subscribers were identified;
- **F0.5-score**: to give more weight to Precision than Recall;
- **PR-AUC**: more appropriate for imbalanced datasets focused on the positive class;
- **Brier Score**: to evaluate probability quality;
- **Log Loss**: to penalize incorrect probability estimates, especially confident wrong predictions;
- calibration curves.

A `DummyClassifier` was used as a baseline to validate whether the trained models added value beyond a naive strategy.

---

## 6. Models Tested

The following models were evaluated:

- DummyClassifier;
- Logistic Regression;
- Naive Bayes;
- SVM;
- Random Forest;
- CatBoost.

The CatBoost model was selected as the main candidate because it showed:

- the best PR-AUC among the evaluated models;
- good performance after threshold adjustment;
- strong ability to work with categorical variables;
- results consistent with the exploratory analysis.

---

## 7. Threshold Optimization

Instead of using the default classification threshold of `0.5`, different thresholds were evaluated.

Since false positives were considered especially relevant, the threshold was selected using the **F0.5-score**, which gives more weight to Precision than Recall.

For the base CatBoost model, the best threshold was approximately 0.81. For the calibrated model, the best threshold was 0.41.

The table below shows the comparison between the base model and the calibrated model at their best threshold:

|model | threshold | precision | recall | f0_5 | pr_auc | brier_score | log_loss | total predicted positives |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| CatBoost Base | 0.8100 | 0.5696 | 0.3790 | 0.5176 | 0.4533 | 0.1512 | 0.4824| 704 |
| CatBoost + Venn-Abers | 0.4100 | 0.5340 | 0.4159 | 0.5053 | 0.4360 | 0.0811 | 0.2838 | 824 |

This threshold provided the best balance between reducing false positives and still identifying a relevant share of actual subscribers. 

---

## 8. Hyperparameter Tuning

Hyperparameter tuning was performed using Optuna.

The tuned CatBoost model became more conservative than the base model. It improved Precision from 0.5696 to 0.6077 and reduced false positives from 303 to 215.

However, this improvement came at the cost of lower Recall, which decreased from 0.3790 to 0.3147. The number of true positives also decreased from 401 to 333, while false negatives increased from 657 to 725.

The tuned model had slightly lower F0.5-score and PR-AUC than the base model. F0.5 decreased from 0.5176 to 0.5123, while PR-AUC decreased from 0.4533 to 0.4523.

Although Brier Score and Log Loss improved marginally, the gains were too small to justify replacing the base model.

Therefore, the tuned CatBoost was considered a more conservative alternative, but the base CatBoost was kept as the main model due to its better overall balance between Precision, Recall, F0.5-score, PR-AUC and true positives captured.

---

## 9. Probability Calibration

The base CatBoost model showed good ranking ability, but its predicted probabilities were overestimated.

Therefore, probability calibration was applied using **Venn-Abers**.

The goal of calibration was to make predicted probabilities more reliable. For example, among customers predicted with approximately 70% probability of subscription, around 70% should actually subscribe.

After calibration, probability quality improved significantly:

![Model Calibration](Calibration.png)

| Model | Brier Score | Log Loss |
|---|---:|---:|
| CatBoost Base | 0.1512 | 0.4824 |
| CatBoost + Venn-Abers | 0.0811 | 0.2838 |

Although the calibrated model had a small decrease in some classification metrics, it produced much more reliable probabilities. We can see how the calibration diminished the `brier score` and `log loss` (which are metrics that penalize bad probabilities) by almost 50%.

Since the project's objective involves probabilistic decision-making and risk estimation, the calibrated model was considered more appropriate for business use.

---

## 10. Final Model

The final selected solution was:

```text
CatBoost + Venn-Abers calibration
```

With the calibrated model, the selected threshold was:

```text
threshold = 0.41
```

The calibrated model produced:

- calibrated probability of subscription;
- final predicted class;
- uncertainty interval for the prediction;
- a more reliable probability score for business decision-making.

---

## 11. Business Impact

To translate model performance into business value, the model was compared against a random customer selection strategy.

In the test set, the average subscription rate was approximately:

```text
11.7%
```

This means that if the marketing team selected customers randomly, around 11.7 out of every 100 contacted customers would be expected to subscribe.

Using the calibrated model and the selected threshold, the prioritized group had an observed subscription rate close to:

```text
53.4%
```

For the same number of contacted customers:

| Strategy | Customers Contacted | Expected Subscription Rate | Expected Subscriptions |
|---|---:|---:|---:|
| Random selection | 824 | 11.7% | ~96 |
| Model-based selection | 824 | 53.4% | ~440 |

This represents approximately:

```text
+344 incremental subscriptions
~4.6x lift compared to random selection
```

In practical terms, the model improves the quality of the contact list, allowing the sales team to focus on customers with much higher conversion probability. Alternatively, the marketing team can focus on the customers with lower conversion probability and target them with ads campaigns.

---

## 12. Explainability

Two explainability approaches were used.

### 12.1 Global Feature Importance

Feature importance from CatBoost showed that the most relevant variables were:

- `month`;
- `balance`;
- `poutcome`;
- `contact`;
- `job`;
- `age`;
- `day_period`;
- `campaign`.

These variables are consistent with the exploratory analysis, since they are related to:

- campaign timing;
- customer financial profile;
- previous campaign outcome;
- contact channel;
- occupational profile.

Feature importance should not be interpreted as causality. It only indicates which variables were most useful to the model for prediction.

---

### 12.2 Local Explanation with Calibrated Explanations

A local explanation was generated for one specific prediction using `calibrated-explanations` ( available at https://github.com/Moffran/calibrated_explanations )

This made it possible to understand which features increased or decreased the predicted probability for a specific customer:

In one example, the following factors increased the probability of subscription:

- `contact = cellular`;
- `poutcome = success`;
- `month = jun`;
- `default = no`;
- `age > 60.5`;
- `job = retired`;
- `campaign <= 1.5`;
- `housing = no`.

The main negative factor in that local explanation was:

- `pdays_since_previous_contact > 14.5`.

This explanation is local and describes the model behavior for one customer. It should not be interpreted as a general causal rule.

---

## 13. Production Usage and MLOps

In production, the model could be used as a commercial prioritization engine.

For each new customer base, the pipeline should:

1. apply the same preprocessing rules used during training;
2. generate the calibrated probability of subscription;
3. estimate prediction uncertainty;
4. apply the selected threshold;
5. assign the final predicted class;
6. return a prioritized customer list.

The expected output for each customer could include:

- customer identifier;
- calibrated probability of subscription;
- uncertainty interval;
- final predicted class;
- threshold used;
- model version.

---

## 14. Monitoring

After deployment, the model should be continuously monitored.

Key monitoring points include:

- input feature distributions;
- unseen or new categorical levels;
- distribution of predicted probabilities;
- number of customers classified as positive;
- Precision;
- Recall;
- F0.5-score;
- PR-AUC;
- Brier Score;
- Log Loss;
- calibration quality over time.

The model should be retrained or reviewed when:

- performance decreases significantly;
- probability calibration worsens;
- the customer profile changes;
- many new categories appear;
- the commercial strategy changes;
- enough new labeled data becomes available.

---

## 15. Technologies Used

- Python;
- pandas;
- numpy;
- matplotlib;
- seaborn;
- scikit-learn;
- CatBoost;
- Optuna;
- [MAPIE](https://mapie.readthedocs.io/en/stable/);
- [calibrated-explanations](https://github.com/Moffran/calibrated_explanations);
- joblib.

---

## 16. Expected Project Structure

```text
.
├── bank-full.csv
├── Notebook.ipynb
├── requirements.txt
├── README.md
└── Artifacts/
    └── catboost_venn_abers_artifacts.pkl
```

---

## 17. How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Or using `uv`:

```bash
uv pip install -r requirements.txt
```

Then run the notebook.

If using Jupytext, the `.py` file can be opened as a notebook-compatible script.

---

## 18. Conclusion

This project showed that it is possible to build a model that supports banking marketing campaigns by prioritizing customers with higher probability of subscription.

The final solution:

- outperformed the naive baseline;
- generated calibrated probabilities;
- allowed risk-aware decision-making;
- provided global and local explainability;
- can be integrated into a production workflow with monitoring and retraining.

From a business perspective, the model showed a potential lift of approximately **4.6x** compared to random customer selection for the same number of contacted customers.

The main value of the model is not simply predicting `yes` or `no`, but helping the commercial team allocate effort more efficiently by focusing on customers with higher expected conversion probability.
