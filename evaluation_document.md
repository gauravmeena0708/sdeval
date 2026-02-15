# A Comprehensive Guide to Evaluating Conditional Synthetic Tabular Data

## 1. Introduction

Evaluating the quality of synthetic data is a critical but complex task. For **conditional** synthetic data, the challenge is amplified. It's not enough for the generated data to match the overall statistical properties of the real data; it must also accurately capture the **conditional distributions**. For example, if we generate data conditioned on `country='USA'`, the distributions of features like `income` and `age` within that slice must match the real data for the USA, not the global average.

This document provides a comprehensive framework for evaluating and comparing different methods for generating conditional synthetic tabular data. It is based on a critical analysis of the robust evaluation suite found in the `sd_framework` project, supplemented with other standard best practices.

## 2. The Core Pillars of Evaluation

A thorough evaluation should be structured around three fundamental pillars, each answering a key question about the synthetic data's quality.

1.  **Pillar 1: Statistical Fidelity & Correctness**
    *   *Question:* Does the synthetic data have the same statistical properties as the real data, both globally and within specific conditions?

2.  **Pillar 2: Efficacy & Utility**
    *   *Question:* Can the synthetic data be used as a drop-in replacement for real data in a downstream machine learning task?

3.  **Pillar 3: Privacy**
    *   *Question:* Does the synthetic data expose private information about individuals in the real dataset?

## 3. Detailed Metrics for Each Pillar

### Pillar 1: Statistical Fidelity & Correctness

This is the most critical pillar for conditional generation. We break it down into three parts: a fundamental check for correctness, an assessment of global (overall) quality, and a deep dive into conditional quality.

#### A. The Sanity Check: Constraint Satisfaction Rate (CSR)

*   **What it is:** The percentage of rows in the synthetic dataset that correctly satisfy the condition(s) they were generated for.
*   **How to Interpret:** This is a non-negotiable, pass/fail metric. If you ask a model to generate data where `age > 50`, the CSR tells you what percentage of the output records actually have `age > 50`.
*   **Analysis:** A CSR below 100% (or very close to it, allowing for minor floating point issues) indicates a fundamental failure of the conditional generation method. **This should be the very first metric you check.**

#### B. Global Fidelity: Assessing Overall Distributional Similarity

These metrics provide a high-level view of how well the synthetic data matches the real data across the entire dataset.

*   **Kolmogorov-Smirnov (KS) Complement (for Numerical Columns):**
    *   **What it is:** Measures the similarity between the distribution of a numerical column in the real data and the synthetic data. It is the complement of the standard KS-test statistic.
    *   **Interpretation:** A score of **1.0 is a perfect match**, while a score of 0.0 indicates completely different distributions. Higher is better. The `avg_ks_complement` is the average of this score across all numerical columns.
*   **Alpha-Precision & Beta-Recall (for Categorical Columns):**
    *   **`alpha-precision`:** Answers: "Are the synthetic data points realistic?" It measures the fraction of categories in the synthetic data that also appear in the real data. A low score means the model is "hallucinating" or inventing categories that don't exist, which is bad. **Higher is better.**
    *   **`beta-recall`:** Answers: "Is the synthetic data diverse enough?" It measures the fraction of real-data categories that are captured in the synthetic data. A low score means the model is failing to generate rare categories (a common problem known as mode collapse). **Higher is better.**
*   **Correlation Matrix Difference:**
    *   **What it is:** A visual comparison of the correlation matrices between all numerical columns in the real vs. synthetic data.
    *   **Interpretation:** The heatmaps of the two matrices should look visually similar. A heatmap of the *absolute difference* between the two matrices is even better; in this case, a dark matrix (values close to zero) is ideal.

#### C. Conditional Fidelity: The True Test of Conditional Models

This is the most important part of the evaluation. It assesses whether the relationships *between* variables are preserved correctly for a given condition.

*   **The Plausibility Score (Per-Row NLL):**
    *   **What it is:** This is a powerful, holistic metric that scores each individual row on how "likely" or "plausible" it is. It is the **negative log-likelihood (NLL)** of the row, calculated by an auto-regressive model trained on the real data.
    *   **How it works:** An auto-regressive model learns the probability of a feature's value given the values of the features that came before it in the row (`p(feature_i | feature_1, ..., feature_{i-1})`). The NLL score is the sum of the negative logs of these probabilities. A row with a common, expected combination of features will have a low score, while a row with a rare or nonsensical combination (e.g., `occupation='Surgeon'` and `education='High School dropout'`) will have a very high score.
    *   **Interpretation:** **Lower is better.** You should compare the *distribution* of plausibility scores for different synthetic datasets. A model that produces a distribution with a lower average plausibility score and a tighter distribution is generally better at capturing the complex, joint distribution of the real data.
    *   **Critical Analysis:** This is arguably the **single most important metric for conditional quality**, as it directly assesses the correctness of the relationships between features within each generated row. Its main weakness is that it depends on a well-trained plausibility model.
*   **Conditional Distribution Analysis (Slice and Compare):**
    *   **What it is:** The practice of filtering both the real and synthetic data by a specific condition, and then running standard distribution comparison metrics on those slices.
    *   **Example:** To check if `age` is generated correctly for high-income individuals, create two dataframes: `real_high_income = real_df[real_df['income'] == '>50K']` and `synth_high_income = synthetic_df[synthetic_df['income'] == '>50K']`. Then, calculate the KS Complement for the `age` column between these two sliced dataframes.
    *   **Interpretation:** This provides direct, quantitative evidence of how well the conditional distributions are being modeled. This should be repeated for several important feature/condition combinations.

### Pillar 2: Efficacy & Utility

*   **Train-Synthetic-Test-Real (TSTR):**
    *   **What it is:** This metric answers the question: "Is the synthetic data actually useful for training a machine learning model?" The process is:
        1. Train a model on the **real** data and test it on a held-out set of **real** data (TRTR). This is your baseline performance.
        2. Train the same model on the **synthetic** data and test it on the same held-out set of **real** data (TSTR).
    *   **Interpretation:** The TSTR score should be as close as possible to the TRTR score. The ratio `TSTR / TRTR` is often reported. A ratio close to 1.0 indicates that the synthetic data is a high-quality substitute for the real data for that specific ML task.
    *   **Critical Analysis:** This is a very practical and important measure of utility. It should be performed for both the full dataset and for conditional slices to ensure utility holds across different sub-populations.

### Pillar 3: Privacy

*   **Distance to Closest Record (DCR):**
    *   **What it is:** For each synthetic record, this metric finds the Euclidean distance to the *single closest* record in the real dataset.
    *   **Interpretation:** A very small DCR value is a red flag, suggesting that a synthetic record might be a near-perfect copy of a real one, indicating a privacy leak. You should look at the **mean DCR** and, more importantly, the **5th percentile DCR** (which represents the "worst-case" privacy risk). **Higher is better.**
*   **Nearest Neighbor Distance Ratio (NNDR):**
    *   **What it is:** The ratio of the distance to the *closest* real record to the distance to the *second closest* real record.
    *   **Interpretation:** A value close to 0 suggests the synthetic point is an outlier, while a value close to 1 suggests it's in a dense region of the real data, which can also be a privacy concern.
*   **Privacy Score:**
    *   **What it is:** The `sd_framework` calculates a `dcr_baseline` by measuring the DCR of a *shuffled* version of the real data against the original. The final `privacy_score` is `dcr_mean / dcr_baseline`.
    *   **Interpretation:** This is an excellent, easy-to-interpret summary. A score **>= 1.0 is good**, as it means your synthetic data points are, on average, at least as far from real data points as randomly shuffled data is. A score significantly below 1.0 suggests potential memorization and privacy leakage.
*   **Membership Inference Attack (MIA):**
    *   **What it is:** This metric simulates an attacker trying to determine if a specific record from the real dataset was used in the training of the generative model. It trains a classifier to distinguish between synthetic records and real records.
    *   **Interpretation:** The accuracy of this classifier is the MIA score. A score of 50% is ideal (the classifier is just guessing), while a score of 100% means the synthetic data is easily distinguishable and offers poor privacy. **Lower is better.**

## 4. Recommended Methodology for Comparison

Here is a step-by-step guide to comparing two or more conditional synthetic datasets (e.g., from `method_A` and `method_B`).

**Step 1: The Gatekeeper - Check Constraint Satisfaction**
*   For each dataset, calculate the **CSR** for the conditions it was generated under.
*   **Decision:** If any model has a CSR significantly below 100%, it should be heavily penalized or even disqualified, as it failed the primary task.

**Step 2: High-Level Quality Check**
*   Calculate all the **global fidelity metrics** (`avg_ks_complement`, `alpha_precision`, `beta_recall`) and the global **TSTR score**.
*   **Decision:** This gives you a quick overview. Does one model clearly outperform the other on all metrics?

**Step 3: The Deep Dive - Conditional Correctness**
*   This is the most important step for conditional evaluation.
*   **a) Compare Plausibility Scores:** For each model, generate a distribution of **Plausibility Scores** for all synthetic rows. Plot these distributions side-by-side (e.g., using a boxplot or violin plot).
    *   **Decision:** The model that produces a distribution with a lower median score and less variance (i.e., fewer high-score outliers) is better at generating realistic combinations of features.
*   **b) Slice and Dice:** Choose 2-3 of the most important conditions and/or features. For each:
    *   Filter the real and synthetic datasets to only include rows matching that condition.
    *   On this data slice, calculate the **KS Complement** for a key numerical feature.
    *   On this data slice, calculate the **TSTR score** if applicable.
    *   **Decision:** The model that maintains better statistical similarity and ML utility *within* these conditional slices is the superior conditional generator.

**Step 4: Assess Privacy Risk**
*   Calculate the **DCR (5th percentile)** and the **Privacy Score** for each dataset.
*   **Decision:** Favor the model with the higher DCR and a Privacy Score closer to or above 1.0.

**Step 5: Synthesize and Decide**
*   Create a summary table comparing the models across all the key metrics from the steps above.

| Metric                        | Method A | Method B | Winner |
| ----------------------------- | -------- | -------- | ------ |
| **CSR**                       | 100%     | 98%      | A      |
| **Avg KS Complement (Global)**  | 0.85     | 0.82     | A      |
| **TSTR (Global)**             | 0.92     | 0.90     | A      |
| **Plausibility (Median NLL)**   | 15.3     | 18.1     | A      |
| **KS Comp (income | age>50)**   | 0.78     | 0.65     | A      |
| **Privacy Score**             | 0.95     | 1.15     | B      |
| **MIA Score**                 | 55%      | 65%      | A      |

*   **Final Decision:** In the example above, Method A is superior in fidelity and utility, while Method B is better for privacy. The "best" model depends on the final application. For most use cases, Method A would be preferred, but if privacy is paramount, Method B might be chosen despite its slightly lower fidelity.

## 5. Conclusion

Evaluating conditional synthetic data requires going beyond global statistical checks. By systematically assessing **Constraint Satisfaction**, **Conditional Fidelity** (via Plausibility Scores and sliced analysis), **Utility**, and **Privacy**, you can build a comprehensive picture of a model's performance and make an informed decision. Remember to always evaluate metrics on conditional slices of the data, as this is the true test of a conditional generative model.
