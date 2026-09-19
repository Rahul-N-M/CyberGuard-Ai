# CyberGuard-AI Ensemble Model: Plain English Deep Dive
### A Comprehensive, Plain-Language Guide to How Our 6-Stage AI Model Works, Why It Was Built, and How It Protects Real Enterprises

---

## 1. Executive Summary (The Big Picture in Plain English)

Imagine you are the Chief Information Security Officer (CISO) of a major hospital or bank. Every single month, software vendors disclose **thousands of new security flaws (called CVEs)** across your servers, employee laptops, database systems, and payment gateways.

Your security engineers have **finite time** — perhaps **10 to 20 hours per week** — to test, approve, and deploy security patches.

Existing industry tools (like traditional vulnerability scanners) sort vulnerabilities solely by their technical severity score (**CVSS score** from 0 to 10). If you have 500 "Critical" CVEs, which one do you actually patch before leaving the office on Friday?

```
   Traditional Industry Tool                 CyberGuard-AI Ensemble
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│ "Here are 500 Critical CVEs.    │       │ "Out of 500 CVEs, only 3 are     │
│  Fix all of them immediately."  │  VS   │  actively being weaponized.     │
│                                 │       │  Fix CVE-A (3h) and CVE-B (4h)  │
│  Outcome: Panic & Alert Fatigue │       │  to reduce 72% of total risk."  │
└─────────────────────────────────┘       └─────────────────────────────────┘
```

Scientific research (such as FIRST.org's EPSS studies) reveals that **less than 1% of all vulnerabilities are ever exploited by real-world hackers**. The remaining 99% are theoretical flaws that nobody attacks.

The **CyberGuard-AI Ensemble** is an intelligent decision-support system that:
1. Accurately predicts the likelihood that a vulnerability will be exploited in the wild within the next 180 days.
2. Contextualizes that likelihood with your company's actual infrastructure (e.g., whether it affects the public payment server vs. an internal dev machine).
3. Tells engineers exactly which patches to deploy within their weekly sprint hours.

---

## 2. Why a Single AI Model Fails (The Need for an Ensemble)

In machine learning, an **"ensemble"** means assembling a team of distinct algorithms that collaborate, debate, and balance each other out — exactly like a hospital board of specialists consulting on a difficult patient rather than relying on one doctor.

When we trained a single standard machine learning model (like a standalone LightGBM tree), it suffered from three fatal real-world problems:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       3 FATAL FLAWS OF A SINGLE MODEL                       │
├─────────────────────────┬───────────────────────────┬───────────────────────┤
│ 1. The Cohort Proxy Trap│ 2. The Saturation Crisis  │ 3. Severe Class Bias  │
│                         │                           │                       │
│ The model gets lazy: it │ The model gets overly     │ Out of 1.5M records,  │
│ assumes every freshly   │ confident: 79,693 CVEs    │ only 601 are true     │
│ published high-CVSS bug │ received a probability of │ positive attacks      │
│ is immediately dangerous│ exactly 1.0000. It is     │ (1 in 2,550). Single  │
│ simply because it's new.│ impossible to prioritize  │ models drown in the   │
│                         │ between 80,000 ties!      │ sea of negatives.     │
└─────────────────────────┴───────────────────────────┴───────────────────────┘
```

By assembling a **6-stage pipeline** (3 diverse base models, 1 stacking meta-judge, 1 mathematical calibrator, and 1 rank aggregator), CyberGuard-AI completely eliminates probability saturation (0 ties) and achieves state-of-the-art predictive accuracy.

---

## 3. The 6 Models & Layers Explained in Plain English

```
                                  Input Features (CVE Text, CVSS, Age)
                                                   │
                     ┌─────────────────────────────┼─────────────────────────────┐
                     ▼                             ▼                             ▼
           ┌───────────────────┐         ┌───────────────────┐         ┌───────────────────┐
           │   Model 1: LGBM   │         │   Model 2: XGB    │         │  Model 3: Extra   │
           │ (Histogram-based) │         │   (Depth-wise)    │         │ (Randomized Trees)│
           │  "The Fast Miner" │         │"The Tough Skeptic"│         │"The Impartial Eye"│
           └─────────┬─────────┘         └─────────┬─────────┘         └─────────┬─────────┘
                     │ p1 = 0.048                  │ p2 = 0.039                  │ p3 = 0.052
                     └─────────────────────────────┼─────────────────────────────┘
                                                   │
                                                   ▼
                                     ┌───────────────────────────┐
                                     │   Model 4: Meta-Learner   │
                                     │ (Ridge Logistic Regressor)│
                                     │    "The Chief Justice"    │
                                     └─────────────┬─────────────┘
                                                   │ Raw Logit Score z = +1.42
                                                   ▼
                                     ┌───────────────────────────┐
                                     │    Model 5: Calibrator    │
                                     │   (Platt Scaling Sigmoid) │
                                     │   "The Reality Check"     │
                                     └─────────────┬─────────────┘
                                                   │ Calibrated P = 0.0421 (4.21%)
                                                   ▼
                                     ┌───────────────────────────┐
                                     │     Layer 6: Ranking      │
                                     │(CVE Borda Rank Aggregator)│
                                     │ "The Actionable Resolver" │
                                     └─────────────┬─────────────┘
                                                   │ Final Borda Score = 99.85th Percentile
                                                   ▼
                                       TOP PRIORITY CVE FOR SPRINT
```

---

### Model 1: LightGBM — "The Fast Pattern Miner"
* **Technical Name:** `lightgbm.LGBMClassifier` (200 trees, leaf-wise splitting)
* **Everyday Analogy:** The emergency room doctor who makes lightning-fast pattern associations based on vital signs.
* **How it works:** LightGBM groups continuous numbers (like CVSS score and vulnerability age) into discrete histogram "buckets". Instead of growing balanced trees row-by-row, it searches for the single leaf that reduces error the most.
* **Its Superpower:** Extremely fast and highly sensitive to non-linear combinations (e.g., `"CVSS > 9.0"` AND `"Description contains 'Remote Code Execution'"`).
* **Its Blindspot:** It can be overly enthusiastic, sometimes over-weighting newly disclosed bugs.

---

### Model 2: XGBoost — "The Tough Skeptic"
* **Technical Name:** `xgboost.XGBClassifier` (150 trees, depth-wise splitting)
* **Everyday Analogy:** The senior auditor who checks every line item and penalizes wild guesses.
* **How it works:** Unlike LightGBM's asymmetric tree growth, XGBoost grows level-by-level (depth-wise) and applies mathematical shrinkage penalties ($L_1$ and $L_2$ regularization) to every branch.
* **Its Superpower:** Keeps LightGBM honest. If a feature looks promising by pure chance in a small cluster of data, XGBoost heavily penalizes that leaf weight, preventing the model from jumping to conclusions.

---

### Model 3: ExtraTrees — "The Impartial Eye"
* **Technical Name:** `sklearn.ensemble.ExtraTreesClassifier` (100 randomized trees)
* **Everyday Analogy:** A diverse jury of citizens who look at the evidence from completely different angles.
* **How it works:** In standard decision trees, the algorithm calculates the mathematically optimal mathematical split point for every feature. ExtraTrees (*Extremely Randomized Trees*) picks split thresholds **randomly** and averages the predictions across 100 trees.
* **Its Superpower:** Variance reduction. In security data, CVSS 9.8 is almost always accompanied by the word "critical" and "buffer overflow". Boosting models get tunnel vision on these dominant keywords. ExtraTrees forces the ensemble to evaluate subtle background signals (like component type and age velocity) that boosting models overlook.

---

### Model 4: Ridge Logistic Regression — "The Chief Justice" (Stacking Meta-Learner)
* **Technical Name:** `sklearn.linear_model.LogisticRegression(C=1.0)`
* **Everyday Analogy:** The presiding judge who listens to the testimonies of Doctor LightGBM, Auditor XGBoost, and Juror ExtraTrees, and decides how much weight to grant each witness.
* **How it works:** Rather than taking a naive mathematical average (which treats a 90% accurate model the same as a 60% accurate model), the Meta-Learner is trained on **Out-Of-Fold (OOF)** predictions. It learns optimal blend weights:
  $$\text{Logit Score } z = w_1 \cdot p_{\text{LGBM}} + w_2 \cdot p_{\text{XGB}} + w_3 \cdot p_{\text{ExtraTrees}} + \text{Bias}$$
* **Why it matters:** If LightGBM is proven to be consistently more reliable on web vulnerabilities, the meta-learner automatically allocates a higher weight to LightGBM for web CVEs.

---

### Model 5: Platt Scaling Calibrator — "The Reality Check"
* **Technical Name:** Logit-space sigmoid calibration
* **Everyday Analogy:** Adjusting a thermometer that reads 150°F down to the true room temperature of 72°F while preserving who is hotter and who is colder.
* **The Critical Problem It Solves:** Tree-based models output raw scores that are not true probabilities. In earlier iterations of CyberGuard, **79,693 CVEs were assigned a score of exactly 1.0000**. If 80,000 vulnerabilities are tied for first place, the tool is completely useless to an engineering team!
* **How it works:** It takes the uncalibrated logit score $z$ from Model 4 and maps it through an empirical sigmoid curve:
  $$P(\text{Exploit}) = \frac{1}{1 + e^{-(A \cdot z + B)}}$$
  where constants $A$ and $B$ are fit against the true background exploit rate of the real world ($\approx 0.027\%$).
* **The Result:** The probabilities now reflect real-world frequencies. Severe vulnerabilities receive probabilities around $3\%$ to $7\%$. While $5\%$ sounds small, it is **over 180 times higher than the background rate**, cleanly separating the dangerous CVEs without creating a single tie at 1.0.

---

### Layer 6: CVE Borda Rank Aggregator — "The Actionable Resolver"
* **Technical Name:** Multi-observation rank percentile Borda count
* **Everyday Analogy:** Computing a student's final GPA by averaging their class percentiles across all semesters, rather than judging them on a single test.
* **The Problem It Solves:** Our machine learning dataset observes vulnerabilities across a 36-month timeline. A single CVE (like Log4Shell) appears in month 1, month 2, month 3, and so on. A security team cannot patch "Log4Shell in March"; they must patch **Log4Shell the software flaw**.
* **How it works:**
  1. For every month a CVE was active, we calculate its percentile rank compared to all other CVEs active in that same month.
  2. We compute the **Borda Score**: the average percentile across all observed months:
     $$\text{Borda Score} = \frac{1}{M} \sum_{m=1}^{M} \text{Percentile}_m$$
* **The Result:** The output is converted from an abstract database row into a clean, actionable score per CVE, ranging from 0.0% (safest) to 100.0% (most urgent).

---

## 4. Complete Walkthrough of a Real-World Example

Let us follow a real vulnerability through the entire 6-stage ensemble:

### The Vulnerability: `CVE-2024-21413` (*Microsoft Outlook Remote Code Execution "MonikerLink"*)
* **CVSS Score:** `9.8` (Critical)
* **Description:** *"Microsoft Outlook Remote Code Execution Vulnerability allows an unauthenticated remote attacker to bypass security checks and execute arbitrary code via malicious moniker links..."*
* **Observation Date:** March 2024 (Age: 25 days, Relative Cohort Percentile: `0.12`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STEP-BY-STEP NUMERICAL TRACE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Feature Extractor:                                                       │
│    • Is_RCE = 1.0 (found "remote code execution")                           │
│    • Is_Remote = 1.0 (found "remote attacker")                              │
│    • Age_Cohort_Pct = 0.12 (in the youngest 12% of bugs that month)         │
│    • CVSS_Norm = 0.98 (9.8 / 10.0)                                          │
│                                                                             │
│ 2. Base Models Evaluate:                                                    │
│    • Model 1 (LightGBM):   p1 = 0.048  (flags high CVSS + RCE keywords)     │
│    • Model 2 (XGBoost):    p2 = 0.039  (cautious due to young age)          │
│    • Model 3 (ExtraTrees): p3 = 0.052  (highlights memory/privilege risk)   │
│                                                                             │
│ 3. Model 4 (Ridge Meta-Learner):                                            │
│    • Combines [0.048, 0.039, 0.052] using learned weights                   │
│    • Computes raw logit score: z = +1.42                                    │
│                                                                             │
│ 4. Model 5 (Platt Scaling Calibrator):                                      │
│    • Maps z = +1.42 to calibrated probability P(Exploit) = 0.0421 (4.21%)   │
│    • Note: 4.21% is 156x higher than background baseline (0.027%)!          │
│    • Zero saturation, zero ties.                                            │
│                                                                             │
│ 5. Layer 6 (CVE Borda Aggregation):                                         │
│    • March 2024 Percentile: 99.85%                                          │
│    • April 2024 Percentile: 99.88%                                          │
│    • May 2024 Percentile:   99.82%                                          │
│    • Final Borda Score: 99.85th Percentile (Rank #1 Top Priority)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. How This Connects to the Enterprise & Knapsack Optimizer

Predicting that `CVE-2024-21413` has a high exploitation probability is only half the battle. If this CVE exists on a developer's isolated sandbox machine with no internet connection, it poses little actual business danger.

Here is how the ML ensemble links directly into the enterprise remediation engine:

### Step A: Business Context Fusion
We calculate the **Actionable Business Risk**:
$$\text{Business Risk} = P(\text{Exploit}) \times \text{Business Impact} \times \text{Exposure Multiplier} \times \left(\frac{\text{CVSS}}{10}\right)$$

* If `CVE-2024-21413` affects the **Public Payment Gateway**:
  $$\text{Risk} = 0.0421 \times 10.0 \text{ (Critical Impact)} \times 1.5 \text{ (Internet Exposed)} \times 0.98 = \mathbf{0.619}$$
* If the same CVE affects an **Internal Staging Box**:
  $$\text{Risk} = 0.0421 \times 2.0 \text{ (Low Impact)} \times 1.0 \text{ (Internal)} \times 0.98 = \mathbf{0.082}$$

The same software flaw is **7.5 times more dangerous** on the payment server than on the staging box!

### Step B: The 0-1 Knapsack Optimization
Your security team has a **10-hour sprint patching quota**.
* Naive sorting by CVSS fixes CVEs in arbitrary order until hours run out, reducing only **12.38 units of risk**.
* CyberGuard's **0-1 Knapsack Optimizer** mathematically selects the combination of fixes that fits inside the 10-hour budget while maximizing total risk reduction, achieving **21.28 units of risk reduction** (**+71.85% more risk eliminated**).

---

## 6. Frequently Asked Questions (For Mentors, Examiners & Non-Technical Stakeholders)

### Q1: Why not use a Deep Learning Neural Network?
**Answer:** In tabular datasets with severe class imbalance (1 positive per 2,550 negatives), Gradient Boosted Decision Trees (GBDTs) consistently outperform Deep Neural Networks. Neural networks require massive parameter tuning to avoid collapsing into predicting all zeros, whereas tree algorithms natively handle tabular column interactions and allow exact feature attribution for auditing.

### Q2: Why not just use CVSS or EPSS directly?
**Answer:**
* **CVSS** measures *technical severity in a vacuum*. A high CVSS bug that has no working exploit in the wild poses zero immediate threat.
* **EPSS** measures *global exploitation probability*, but is completely blind to your company's network. A high EPSS bug on an offline machine is treated the same as one on your primary firewall.
* **CyberGuard-AI** combines exploit likelihood, asset criticality, internet exposure, and engineering patching budgets into one unified decision.

### Q3: How do we know this ensemble actually beats prior academic work?
**Answer:** We conducted a rigorous 4-quadrant benchmark against the published FastEmbed paper ([Fang et al., PLOS ONE 2020](https://doi.org/10.1371/journal.pone.0228439)):
* **On Fang et al.'s own 2009–2015 benchmark:** CyberGuard scores **0.9064 ROC-AUC** vs FastEmbed's **0.8917**.
* **On Fang et al.'s 2013–2018 NVD data:** CyberGuard scores **0.8549 ROC-AUC** vs FastEmbed's **0.8162**.
* **On real-world temporal streams (2022–2024):** FastEmbed's model collapses to **0.5174** (equivalent to flipping a coin), while CyberGuard maintains **0.7720 ROC-AUC** (+224% higher PR-AUC).

---

## 7. Summary of Code & File Locations

| Component | Code Implementation | Jupyter Demonstration Notebook |
| :--- | :--- | :--- |
| **Complete v3 Stacking Ensemble** | [`src/ml/cyberguard_v3_ensemble.py`](file:///d:/ENGG/COE/CyberGuard-Ai/src/ml/cyberguard_v3_ensemble.py) | [`notebooks/01_cyberguard_v3_production_pipeline.ipynb`](file:///d:/ENGG/COE/CyberGuard-Ai/notebooks/01_cyberguard_v3_production_pipeline.ipynb) |
| **FastEmbed Academic Benchmark** | [`src/ml/evaluate_both_on_2009_2015.py`](file:///d:/ENGG/COE/CyberGuard-Ai/src/ml/evaluate_both_on_2009_2015.py) | [`notebooks/02_fastembed_vs_cyberguard_paper_benchmark.ipynb`](file:///d:/ENGG/COE/CyberGuard-Ai/notebooks/02_fastembed_vs_cyberguard_paper_benchmark.ipynb) |
| **Enterprise Knapsack Optimizer** | [`src/optimization/knapsack_solver.py`](file:///d:/ENGG/COE/CyberGuard-Ai/src/optimization/knapsack_solver.py) | [`notebooks/03_enterprise_knapsack_optimization.ipynb`](file:///d:/ENGG/COE/CyberGuard-Ai/notebooks/03_enterprise_knapsack_optimization.ipynb) |
| **Enterprise Asset Topology** | [`src/enterprise/assets_config.py`](file:///d:/ENGG/COE/CyberGuard-Ai/src/enterprise/assets_config.py) | Included in Notebook 03 |
