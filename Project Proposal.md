# Project Proposal

## Multi-Aspect Sentiment Analysis of Restaurant Reviews Using Natural Language Processing

### 1. Introduction

Online restaurant review platforms contain large amounts of user-generated feedback describing customers' experiences with food, service, ambience, pricing, cleanliness, and other characteristics of restaurants. Such reviews can provide valuable information to both customers and restaurant owners. However, the large volume and unstructured nature of textual reviews makes manual analysis difficult.

Traditional sentiment analysis typically assigns a single sentiment label, such as positive, negative, or neutral, to an entire review. This approach has an important limitation in the restaurant domain because customers frequently discuss multiple aspects of their experience within the same review and may express different sentiments toward each aspect.

For example:

> "The pasta was excellent, but the service was extremely slow and the restaurant was expensive."

A traditional sentiment classifier may attempt to assign one overall sentiment to this review. However, the review actually contains several distinct opinions:

- Food → Positive
- Service → Negative
- Price/Value → Negative

Aspect-Based Sentiment Analysis (ABSA) addresses this limitation by identifying the aspects discussed in a piece of text and determining the sentiment associated with each aspect.

This project proposes the development and systematic evaluation of a multi-aspect sentiment analysis system for restaurant reviews. The system will identify multiple restaurant-related aspects within a review and predict the sentiment associated with each identified aspect.

The project will investigate both traditional machine-learning/NLP baselines and modern transformer-based approaches. Particular attention will be given to difficult cases such as reviews containing multiple aspects, conflicting sentiments, implicit aspect references, and linguistic phenomena such as negation.

---

# 2. Problem Statement

Conventional sentiment-analysis systems commonly represent a review using a single sentiment label.

For example:

**Input**

"The burgers were delicious, but the staff were rude."

**Traditional sentiment analysis**

Review → Mixed/Negative

This representation loses important information because the customer has expressed different opinions about different components of the restaurant experience.

A more informative representation would be:

| Aspect | Sentiment |
|---|---|
| Food | Positive |
| Service | Negative |

The problem becomes more challenging when reviews contain several aspects:

**Input**

"The food was fantastic, the restaurant was beautiful, but we waited nearly an hour and the prices were ridiculous."

**Desired output**

| Aspect | Sentiment |
|---|---|
| Food | Positive |
| Ambience | Positive |
| Service | Negative |
| Price/Value | Negative |

Furthermore, aspects are not always explicitly named.

For example:

> "We waited 45 minutes before anyone took our order."

The word "service" does not occur, but the sentence clearly expresses a negative opinion concerning service.

Therefore, the proposed system must solve two related problems:

1. **Aspect Detection:** Determine which restaurant aspects are discussed.
2. **Aspect-Level Sentiment Classification:** Determine the sentiment expressed toward every detected aspect.

Since a review may contain multiple aspects simultaneously, aspect detection is fundamentally a **multi-label classification problem** rather than conventional single-label classification.

---

# 3. Aim

The primary aim of this project is to **design, implement, and evaluate an Aspect-Based Sentiment Analysis system capable of detecting multiple aspects in restaurant reviews and determining the sentiment associated with each detected aspect.**

The project will additionally investigate how different NLP approaches perform on challenging multi-aspect and implicit-aspect reviews.

---

# 4. Objectives

The project will pursue the following objectives:

1. Study existing research in sentiment analysis and Aspect-Based Sentiment Analysis.

2. Identify and prepare suitable publicly available restaurant ABSA datasets.

3. Define an appropriate taxonomy of restaurant aspects.

4. Develop a multi-label aspect detection model capable of identifying multiple aspects from a single review or sentence.

5. Develop an aspect-level sentiment classifier that predicts positive, negative, or neutral sentiment for each identified aspect.

6. Implement appropriate baseline methods for comparison.

7. Develop a transformer-based ABSA approach using a pretrained language model such as BERT or a related architecture.

8. Evaluate the models using appropriate multi-label and sentiment-classification metrics.

9. Investigate model performance on difficult cases including:
   - multiple aspects,
   - conflicting sentiments,
   - implicit aspects,
   - negation,
   - long or complex sentences.

10. Perform detailed error analysis to identify strengths and limitations of the developed approaches.

11. Develop a prototype application demonstrating the practical use of the resulting ABSA system.

---

# 5. Research Questions

The primary research question is:

**RQ1: How effectively can NLP models identify multiple restaurant aspects and their associated sentiments from customer reviews?**

The project will additionally investigate:

**RQ2:** How does a transformer-based ABSA approach compare with simpler machine-learning/NLP baselines for multi-aspect restaurant review analysis?

**RQ3:** How does model performance differ between reviews containing a single aspect and reviews containing multiple aspects?

**RQ4:** How effectively can the developed system handle implicit aspects where the aspect category is not explicitly stated?

**RQ5:** What linguistic and contextual phenomena cause the largest number of errors in aspect detection and aspect-level sentiment classification?

These questions allow the project to go beyond simply reporting the accuracy of one trained model.

---

# 6. Scope of the Project

The project will focus primarily on **English-language restaurant reviews**.

A predefined or dataset-derived set of restaurant aspect categories will be used. Depending on the selected dataset, these may include categories such as:

- Food quality
- Food prices
- Food style/options
- Service
- Ambience
- Restaurant prices/value
- Location
- Restaurant/general experience

For a simpler application interface, fine-grained categories may additionally be mapped into broader categories such as:

**Food, Service, Ambience, Price/Value, and Other.**

The project will primarily consider three sentiment classes:

**Positive, Negative, Neutral**

Additional labels such as conflict may be included if supported by the selected benchmark.

---

# 7. Proposed System

The conceptual system is:

**Restaurant Review**

↓

**Text Preprocessing / Tokenization**

↓

**Multi-Label Aspect Detection**

↓

**Detected Aspects**

↓

**Aspect-Level Sentiment Classification**

↓

**Aspect–Sentiment Pairs**

For example:

**Input**

"The steak was amazing, but the waiter was rude and it was far too expensive."

**Output**

| Detected Aspect | Predicted Sentiment |
|---|---|
| Food | Positive |
| Service | Negative |
| Price/Value | Negative |

Aspects that are not discussed should not receive sentiment predictions.

Therefore, the system is effectively learning:

**Review → {(Aspect₁, Sentiment₁), (Aspect₂, Sentiment₂), ..., (Aspectₙ, Sentimentₙ)}**

where *n* varies between reviews.

---

# 8. Aspect Detection as Multi-Label Classification

Unlike ordinary classification, the proposed aspect detector cannot assume that every input belongs to exactly one category.

For example:

> "Great food and atmosphere but terrible service."

The correct labels might simultaneously be:

**Food = 1**

**Service = 1**

**Ambience = 1**

**Price = 0**

Consequently, aspect detection will be formulated as a **multi-label text classification problem**.

For a system containing *k* possible aspects, the model can output a vector:

`[Food, Service, Ambience, Price, ...]`

For example:

`[1, 1, 1, 0, ...]`

The model will estimate a probability for each aspect independently. Suitable thresholds will then determine whether each aspect is considered present.

This component will be evaluated using multi-label metrics such as precision, recall, F1-score and exact-match accuracy.

---

# 9. Aspect-Level Sentiment Classification

Once an aspect has been detected, the system must determine the sentiment associated with that specific aspect.

The classifier can therefore operate conceptually on:

**(Review, Aspect) → Sentiment**

For example:

**Review:** "The food was amazing but service was painfully slow."

**Aspect:** Food

→ **Positive**

The same review combined with:

**Aspect:** Service

→ **Negative**

This formulation allows a single review to generate different sentiment predictions depending on the target aspect.

---

# 10. Datasets

## 10.1 SemEval Restaurant ABSA Datasets

The primary benchmark will be selected from the SemEval Aspect-Based Sentiment Analysis restaurant datasets.

The SemEval ABSA benchmarks provide manually annotated restaurant-review text containing aspect and sentiment information and have been extensively used in ABSA research.

Importantly for this project, individual sentences can contain **multiple annotated aspects**, allowing the proposed multi-aspect system to be trained and evaluated.

The datasets include fine-grained categories such as:

- FOOD#QUALITY
- FOOD#PRICES
- FOOD#STYLE_OPTIONS
- SERVICE#GENERAL
- AMBIENCE#GENERAL
- RESTAURANT#PRICES
- RESTAURANT#GENERAL

The final selection of SemEval version will be determined after examining annotation structure, dataset size, and compatibility with the research questions.

## 10.2 Additional Dataset

An additional ABSA dataset may be incorporated for experiments involving implicit aspects.

Restaurant-ACOS or related restaurant ABSA resources are potential candidates because newer ABSA formulations include explicit and implicit aspect/opinion information.

The second dataset would primarily be used to investigate **generalization and implicit-aspect performance**, rather than unnecessarily increasing the scope of the primary system.

---

# 11. Explicit and Implicit Aspects

An important experimental component will distinguish between explicit and implicit aspects.

### Explicit

> "The service was terrible."

The aspect is explicitly mentioned.

**Service → Negative**

### Implicit

> "We waited 50 minutes before anyone took our order."

The word "service" does not occur.

Nevertheless:

**Service → Negative**

Another example is:

> "₹900 for two tiny pieces of chicken."

which can imply:

**Price/Value → Negative**

Implicit aspect detection is considerably more difficult because the model must understand contextual meaning rather than simply associate aspect-related keywords with labels.

Performance on explicit and implicit examples will therefore be separately analysed where dataset annotations permit.

---

# 12. Proposed Models

Rather than developing only one model, the project will compare approaches of increasing sophistication.

## 12.1 Baseline

A conventional text-classification baseline will first be implemented.

Possible methods include:

**TF-IDF + Logistic Regression**

or

**TF-IDF + Support Vector Machine**

For aspect detection, one-vs-rest classification can be used to support multiple labels.

This baseline establishes how well relatively simple lexical models can solve the problem.

## 12.2 Transformer-Based Model

A pretrained transformer model will then be fine-tuned for the task.

Potential models include:

- BERT
- RoBERTa
- DistilBERT

For aspect detection, the transformer can produce independent probabilities for each aspect.

For sentiment classification, the review text and target aspect can be provided together to the model so that sentiment is conditioned on the selected aspect.

For example:

`[CLS] The food was excellent but service was terrible. [SEP] FOOD [SEP]`

→ Positive

and:

`[CLS] The food was excellent but service was terrible. [SEP] SERVICE [SEP]`

→ Negative

This allows the same review to produce different sentiment predictions for different aspects.

## 12.3 Optional Advanced Experiment

If time and computational resources permit, an additional experiment may compare the fine-tuned model with a modern instruction-following language model using zero-shot or few-shot prompting.

This experiment will remain an extension rather than a requirement for successful completion of the project.

---

# 13. Experimental Design

The experiments will be structured to answer the research questions rather than merely obtain a single accuracy score.

### Experiment 1 — Baseline Aspect Detection

Train a conventional machine-learning model for multi-label aspect detection.

### Experiment 2 — Transformer Aspect Detection

Fine-tune a pretrained transformer and compare it against the baseline.

### Experiment 3 — Aspect-Level Sentiment Classification

Evaluate sentiment prediction when the target aspect is known.

### Experiment 4 — End-to-End Evaluation

Combine aspect detection and sentiment classification.

This experiment measures the real system:

**Raw review → Aspect–sentiment pairs**

rather than evaluating each component independently.

### Experiment 5 — Single vs Multi-Aspect Reviews

Separate the test data into reviews/sentences containing:

- one aspect,
- two aspects,
- three or more aspects.

Performance can then be compared across these groups.

This will directly answer whether models struggle as the number of simultaneously expressed opinions increases.

### Experiment 6 — Explicit vs Implicit Aspects

Where appropriate annotations are available, examples will be separated into explicit and implicit aspect groups.

The experiment will determine how strongly models rely on direct lexical mentions.

### Experiment 7 — Error Analysis

Incorrect predictions will be manually categorised.

Potential error categories include:

- implicit aspects,
- negation,
- mixed sentiment,
- multiple aspects,
- sarcasm,
- ambiguous language,
- unusual vocabulary,
- long-distance relationships between aspect and opinion,
- incorrect aspect boundaries.

This analysis will provide a deeper research contribution than reporting aggregate metrics alone.

---

# 14. Evaluation Metrics

Different components require different evaluation metrics.

## Aspect Detection

Because aspect detection is multi-label, evaluation will include:

- Precision
- Recall
- Micro F1-score
- Macro F1-score
- Per-aspect F1-score
- Exact-match accuracy where appropriate

Macro F1 is particularly useful because some restaurant aspects may occur considerably less frequently than others.

## Sentiment Classification

Sentiment classification will be evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion matrix

## End-to-End ABSA

The complete system will additionally be evaluated according to whether the predicted:

**(Aspect, Sentiment)**

pair matches the ground-truth pair.

For example:

Ground truth:

`(Food, Positive)`

Prediction:

`(Food, Negative)`

would be considered incorrect despite correctly detecting the Food aspect.

---

# 15. Prototype Application

A lightweight prototype will demonstrate the practical capabilities of the trained model.

A user will be able to enter a restaurant review such as:

> "Loved the biryani and the place looked beautiful, but the staff were extremely slow and it wasn't worth the price."

The application could display:

| Aspect | Sentiment |
|---|---|
| Food | 🟢 Positive |
| Ambience | 🟢 Positive |
| Service | 🔴 Negative |
| Price/Value | 🔴 Negative |

The prototype may additionally support aggregation across many reviews.

For example:

### Restaurant Summary

| Aspect | Positive | Neutral | Negative |
|---|---:|---:|---:|
| Food | 82% | 6% | 12% |
| Service | 49% | 8% | 43% |
| Ambience | 76% | 10% | 14% |
| Price/Value | 35% | 12% | 53% |

This demonstrates why aspect-based sentiment analysis provides considerably more actionable information than an overall star rating or conventional sentiment score.

The prototype is intended as a demonstration of the research results rather than the primary research contribution.

---

# 16. Proposed Technology Stack

The implementation is expected to use:

**Programming language:** Python

**Machine learning:** Scikit-learn

**Deep learning:** PyTorch

**NLP/Transformers:** Hugging Face Transformers

**Data processing:** Pandas, NumPy

**Evaluation:** Scikit-learn metrics

**Visualisation:** Matplotlib and related libraries

**Prototype:** Streamlit or similar lightweight framework

**Version control:** Git/GitHub

Where GPU resources are required, experiments may be conducted using university computing resources or cloud notebook environments such as Google Colab or Kaggle.

---

# 17. Two-Semester Work Plan

## Semester 1 — Foundations and Baselines

### Phase 1: Literature Review

Study:

- traditional sentiment analysis,
- Aspect-Based Sentiment Analysis,
- aspect-category detection,
- multi-label text classification,
- aspect sentiment classification,
- transformer models for ABSA,
- implicit aspect detection.

A structured literature review will identify the limitations of existing methods and refine the final research questions.

### Phase 2: Dataset Investigation

Obtain and analyse the selected SemEval restaurant dataset.

Tasks include:

- understanding annotation formats,
- converting data to a consistent representation,
- analysing class distributions,
- identifying multi-aspect examples,
- examining sentiment distributions,
- identifying class imbalance.

### Phase 3: Exploratory Data Analysis

Analyse:

- number of samples,
- frequency of each aspect,
- frequency of each sentiment,
- number of aspects per sample,
- aspect/sentiment combinations,
- review lengths,
- explicit versus implicit examples where possible.

### Phase 4: Baseline Development

Implement traditional ML baselines.

Establish baseline results for:

- aspect detection,
- aspect sentiment classification.

### Phase 5: Initial Transformer Model

Fine-tune the selected pretrained transformer.

Compare initial results with the baseline.

### Semester 1 Deliverables

By the end of Semester 1:

- literature review,
- finalized research questions,
- cleaned/prepared dataset,
- exploratory analysis,
- baseline system,
- initial transformer model,
- preliminary experimental results.

---

## Semester 2 — Advanced Experiments, Analysis and Final System

### Phase 6: Model Refinement

Optimise the transformer-based system.

Potential work includes:

- hyperparameter tuning,
- classification threshold optimisation,
- handling class imbalance,
- alternative input representations,
- model comparison.

### Phase 7: Multi-Aspect Analysis

Analyse performance according to number of aspects.

Compare:

**single-aspect vs multi-aspect samples.**

### Phase 8: Implicit Aspect Analysis

Evaluate implicit aspect detection where supported by the selected dataset.

Compare:

**explicit vs implicit aspect performance.**

### Phase 9: End-to-End ABSA Evaluation

Evaluate the complete pipeline:

**Review → Aspect detection → Sentiment prediction**

and quantify error propagation between the two stages.

### Phase 10: Error Analysis and Ablation Studies

Perform systematic error analysis.

Where appropriate, conduct ablation experiments to determine which components or modelling choices contribute to performance.

### Phase 11: Prototype Development

Develop the demonstration application and restaurant-level aggregation interface.

### Phase 12: Dissertation and Presentation

Complete:

- final dissertation/report,
- experiment documentation,
- result visualisations,
- final system demonstration,
- presentation/poster,
- source-code documentation.

---

# 18. Expected Contributions

The project is not intended to claim the invention of a fundamentally new deep-learning architecture.

Instead, its contribution will consist of a rigorous investigation and implementation of multi-aspect sentiment analysis for restaurant reviews.

Expected contributions include:

1. An end-to-end ABSA system capable of identifying multiple restaurant aspects.

2. Aspect-specific sentiment predictions rather than a single review-level sentiment.

3. An empirical comparison between traditional machine-learning and transformer-based approaches.

4. Analysis of model performance on single-aspect versus multi-aspect reviews.

5. Investigation of explicit versus implicit aspect detection where supported by the selected data.

6. Detailed analysis of common ABSA failure cases.

7. A practical prototype demonstrating how aspect-level predictions can be aggregated into useful restaurant insights.

---

# 19. Expected Challenges

### Class Imbalance

Some aspects may occur much more frequently than others.

Possible solutions include weighted loss functions, threshold tuning and appropriate macro-level evaluation.

### Implicit Aspects

Implicit aspects may be difficult to detect because no direct aspect keyword appears.

Transformer-based contextual models are expected to outperform purely lexical baselines in these situations, providing an interesting hypothesis for experimentation.

### Conflicting Sentiments

A single sentence can contain opposite sentiments toward different aspects.

For example:

> "Excellent food but horrible service."

The model must correctly associate each sentiment with its corresponding aspect.

### Dataset Size

High-quality manually annotated ABSA datasets are relatively small compared with general sentiment datasets.

Transfer learning from pretrained language models therefore becomes particularly useful.

### Error Propagation

In a pipeline architecture, failure to detect an aspect means sentiment classification cannot recover the missing prediction.

For this reason, component-level and end-to-end performance will both be measured.

---

# 20. Ethical Considerations

The primary datasets used in this project will be publicly available research datasets.

The system will analyse opinions expressed in restaurant reviews and will not attempt to infer sensitive personal characteristics of reviewers.

Potential biases within review datasets will nevertheless be acknowledged. For example, language patterns and restaurant categories represented in benchmark datasets may not accurately represent restaurants, customers, cultures, or dialects in other geographic regions.

The resulting system should therefore be interpreted as a research prototype rather than an objective measure of restaurant quality.

---

# 21. Project Limitations

Several limitations are anticipated.

First, the project will primarily focus on English-language restaurant reviews.

Second, performance will depend on the aspect taxonomy provided by or derived from the selected datasets.

Third, implicit opinions, sarcasm, humour and highly context-dependent language are expected to remain challenging.

Fourth, benchmark datasets may contain relatively short review sentences and therefore may not fully represent long, noisy reviews found on real-world platforms.

Finally, because this is an undergraduate project, the emphasis will be placed on rigorous experimentation and analysis rather than training very large language models from scratch.

---

# 22. Success Criteria

The project will be considered successful if it produces:

1. A functioning multi-label restaurant aspect detector.
2. A functioning aspect-level sentiment classifier.
3. An integrated end-to-end ABSA pipeline.
4. Reproducible baseline and transformer experiments.
5. Quantitative evaluation using appropriate metrics.
6. Experimental analysis of multi-aspect cases.
7. Analysis of implicit aspects where dataset annotations permit.
8. Detailed error analysis.
9. A working demonstration application.
10. A documented and reproducible codebase.

A successful project does **not** require achieving state-of-the-art benchmark performance. The academic value comes from properly defining the problem, designing controlled experiments, comparing methods, interpreting the results and identifying limitations.

---

# 23. Proposed Project Title

**Multi-Aspect Sentiment Analysis of Restaurant Reviews Using Transformer-Based Natural Language Processing**

A more research-oriented alternative is:

**Investigating Multi-Aspect and Implicit Sentiment Analysis in Restaurant Reviews Using Transformer-Based Language Models**

The second title is preferable if implicit-aspect analysis becomes a major experimental component.

---

# 24. Proposed Research Hypotheses

The following hypotheses can guide the experiments:

**H1:** Transformer-based models will achieve higher F1-scores than traditional TF-IDF-based machine-learning approaches for restaurant aspect detection.

**H2:** Aspect and sentiment classification performance will decrease as the number of aspects expressed within a single text increases.

**H3:** Implicit aspects will be more difficult to identify than explicitly mentioned aspects.

**H4:** Transformer-based models will exhibit a smaller performance decrease between explicit and implicit aspects than lexical machine-learning baselines.

These hypotheses are measurable and can be accepted or rejected using the experimental results.

---

# 25. Final Project Concept

At its simplest, the complete research project can be represented as:

**INPUT**

Restaurant review:

> "Fantastic pizza and a lovely atmosphere, but the waiter took forever and the meal was overpriced."

**SYSTEM**

Multi-label aspect detection

↓

Food ✓  
Service ✓  
Ambience ✓  
Price/Value ✓

↓

Aspect-conditioned sentiment classification

↓

**OUTPUT**

Food → Positive  
Ambience → Positive  
Service → Negative  
Price/Value → Negative

The research then asks not merely whether this system can be built, but **how reliably different NLP approaches can perform this task, what happens as reviews become genuinely multi-aspect, whether implicit aspects can be recognised, and where current approaches fail.**

This transforms the work from a simple sentiment-classification application into a focused experimental study of Aspect-Based Sentiment Analysis suitable for a two-semester undergraduate final-year project.