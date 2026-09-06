# Project Proposal

## Multi-Aspect Sentiment Analysis of Restaurant Reviews

### 1. Introduction

Online restaurant reviews frequently discuss multiple aspects of an experience — food, service, ambience, price — within a single piece of text, often with different sentiment toward each. Traditional sentiment analysis assigns one label per review and loses this information. For example:

> "The pasta was excellent, but the service was extremely slow."

A single-label classifier might output "Mixed" or "Negative," discarding the fact that Food is Positive and Service is Negative. Aspect-Based Sentiment Analysis (ABSA) addresses this by detecting the aspects present in a review and predicting sentiment for each one independently. This project proposes building and evaluating such a system for restaurant reviews.

### 2. Problem Statement

The system must solve two related sub-problems:

1. **Aspect Detection** — identify which aspects (e.g., Food, Service, Ambience, Price) are discussed in a review. Since a review can discuss several aspects at once, this is a **multi-label classification problem**.
2. **Aspect-Level Sentiment Classification** — determine the sentiment (Positive / Negative / Neutral) expressed toward each detected aspect.

A key challenge is that aspects are often **implicit** — expressed without naming the aspect directly:

> "We waited 45 minutes before anyone took our order." → Service: Negative

Detecting such cases requires contextual understanding rather than keyword matching, and is a central focus of this project.

### 3. Aim and Objectives

**Aim:** Design, implement, and evaluate a multi-aspect sentiment analysis system for restaurant reviews, with particular attention to implicit aspect detection and multi-aspect reviews.

**Objectives:**
1. Review existing ABSA literature and methods.
2. Define a restaurant aspect taxonomy.
3. **Construct a labeled dataset** using LLM-assisted annotation, human-verified for quality.
4. Build baseline models for aspect detection and sentiment classification.
5. Build transformer-based models and compare against baselines.
6. Evaluate performance on multi-aspect and implicit-aspect cases specifically.
7. Perform error analysis and build a demonstration prototype.

### 4. Dataset Construction

Rather than relying solely on existing benchmark data, this project will construct a custom annotated dataset:

- Collect raw restaurant review text.
- Use an LLM to generate **silver-standard** aspect and sentiment labels at scale.
- Manually verify and correct a sampled subset to create a smaller **gold-standard** test set, and to measure LLM labeling accuracy.
- Document annotation guidelines to ensure consistency, particularly around implicit-aspect cases.

This approach allows more control over aspect coverage (including implicit examples) than typical benchmark datasets provide, and gives the project a genuine data-engineering component alongside the modeling work.

### 5. Proposed System

```
Review → Preprocessing → Multi-Label Aspect Detection → Aspect-Level Sentiment Classification → (Aspect, Sentiment) pairs
```

Baselines will use TF-IDF with classical ML classifiers (e.g., Logistic Regression / SVM). The primary system will fine-tune a pretrained transformer (e.g., BERT/RoBERTa), with sentiment conditioned on a given aspect via input formatting. Evaluation will use precision/recall/F1 (micro and macro) for aspect detection, and accuracy/macro-F1 for sentiment classification, with separate breakdowns for single- vs. multi-aspect and explicit- vs. implicit-aspect cases.

### 6. Work Plan

**Semester 1 — Dataset and Baselines**
- Literature review and taxonomy definition
- Dataset construction: raw data collection, LLM-assisted labeling, human verification of gold subset
- Exploratory data analysis and data cleaning
- Baseline model construction (aspect detection + sentiment classification)

**Semester 2 — Advanced Modeling and Demonstration**
- Transformer-based model development and tuning
- LLM zero-shot and few-shot prompting experiments, compared against fine-tuned models
- Multi-aspect and implicit-aspect performance analysis
- End-to-end pipeline evaluation and error analysis
- Prototype dashboard: enter a review (or aggregate many reviews) and visualize aspect-level sentiment breakdown
