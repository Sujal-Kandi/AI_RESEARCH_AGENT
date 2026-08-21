# GitHub Profile Fixes
## Instructions: Copy each section into the respective repo on GitHub

---

## 1. REPO CARD DESCRIPTIONS
*(Go to each repo → click the ⚙️ gear icon next to "About" → paste description)*

**AI_RESEARCH_AGENT**
```
Autonomous multi-agent research pipeline — plans queries, crawls the web, writes, self-critiques, rewrites weak sections, and exports a cited PDF report. Built with LangGraph + LangChain.
```

**Blog_Agent**
```
9-step autonomous blog writing agent — web research, fact extraction, self-critique, cliché detection, SEO metadata, and quality scoring. Built with LangChain + FastAPI.
```

**PhonePricePredictionProject**
```
End-to-end phone price prediction on a real-world noisy dataset. XGBoost + CatBoost, 15+ features, ~84% accuracy. Deployed as a live Flask web app.
```

**FruitFreshnessPredictionProject**
```
Computer vision model (MobileNetV2) classifying fresh vs rotten fruits in real time. Deployed as a Flask web app on Railway.
```

**Machine-Learning-Projects-**
```
ML fundamentals — regression, classification, ANN, XGBoost, CatBoost across real-world datasets including car pricing, laptop pricing, and income classification.
```

---

## 2. README — Machine-Learning-Projects-
*(Go to repo → edit README.md → replace with this)*

```markdown
# Machine Learning Projects

A collection of ML projects covering regression, classification, and neural networks
on real-world datasets.

---

## Projects

### Car Price Prediction
Regression model to predict car prices based on features like brand, mileage, and year.
**Tech:** Python, Pandas, Scikit-learn

### House Price Prediction
Regression model analyzing feature impact on house prices with EDA and feature selection.
**Tech:** Python, Pandas, Scikit-learn, Matplotlib

### Laptop Price Prediction
Multi-model comparison on a messy real-world dataset.
Models tested: Linear Regression, Decision Tree, XGBoost
**Tech:** Python, XGBoost, Scikit-learn

### Income Classification (>50K / <50K)
Binary classification using ANN to predict income category from census data.
**Tech:** Python, TensorFlow/Keras, Pandas

### XGBoost Classifier
Explored XGBoost for classification — hyperparameter tuning and performance evaluation.
**Tech:** Python, XGBoost, Scikit-learn

---

## Tech Stack
Python · Pandas · NumPy · Matplotlib · Scikit-learn · XGBoost · TensorFlow/Keras
```

---

## 3. README — PhonePricePredictionProject
*(Go to repo → edit README.md → replace with this)*

```markdown
# Phone Price Prediction

End-to-end ML project predicting mobile phone prices from real-world industry data.
Focused on data understanding, feature engineering, and production deployment.

🔗 **[Live App](https://phonepricepredictionproject.onrender.com)**

---

## Problem
Predict the price range of a mobile phone from 15+ hardware and software features:
RAM, storage, battery, CPU specs, camera, connectivity, and more.

---

## Pipeline

```
Raw Noisy Data
      │
      ▼
Data Cleaning & EDA
      │
      ▼
Feature Engineering
      │
      ▼
Model Training (ANN → XGBoost → CatBoost)
      │
      ▼
Flask Web App → Deployed on Render
```

---

## Data Challenges
Real-world data is never clean. This project involved:
- Handling missing and inconsistent values with domain reasoning
- Fixing data type mismatches and noisy entries
- Feature selection to reduce overfitting without losing signal

---

## Models & Results

| Model | Notes |
|---|---|
| ANN | Underperformed on this tabular dataset |
| XGBoost | Strong baseline |
| CatBoost | Best generalization — **~84% accuracy** |

Achieved ~84% accuracy on noisy, real-world data without data leakage.

---

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · CatBoost · Flask · HTML

---

## Project Structure
```
├── app.py              # Flask web app
├── model.pkl           # Trained CatBoost model
├── requirements.txt
├── templates/          # HTML frontend
└── .gitignore
```
```

---

## 4. README — FruitFreshnessPredictionProject
*(Go to repo → edit README.md → replace with this)*

```markdown
# Fruit Freshness Detection

Computer vision model that classifies fruits as **Fresh** or **Rotten** in real time.
Built with MobileNetV2 and deployed as a Flask web app.

🔗 **[Live App](https://fruit-freshness-production.up.railway.app)**

---

## Supported Fruits
Apple · Banana · Orange

---

## Pipeline

```
Fruit Image Upload
      │
      ▼
Image Preprocessing (resize + normalize)
      │
      ▼
MobileNetV2 Classification
      │
      ▼
Fresh / Rotten → Displayed in UI
```

---

## Why MobileNetV2?
- Lightweight architecture — fast inference
- Strong performance on limited datasets
- Mobile-friendly — suitable for real-world edge deployment

---

## Real-World Use Cases
- Fruit shops — quick quality checks at point of sale
- Warehouses — reduce wastage before storage
- Food supply chains — automated inspection on conveyor belts

---

## Tech Stack
Python · TensorFlow/Keras · MobileNetV2 · Flask · HTML

---

## Project Structure
```
├── app.py              # Flask web app
├── model.h5            # Trained MobileNetV2 model
├── requirements.txt
├── templates/          # HTML frontend
└── static/             # CSS and assets
```
```

---

## 5. PROFILE BIO
*(Go to github.com/Bryan-eng-lng → Edit profile → Bio)*

```
AI Engineer | Agentic AI & LLM Systems | LangGraph · LangChain · RAG | Building autonomous agents that plan, research, and self-correct
```

---

## 6. PROFILE README (Optional but powerful)
*(Create a repo named exactly "Bryan-eng-lng" → add README.md → it shows on your profile page)*

```markdown
# Hi, I'm Sujal Kandi 👋

AI Engineer focused on Agentic AI and LLM systems.
I build autonomous multi-agent pipelines that plan, execute, self-critique, and self-correct.

## What I Build
- 🤖 **Agentic AI** — Multi-agent pipelines with LangGraph, self-critique loops, RAG
- 🧠 **NLP** — Transformers, BERT, sentiment analysis, embeddings
- 📊 **ML** — End-to-end models from noisy real-world data to deployed web apps
- 👁️ **Computer Vision** — MobileNetV2, real-time classification

## Featured Projects

| Project | What it does | Stack |
|---|---|---|
| [AI Research Agent](https://github.com/Bryan-eng-lng/AI_RESEARCH_AGENT) | Autonomous research pipeline → cited PDF report | LangGraph, LangChain, FastAPI |
| [Blog Writer Agent](https://github.com/Bryan-eng-lng/Blog_Agent) | 9-step blog writing pipeline with self-critique | LangChain, Groq, Tavily |
| [Smart Chatbot](https://github.com/Bryan-eng-lng/Smart-Chatbot) | RAG + web search with auto-routing | ChromaDB, Ollama, FastAPI |
| [Phone Price Prediction](https://github.com/Bryan-eng-lng/PhonePricePredictionProject) | ML on real-world noisy data, deployed | XGBoost, CatBoost, Flask |

## Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-000000?style=flat)
![LangGraph](https://img.shields.io/badge/LangGraph-000000?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat)

## Connect
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sujal-kandi-914974372/)
```
