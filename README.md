# 🔮 Interactive Word Prediction Graph using LLMs

This project visualizes **next-word predictions** from a language model (LLM) like `distilgpt2` as a dynamically expanding **graph tree**.

Originally developed for the workshop **"AI & LLMs: From Concepts to Code"**, this tool helps students and developers explore how language models make predictions in a transparent and interactive way.

---

## 🚀 Features

- 🌐 Web UI to explore predictions word-by-word
- 🌳 Tree-structured graph with word tokens as nodes and prediction probabilities as edge weights
- 🔁 Auto-generation mode to visualize branching sequences
- 📈 Real-time rendering with Plotly
- 🔍 Hover on a node to see full sequence, token ID, prediction probability, and structure info
- 🧠 Uses Hugging Face `transformers` for LLM inference (default: `distilgpt2`)

---

## 📸 Demo Preview

Avilable on LinkedIn | faran0321

---

## 🛠️ Tech Stack

| Layer     | Technology                       |
|-----------|----------------------------------|
| Frontend  | HTML, CSS, JavaScript, Plotly.js |
| Backend   | Flask (Python)                   |
| Model     | Hugging Face Transformers (`distilgpt2`) |
| Visualization | Plotly (Python + JS)         |
