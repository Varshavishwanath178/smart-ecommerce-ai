# 🛍️ Smart E-Commerce AI & Multi-Language Voice Assistant

A unified Python & Streamlit web application integrating three core Artificial Intelligence modules to enhance online shopping experience, accessibility, and trust.

---

## 📌 1. Problem Identification
* **Choice Overload:** Shoppers struggle to discover relevant items across large multi-category catalogs.
* **Language & Support Barriers:** Traditional text-only FAQs are slow and inaccessible to regional-language users.
* **Deceptive Reviews:** Bot-generated and spam reviews mislead buyers and reduce platform trust.

---

## 🧠 2. AI Concepts & Techniques Used
1. **Multi-Category Product Recommender:** Content-Based Filtering using **TF-IDF Vectorization** and **Cosine Similarity** across 20 diverse products (Electronics, Fashion, Home & Kitchen, Beauty, Sports, and Books).
2. **Multi-Language AI Voice & Text Chatbot:** Natural Language Processing (**NLP**) intent matching paired with **SpeechRecognition** (Speech-to-Text) and **gTTS** (Text-to-Speech) supporting English, Kannada, Hindi, Tamil, Telugu, and more.
3. **Fake Review & Spam Detector:** Supervised Machine Learning (**Logistic Regression** + TF-IDF features) trained to classify reviews as *Genuine* or *Fake/Suspicious* with confidence scores.

---

## ⚙️ 3. How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com
   cd smart-ecommerce-ai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
