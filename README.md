# 🎬 Movie Recommendation System using Collaborative Filtering  

A Machine Learning-based **Movie Recommendation System** built using **User Similarity and Collaborative Filtering**.  
The goal is to recommend movies to users based on the preferences and rating patterns of similar users.  

---

## 📂 Project Overview  

This project applies **user-based collaborative filtering** on the **MovieLens 100K dataset** to generate personalized movie recommendations.  
It identifies users with similar tastes using **cosine similarity** and suggests top-rated movies they haven’t seen yet.  

### **Key Steps:**  
- **Data Loading** – Imported `movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv`  
- **Data Merging** – Combined all datasets for a rich user-movie relationship matrix  
- **Data Cleaning** – Checked for nulls and handled missing data  
- **User-Item Matrix Creation** – Built a pivot table of user ratings for each movie  
- **Similarity Computation** – Used cosine similarity to find users with matching preferences  
- **Recommendation Logic** – Suggested top 5 unseen movies for a given user  
- **Evaluation Metric** – Used *Precision@K* to evaluate the quality of recommendations  

---

## 🛠️ Technical Implementation  

**Algorithm:** User-Based Collaborative Filtering  
**Similarity Metric:** Cosine Similarity  
**Libraries Used:** Pandas, NumPy, Scikit-learn, Matplotlib  
**Evaluation Metric:** Precision@K  
**Visualization:** Bar chart of top 5 predicted movies  

---

## 📊 Results  

- Successfully generated top 5 movie recommendations per user  
- Identified viewing patterns and similarity scores among 600+ users  
- Visualized predicted movie scores using Matplotlib  

**Example Output for User 6:**  
Top 5 Movie Recommendations with highest predicted ratings plotted on a bar graph.  

---

## 🎯 Real-World Application  

This system demonstrates how streaming platforms like **Netflix** or **Disney+** use collaborative filtering to:  
- Recommend movies based on user history  
- Improve user engagement and satisfaction  
- Personalize the viewing experience  

---

## 📁 Files  

- `movie_recommendation.py` – Main implementation code  
- `movies.csv`, `ratings.csv`, `tags.csv`, `links.csv` – Datasets from MovieLens  
- `README.md` – Project documentation  

---

## 🚀 How to Run  

1. **Install dependencies:**  
   ```bash
   pip install pandas numpy scikit-learn matplotlib
