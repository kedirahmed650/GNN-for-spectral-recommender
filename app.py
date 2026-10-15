# -*- coding: utf-8 -*-
"""
Movie Recommender Web App (Spectral Method, K=30) - Clean UI
双击运行后，在浏览器打开 http://127.0.0.1:5000
"""

import os
import sys
import numpy as np
from flask import Flask, render_template_string, request

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "Desktop"))
from movie_recommender import (
    load_ratings, load_movie_info, build_id_maps,
    movie_display_name, TRAIN_FILE, MOVIE_INFO_FILE
)
from scipy.sparse import diags
from scipy.linalg import eigh

app = Flask(__name__)

print("Loading data and computing spectral embeddings...")

ratings = load_ratings(TRAIN_FILE)
user_ids, item_ids, user_index, item_index = build_id_maps(ratings, [])
movie_info = load_movie_info(MOVIE_INFO_FILE)

num_users = len(user_ids)
num_items = len(item_ids)

user_watched = {u: [] for u in range(num_users)}
user_genre_counts = {}
for raw_u, raw_i, _ in ratings:
    u = user_index[raw_u]
    i = item_index[raw_i]
    user_watched[u].append(i)
    genres = movie_info.get(raw_i, {}).get("genres", [])
    if u not in user_genre_counts:
        user_genre_counts[u] = {}
    for g in genres:
        user_genre_counts[u][g] = user_genre_counts[u].get(g, 0) + 1

R = np.zeros((num_users, num_items))
for raw_u, raw_i, _ in ratings:
    R[user_index[raw_u], item_index[raw_i]] = 1

top = np.hstack([np.zeros((num_users, num_users)), R])
bottom = np.hstack([R.T, np.zeros((num_items, num_items))])
A = np.vstack([top, bottom])
deg = np.sum(A, axis=1)
D_inv_sqrt = diags(1 / np.sqrt(deg))
L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

eigvals, eigvecs = eigh(L, subset_by_index=(0, 29))
emb = eigvecs[:, :30]
user_emb = emb[:num_users, :]
item_emb = emb[num_users:, :]

scores = np.dot(user_emb, item_emb.T)
train_bool = R > 0
scores[train_bool] = -np.inf
top10_indices = np.argsort(-scores, axis=1)[:, :10]

print(f"Ready! {num_users} users, {num_items} movies loaded.")

# ========== 简洁 HTML 模板（所有小字已删除） ==========
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spectral Movie Recommender</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(145deg, #f6f9fc 0%, #e9f1f8 100%);
            min-height: 100vh;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            max-width: 1100px;
            width: 100%;
        }
        .header-card {
            background: linear-gradient(135deg, #1a2a6c 0%, #2d4373 40%, #4a6fa5 100%);
            border-radius: 24px;
            padding: 40px 48px;
            color: white;
            box-shadow: 0 20px 60px rgba(26, 42, 108, 0.30);
            margin-bottom: 32px;
            position: relative;
            overflow: hidden;
        }
        .header-card::before {
            content: "🎬";
            position: absolute;
            right: 30px;
            bottom: -10px;
            font-size: 120px;
            opacity: 0.12;
        }
        .header-card h1 {
            font-size: 32px;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 0;
        }
        /* 删除了 .subtitle */
        .search-card {
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 28px 32px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.06);
            margin-bottom: 28px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 16px;
            border: 1px solid rgba(255,255,255,0.6);
        }
        .search-card label {
            font-weight: 600;
            font-size: 15px;
            color: #1e293b;
        }
        .search-card input {
            flex: 1;
            min-width: 180px;
            padding: 12px 20px;
            border: 2px solid #e2e8f0;
            border-radius: 14px;
            font-size: 16px;
            font-weight: 500;
            transition: all 0.25s ease;
            background: white;
            color: #0f172a;
            outline: none;
        }
        .search-card input:focus {
            border-color: #4a6fa5;
            box-shadow: 0 0 0 4px rgba(74, 111, 165, 0.15);
        }
        .search-card button {
            padding: 12px 32px;
            background: linear-gradient(135deg, #1a2a6c, #4a6fa5);
            color: white;
            border: none;
            border-radius: 14px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.25s ease;
            box-shadow: 0 4px 16px rgba(26, 42, 108, 0.25);
            letter-spacing: 0.3px;
        }
        .search-card button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 28px rgba(26, 42, 108, 0.35);
        }
        /* 删除了 .hint */
        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 24px;
        }
        @media (max-width: 780px) {
            .results-grid {
                grid-template-columns: 1fr;
            }
        }
        .card {
            background: white;
            border-radius: 20px;
            padding: 24px 28px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.05);
            border: 1px solid rgba(255,255,255,0.5);
            backdrop-filter: blur(4px);
            transition: all 0.2s ease;
        }
        .card:hover {
            box-shadow: 0 12px 40px rgba(0,0,0,0.07);
        }
        .card-header {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 18px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid #f1f5f9;
        }
        .card-header .icon {
            font-size: 22px;
        }
        .card-header .count-badge {
            margin-left: auto;
            background: #f1f5f9;
            padding: 2px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            color: #475569;
        }
        .genre-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 14px;
        }
        .genre-tag {
            background: linear-gradient(135deg, #eef2ff, #e0e7ff);
            color: #3730a3;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }
        .genre-tag .count {
            background: rgba(55, 48, 163, 0.12);
            padding: 0 6px;
            border-radius: 10px;
            margin-left: 4px;
            font-weight: 700;
        }
        .movie-list {
            display: flex;
            flex-direction: column;
            gap: 4px;
            max-height: 420px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .movie-list::-webkit-scrollbar {
            width: 4px;
        }
        .movie-list::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 10px;
        }
        .movie-item {
            display: flex;
            align-items: center;
            padding: 10px 12px;
            border-radius: 12px;
            transition: all 0.2s ease;
            gap: 12px;
            background: #fafcff;
            border-bottom: 1px solid #f1f5f9;
        }
        .movie-item:hover {
            background: #f1f5f9;
            transform: translateX(4px);
        }
        .movie-item .rank {
            font-weight: 700;
            color: #94a3b8;
            font-size: 14px;
            min-width: 32px;
            text-align: center;
        }
        .movie-item .rank.top1 { color: #f59e0b; }
        .movie-item .rank.top2 { color: #94a3b8; }
        .movie-item .rank.top3 { color: #d97706; }
        .movie-item .info {
            flex: 1;
        }
        .movie-item .title {
            font-weight: 600;
            color: #0f172a;
            font-size: 15px;
        }
        .movie-item .genres {
            font-size: 12px;
            color: #94a3b8;
            margin-top: 2px;
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }
        .movie-item .genres span {
            background: #f1f5f9;
            padding: 0 8px;
            border-radius: 10px;
            font-size: 11px;
            color: #475569;
        }
        .movie-item.watched {
            background: #f8fafc;
            opacity: 0.85;
        }
        .movie-item.watched .title {
            font-weight: 500;
            color: #475569;
        }
        .empty-state {
            color: #94a3b8;
            font-size: 14px;
            padding: 24px 0;
            text-align: center;
        }
        /* 删除了 .footer */
        .error {
            color: #dc2626;
            font-weight: 600;
            padding: 8px 16px;
            background: #fef2f2;
            border-radius: 12px;
            display: inline-block;
        }
        /* 删除了卡片底部小字相关样式 */
    </style>
</head>
<body>
<div class="container">
    <!-- Header - 只有标题 -->
    <div class="header-card">
        <h1>🎬 Spectral Recommender</h1>
    </div>

    <!-- Search - 删除了 hint -->
    <form method="POST" class="search-card">
        <label for="user_id">👤 User ID</label>
        <input type="number" id="user_id" name="user_id" placeholder="e.g. 386" value="{{ user_id if user_id is not none else '' }}" min="0" max="{{ num_users - 1 }}" required>
        <button type="submit">🔍 Recommend</button>
    </form>

    {% if error %}
        <div class="error">⚠️ {{ error }}</div>
    {% endif %}

    {% if user_id is not none %}
    <div class="results-grid">
        <div class="card">
            <div class="card-header">
                <span class="icon">👤</span> User {{ user_id }}
                <span class="count-badge">{{ watched_count }} ratings</span>
            </div>

            {% if top_genres %}
            <div style="margin-bottom: 16px;">
                <div style="font-size:13px; font-weight:600; color:#64748b; margin-bottom:6px;">🎯 Top Genres</div>
                <div class="genre-tags">
                    {% for g, count in top_genres %}
                        <span class="genre-tag">{{ g }} <span class="count">{{ count }}</span></span>
                    {% endfor %}
                </div>
            </div>
            {% endif %}

            <div class="card-header" style="border-bottom-color:#f1f5f9; margin-bottom:10px; font-size:15px;">
                <span class="icon">📚</span> Watched History
                <span class="count-badge">{{ watched|length }} shown</span>
            </div>

            <div class="movie-list">
                {% if watched %}
                    {% for item in watched %}
                    <div class="movie-item watched">
                        <span class="rank">#{{ loop.index }}</span>
                        <div class="info">
                            <div class="title">{{ item.title }}</div>
                            <div class="genres">
                                {% for g in item.genres.split(', ') if g %}
                                    <span>{{ g }}</span>
                                {% endfor %}
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="empty-state">No training history for this user.</div>
                {% endif %}
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <span class="icon">🎯</span> Top 10 Recommendations
                <span class="count-badge">spectral K=30</span>
            </div>

            <div class="movie-list">
                {% if recommendations %}
                    {% for item in recommendations %}
                    <div class="movie-item">
                        <span class="rank {% if loop.index == 1 %}top1{% elif loop.index == 2 %}top2{% elif loop.index == 3 %}top3{% endif %}">#{{ loop.index }}</span>
                        <div class="info">
                            <div class="title">{{ item.title }}</div>
                            <div class="genres">
                                {% for g in item.genres.split(', ') if g %}
                                    <span>{{ g }}</span>
                                {% endfor %}
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="empty-state">No recommendations available.</div>
                {% endif %}
            </div>
            <!-- 删除了卡片底部小字 -->
        </div>
    </div>
    {% endif %}

    <!-- 删除了 footer -->
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    user_id = None
    recommendations = None
    watched = []
    top_genres = []
    watched_count = 0
    error = None

    if request.method == "POST":
        try:
            user_id = int(request.form.get("user_id", ""))
            if user_id < 0 or user_id >= num_users:
                error = f"User ID must be between 0 and {num_users - 1}."
            else:
                watched_indices = user_watched.get(user_id, [])
                watched = []
                for idx in watched_indices[:20]:
                    movie_id = item_ids[idx]
                    title = movie_display_name(movie_id, movie_info)
                    genres = ", ".join(movie_info.get(movie_id, {}).get("genres", []))
                    watched.append({"title": title, "genres": genres})
                watched_count = len(watched_indices)

                counts = user_genre_counts.get(user_id, {})
                top_genres = sorted(counts.items(), key=lambda x: -x[1])[:5]

                top_item_indices = top10_indices[user_id]
                recommendations = []
                for idx in top_item_indices:
                    movie_id = item_ids[idx]
                    title = movie_display_name(movie_id, movie_info)
                    genres = ", ".join(movie_info.get(movie_id, {}).get("genres", []))
                    recommendations.append({"title": title, "genres": genres})

        except ValueError:
            error = "Please enter a valid number."

    return render_template_string(
        HTML_TEMPLATE,
        num_users=num_users,
        user_id=user_id,
        recommendations=recommendations,
        watched=watched,
        watched_count=watched_count,
        top_genres=top_genres,
        error=error
    )

if __name__ == "__main__":
    print("Starting web server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)