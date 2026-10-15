# -*- coding: utf-8 -*-
"""
Spectral Movie Recommendation System (MovieLens 100K)

How to run: python <file>.py
Example: python movie_recommender.py

Features:
1. Automatically download MovieLens 100K data and extract it next to the script (skips if already downloaded).
2. Use u1.base as training set and u1.test as test set; build the user-item bipartite graph.
3. Compute the 50 smallest eigenvalues/eigenvectors of the normalized Laplacian.
4. Scan K = [5, 10, 20, 30, 50] and compute Recall@10 and NDCG@10.
5. Include a popularity baseline for comparison.
6. Add an SVD matrix factorization baseline using sklearn's TruncatedSVD.
7. Save spectral_result.png and metrics_report.txt to the Windows desktop.
8. Load real movie titles/genres from u.item and display them as 《Title (Year)》.
9. Generate 推荐示例.txt with Top-5 cards for Action and Romance fans.
10. Generate embedding_map.png and a cold-start comparison of spectral vs SVD.

If a required library is missing, run:
pip install numpy scipy matplotlib scikit-learn
Then run this script again.
"""

import os
import sys
import shutil
import zipfile
import urllib.request

# Make stdout safe for UTF-8 text on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Check that required third-party libraries are installed.
try:
    import numpy as np
    import scipy.sparse as sp
    from scipy.linalg import eigh
    import matplotlib

    matplotlib.use("Agg")  # Do not open a window; save the image directly
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD
except ImportError as exc:
    print("Missing third-party library:", exc)
    print("Please run: pip install numpy scipy matplotlib scikit-learn")
    print("Then run this script again.")
    try:
        if sys.stdin.isatty():
            input("Press Enter to exit...")
    except Exception:
        pass
    sys.exit(1)

# Dataset download URL and file locations
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_NAME = "ml-100k.zip"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "ml-100k")
TRAIN_FILE = os.path.join(DATA_DIR, "u1.base")
TEST_FILE = os.path.join(DATA_DIR, "u1.test")
MOVIE_INFO_FILE = os.path.join(DATA_DIR, "u.item")

# Desktop path (detect the current user's desktop instead of hardcoding a username)
DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
IMAGE_FILE = os.path.join(DESKTOP_DIR, "spectral_result.png")
REPORT_FILE = os.path.join(DESKTOP_DIR, "metrics_report.txt")
EMBEDDING_IMAGE_FILE = os.path.join(DESKTOP_DIR, "embedding_map.png")
RECOMMENDATION_FILE = os.path.join(DESKTOP_DIR, "推荐示例.txt")

# Experiment parameters
K_VALUES = [5, 10, 20, 30, 50]
TOP_N = 10
SVD_DIMENSIONS = [5, 10, 20, 30, 50]
SVD_REPORT_DIM = 30
SPECTRAL_EMBEDDING_K = 30
COLD_START_THRESHOLD = 5
COLD_LOW_THRESHOLD = 10
RECOMMEND_TOP_N = 5
MIN_FAN_RATINGS = 10
MIN_FAN_HITS = 3


def pause_if_interactive():
    """Wait for Enter when running interactively so the window does not close immediately."""
    try:
        if sys.stdin.isatty():
            input("Press Enter to exit...")
    except Exception:
        pass


def download_data(zip_path, ssl_context=None):
    """Download the data archive; ssl_context handles networks where certificate verification fails."""
    request = urllib.request.Request(
        DATA_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    open_kwargs = {"timeout": 120}
    if ssl_context is not None:
        open_kwargs["context"] = ssl_context
    with urllib.request.urlopen(request, **open_kwargs) as response, open(
        zip_path, "wb"
    ) as output_file:
        shutil.copyfileobj(response, output_file)


def ensure_dataset():
    """Check whether the data exists; download and extract it if it does not."""
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        print("MovieLens 100K data already exists; skipping download.")
        return

    zip_path = os.path.join(SCRIPT_DIR, ZIP_NAME)
    if not os.path.exists(zip_path):
        print("Downloading MovieLens 100K data...")
        print("Source:", DATA_URL)
        try:
            download_data(zip_path)
        except Exception as primary_error:
            print("Normal download failed:", primary_error)
            print("Trying compatibility mode (skipping certificate verification)...")
            try:
                import ssl
                download_data(zip_path, ssl._create_unverified_context())
            except Exception as exc:
                print("Download failed:", exc)
                print("Please check your network and run this script again.")
                pause_if_interactive()
                sys.exit(1)
        print("Download complete.")

    print("Extracting data...")
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(SCRIPT_DIR)
    print("Data extracted.")

    if not (os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE)):
        print("Cannot find u1.base / u1.test after extraction; please check the data files.")
        pause_if_interactive()
        sys.exit(1)


def load_ratings(file_path):
    """Read a u.data-style file: userID itemID rating timestamp (timestamp is ignored)."""
    ratings = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file_handle:
        for line in file_handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                user_id = int(parts[0])
                item_id = int(parts[1])
                rating = float(parts[2])
            except ValueError:
                continue
            ratings.append((user_id, item_id, rating))
    return ratings


def load_movie_info(file_path):
    """Load movie metadata (title, year, genres) from the latin-1 encoded u.item file."""
    movie_info = {}
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    if not os.path.exists(file_path):
        print("Warning: u.item not found, movie names will fall back to IDs.")
        return movie_info
    with open(file_path, "r", encoding="latin-1", errors="ignore") as file_handle:
        for line in file_handle:
            parts = line.strip().split("|")
            if len(parts) < 24:
                continue
            try:
                movie_id = int(parts[0])
            except ValueError:
                continue
            raw_title = parts[1].strip()
            year = None
            if raw_title.endswith(")"):
                start = raw_title.rfind("(")
                if start != -1:
                    candidate = raw_title[start + 1:-1].strip()
                    if len(candidate) == 4 and candidate.isdigit():
                        year = int(candidate)
                        raw_title = raw_title[:start].strip()
            genres = []
            for index, genre_name in enumerate(genre_names):
                if index + 5 < len(parts) and parts[index + 5] == "1":
                    genres.append(genre_name)
            movie_info[movie_id] = {
                "title": raw_title, "year": year, "genres": genres
            }
    return movie_info


def movie_display_name(movie_id, movie_info):
    """Return 《Title (Year)》 for reports; fall back to a plain movie ID if metadata is missing."""
    info = movie_info.get(movie_id)
    if info is None:
        return "Movie %d" % movie_id
    title = info.get("title", "Unknown")
    year = info.get("year")
    if year:
        return "《%s (%d)》" % (title, year)
    return "《%s》" % title


def movie_genres_text(movie_id, movie_info):
    """Return a comma-separated genre list for report tables."""
    genres = movie_info.get(movie_id, {}).get("genres", [])
    return ", ".join(genres) if genres else "Unknown"


def select_representative_users(train_ratings, movie_info):
    """Pick one Action fan and one Romance fan based on training genre concentration."""
    user_counts = {}
    for user_id, item_id, _ in train_ratings:
        counts = user_counts.setdefault(user_id, {"_total": 0})
        counts["_total"] += 1
        for genre in movie_info.get(item_id, {}).get("genres", []):
            counts[genre] = counts.get(genre, 0) + 1

    def preference_score(user_id, genre):
        counts = user_counts.get(user_id, {})
        total = counts.get("_total", 0)
        hits = counts.get(genre, 0)
        if total < MIN_FAN_RATINGS or hits < MIN_FAN_HITS:
            return -1.0
        return hits / float(total)

    all_user_ids = list(user_counts)
    if not all_user_ids:
        return None, None, user_counts
    action_user_id = max(all_user_ids, key=lambda uid: preference_score(uid, "Action"))
    romance_candidates = [
        uid for uid in all_user_ids if uid != action_user_id
    ]
    romance_user_id = max(
        romance_candidates, key=lambda uid: preference_score(uid, "Romance")
    ) if romance_candidates else None
    return action_user_id, romance_user_id, user_counts


def user_profile_sentence(user_id, user_counts):
    """Build a short profile string such as Action 62.5%, Thriller 12.5%. """
    counts = user_counts.get(user_id, {})
    total = counts.get("_total", 0)
    top_genres = sorted(
        [(genre, count) for genre, count in counts.items() if genre != "_total"],
        key=lambda item: -item[1]
    )[:3]
    if total == 0:
        return "User %d (ratings=%d): no training ratings" % (user_id, total)
    profile = ", ".join(
        "%s %.1f%%" % (genre, count * 100.0 / total)
        for genre, count in top_genres
    )
    return "User %d (ratings=%d): %s" % (user_id, total, profile)


def build_id_maps(train_ratings, test_ratings):
    """Map all observed user IDs and item IDs to zero-based matrix indices."""
    user_ids = sorted({user_id for user_id, _, _ in train_ratings} |
                      {user_id for user_id, _, _ in test_ratings})
    item_ids = sorted({item_id for _, item_id, _ in train_ratings} |
                      {item_id for _, item_id, _ in test_ratings})
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    return user_ids, item_ids, user_index, item_index


def build_graph_matrices(train_ratings, user_index, item_index):
    """Build adjacency matrix A, the training boolean matrix, and the normalized Laplacian of the bipartite graph."""
    num_users = len(user_index)
    num_items = len(item_index)

    rows = [user_index[user_id] for user_id, item_id, _ in train_ratings]
    cols = [item_index[item_id] for user_id, item_id, _ in train_ratings]
    weight = np.ones(len(train_ratings), dtype=float)

    # A is the user-item adjacency matrix: A[u, i] = 1 means user u rated item i
    adjacency = sp.coo_matrix(
        (weight, (rows, cols)), shape=(num_users, num_items)
    ).tocsr()

    train_bool = np.zeros((num_users, num_items), dtype=bool)
    for user_idx, item_idx in zip(rows, cols):
        train_bool[user_idx, item_idx] = True

    # The full bipartite adjacency matrix is [[0, A], [A.T, 0]],
    # so L = I - D^(-1/2) * B * D^(-1/2) is symmetric and can be used with scipy.linalg.eigh.
    bipartite_adjacency = sp.bmat(
        [[None, adjacency], [adjacency.T, None]], format="csr"
    )

    degrees = np.asarray(bipartite_adjacency.sum(axis=1)).ravel().astype(float)
    degrees[degrees == 0] = 1.0
    inv_sqrt_degree = sp.diags(1.0 / np.sqrt(degrees))

    lap_sparse = (
        sp.eye(bipartite_adjacency.shape[0], format="csr")
        - inv_sqrt_degree.dot(bipartite_adjacency).dot(inv_sqrt_degree)
    )
    lap_dense = lap_sparse.toarray()
    lap_dense = (lap_dense + lap_dense.T) * 0.5

    return adjacency, train_bool, lap_dense


def top_from_scores(score_matrix, train_bool):
    """Select the Top-N movies with the highest scores for each user, excluding training items."""
    scores = np.array(score_matrix, dtype=float)
    scores[train_bool] = -np.inf
    return np.argsort(-scores, axis=1)[:, :TOP_N]


def compute_top_matrix(user_emb, item_emb, train_bool):
    """Compute Top-N recommendations for all users from low-dimensional embeddings (already-seen items are excluded)."""
    scores = np.dot(user_emb, item_emb.T)
    return top_from_scores(scores, train_bool)


def compute_metrics(top_items, test_items):
    """Compute mean Recall@10 and mean NDCG@10."""
    recall_list = []
    ndcg_list = []
    for user_idx, test_set in test_items.items():
        if not test_set:
            continue
        top10 = top_items[user_idx, :TOP_N]
        hits = 0
        dcg = 0.0
        for rank, item_idx in enumerate(top10):
            if item_idx in test_set:
                hits += 1
                dcg += 1.0 / np.log2(rank + 2)
        recall_list.append(hits / len(test_set))
        ideal_dcg = sum(
            1.0 / np.log2(rank + 2) for rank in range(min(TOP_N, len(test_set)))
        )
        ndcg_list.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
    return float(np.mean(recall_list)), float(np.mean(ndcg_list))


def evaluate_spectral(k_value, eigvecs, num_users, train_bool, test_items):
    """Use the first k eigenvectors as a spectral embedding and evaluate Recall@10 / NDCG@10."""
    eigen_data = eigvecs[:, :k_value]
    user_emb = eigen_data[:num_users, :]
    item_emb = eigen_data[num_users:, :]

    user_norm = np.linalg.norm(user_emb, axis=1, keepdims=True)
    item_norm = np.linalg.norm(item_emb, axis=1, keepdims=True)
    user_norm[user_norm == 0] = 1.0
    item_norm[item_norm == 0] = 1.0
    user_emb = user_emb / user_norm
    item_emb = item_emb / item_norm

    top_items = compute_top_matrix(user_emb, item_emb, train_bool)
    return compute_metrics(top_items, test_items)


def evaluate_popularity_baseline(adjacency, num_users, train_bool, test_items):
    """Popularity baseline: recommend movies by the number of ratings in the training set."""
    item_popularity = np.asarray(adjacency.sum(axis=0)).ravel().astype(float)
    score_matrix = np.tile(item_popularity, (num_users, 1))
    top_items = top_from_scores(score_matrix, train_bool)
    return compute_metrics(top_items, test_items)


def evaluate_svd(adjacency, train_bool, test_items):
    """Use TruncatedSVD for matrix factorization and return metrics for d = [5, 10, 20, 30, 50]."""
    results = []
    for d_value in SVD_DIMENSIONS:
        svd = TruncatedSVD(n_components=d_value, random_state=42)
        user_emb = svd.fit_transform(adjacency)
        item_emb = svd.components_.T
        score_matrix = np.dot(user_emb, item_emb.T)
        top_items = top_from_scores(score_matrix, train_bool)
        recall, ndcg = compute_metrics(top_items, test_items)
        results.append((recall, ndcg))
    return results


def build_spectral_embeddings(eigvecs, num_users, k_value):
    """Build normalized spectral user/item embeddings for a given k."""
    eigen_data = eigvecs[:, :k_value]
    user_emb = eigen_data[:num_users, :]
    item_emb = eigen_data[num_users:, :]
    user_norm = np.linalg.norm(user_emb, axis=1, keepdims=True)
    item_norm = np.linalg.norm(item_emb, axis=1, keepdims=True)
    user_norm[user_norm == 0] = 1.0
    item_norm[item_norm == 0] = 1.0
    return user_emb / user_norm, item_emb / item_norm


def evaluate_cold_start(adjacency, eigvecs, num_users, train_bool,
                        train_items, test_items, full_spectral_recall,
                        full_svd_recall):
    """Compare spectral vs SVD on strict (<5) and low-activity (<10) cold-start users."""
    train_counts = [len(train_items[u]) for u in range(num_users)]

    def evaluate_group(user_indices):
        if not user_indices:
            return None
        cold_test_items = {u: test_items[u] for u in user_indices}
        spectral_recall, spectral_ndcg = evaluate_spectral(
            SPECTRAL_EMBEDDING_K, eigvecs, num_users, train_bool, cold_test_items
        )
        svd = TruncatedSVD(n_components=SVD_REPORT_DIM, random_state=42)
        user_emb = svd.fit_transform(adjacency)
        item_emb = svd.components_.T
        score_matrix = np.dot(user_emb, item_emb.T)
        top_items = top_from_scores(score_matrix, train_bool)
        svd_recall, svd_ndcg = compute_metrics(top_items, cold_test_items)
        spectral_retention = (
            spectral_recall / full_spectral_recall if full_spectral_recall > 0 else 0.0
        )
        svd_retention = (
            svd_recall / full_svd_recall if full_svd_recall > 0 else 0.0
        )
        return {
            "count": len(user_indices),
            "training_counts": sorted(train_counts[u] for u in user_indices),
            "spectral_recall": spectral_recall,
            "spectral_ndcg": spectral_ndcg,
            "svd_recall": svd_recall,
            "svd_ndcg": svd_ndcg,
            "spectral_retention": spectral_retention,
            "svd_retention": svd_retention,
        }

    strict_users = [
        u for u in range(num_users)
        if train_counts[u] < COLD_START_THRESHOLD and test_items[u]
    ]
    low_users = [
        u for u in range(num_users)
        if train_counts[u] < COLD_LOW_THRESHOLD and test_items[u]
    ]
    strict_results = evaluate_group(strict_users)
    low_results = evaluate_group(low_users)
    if strict_results is None and low_results is None:
        return None
    return {
        "strict": strict_results,
        "low": low_results,
    }


def build_recommendation_file_lines(user_emb, item_emb, train_bool, user_index,
                                    item_ids, movie_info, action_user_id,
                                    romance_user_id, user_counts):
    """Create a detailed text card for an Action fan and a Romance fan."""
    top_items = top_from_scores(np.dot(user_emb, item_emb.T), train_bool)
    lines = []
    lines.append("=" * 100)
    lines.append("Recommendation Examples for Two Typical Users (Spectral Method, K=30)")
    lines.append("=" * 100)
    lines.append("Model trained on u1.base; already-rated movies are excluded.")
    for fan_label, user_id in (("Action Fan", action_user_id),
                               ("Romance Fan", romance_user_id)):
        if user_id is None:
            continue
        lines.append("")
        lines.append(fan_label + " -- Top %d Recommendations" % RECOMMEND_TOP_N)
        lines.append(user_profile_sentence(user_id, user_counts))
        lines.append("-" * 100)
        lines.append("%-4s %-50s %-34s" % ("Rank", "Movie", "Genres"))
        lines.append("-" * 100)
        user_idx = user_index[user_id]
        for rank, item_idx in enumerate(top_items[user_idx][:RECOMMEND_TOP_N], start=1):
            movie_id = item_ids[item_idx]
            lines.append("%-4d %-50s %-34s" % (
                rank,
                movie_display_name(movie_id, movie_info),
                movie_genres_text(movie_id, movie_info)))
    lines.append("")
    lines.append("The action fan is selected by Action-genre concentration in training ratings,")
    lines.append("and the romance fan is selected by Romance-genre concentration. Genre labels come from u.item.")
    return lines


def find_movie_by_keyword(movie_info, keyword):
    """Return the first movie ID whose title contains keyword (case-insensitive)."""
    lower_keyword = keyword.lower()
    for movie_id, info in movie_info.items():
        if lower_keyword in info.get("title", "").lower():
            return movie_id
    return None


def build_embedding_map(eigvecs, num_users, item_ids, movie_info,
                        fiedler_high, fiedler_low, image_path):
    """Scatter all movies using the Fiedler axis plus the next eigenvector as 2D coordinates."""
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    num_items = len(item_ids)
    coordinates = eigvecs[num_users:, 1:3]
    x_values = coordinates[:, 0]
    y_values = coordinates[:, 1]
    positive_set = set(fiedler_high)
    negative_set = set(fiedler_low)
    neutral_indices = [
        i for i in range(num_items) if i not in positive_set and i not in negative_set
    ]
    positive_indices = list(positive_set)
    negative_indices = list(negative_set)
    try:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC"
        ]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(x_values[neutral_indices], y_values[neutral_indices],
               color="gray", s=16, alpha=0.65, label="Other Movies")
    ax.scatter(x_values[positive_indices], y_values[positive_indices],
               color="red", s=45, edgecolors="black", linewidths=0.5,
               label="Fiedler Positive Extreme (Top 10)")
    ax.scatter(x_values[negative_indices], y_values[negative_indices],
               color="blue", s=45, edgecolors="black", linewidths=0.5,
               label="Fiedler Negative Extreme (Top 10)")
    label_movie_ids = []
    for keyword in ("Star Wars", "Titanic", "Toy Story"):
        movie_id = find_movie_by_keyword(movie_info, keyword)
        if movie_id is not None and movie_id not in label_movie_ids:
            label_movie_ids.append(movie_id)
    for item_idx in positive_indices[:2] + negative_indices[:1]:
        movie_id = item_ids[item_idx]
        if len(label_movie_ids) < 3 and movie_id not in label_movie_ids:
            label_movie_ids.append(movie_id)
    x_range = float(np.max(x_values) - np.min(x_values))
    y_range = float(np.max(y_values) - np.min(y_values))
    for movie_id in label_movie_ids:
        item_idx = item_to_index.get(movie_id)
        if item_idx is None:
            continue
        label_y = y_values[item_idx] + (
            y_range * 0.04 if x_values[item_idx] >= 0 else -y_range * 0.04
        )
        ax.annotate(
            movie_display_name(movie_id, movie_info),
            xy=(x_values[item_idx], y_values[item_idx]),
            xytext=(x_values[item_idx], label_y),
            fontsize=10, ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8)
        )
    ax.set_title("Spectral Movie Embedding Space (K=30, Fiedler Projection)")
    ax.set_xlabel("Fiedler Coordinate (2nd Smallest Eigenvector)")
    ax.set_ylabel("Next Eigenvector Coordinate")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_report_lines(spectral_results, svd_results, svd_report_dim,
                       baseline_recall, baseline_ndcg,
                       train_count, test_count, eigvals, item_ids,
                       movie_fiedler, movie_info, cold_results):
    """Generate the numeric report text (table + Fiedler analysis)."""
    lines = []
    lines.append("=" * 60)
    lines.append("Experiment Report for Spectral Recommender (MovieLens 100K)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Training set: u1.base, Samples = %d" % train_count)
    lines.append("Test set: u1.test, Samples = %d" % test_count)
    lines.append("Metrics: Recall@10, NDCG@10")
    lines.append("")
    lines.append("-" * 60)
    lines.append("%-10s %-16s %-16s" % ("Method / K", "Recall@10", "NDCG@10"))
    lines.append("-" * 60)
    for k_value, (recall, ndcg) in zip(K_VALUES, spectral_results):
        lines.append("%-10d %-16.6f %-16.6f" % (k_value, recall, ndcg))
    lines.append("-" * 60)
    lines.append("%-10s %-16.6f %-16.6f" % (
        "Popularity Baseline", baseline_recall, baseline_ndcg))
    svd_index = SVD_DIMENSIONS.index(svd_report_dim)
    svd_recall, svd_ndcg = svd_results[svd_index]
    lines.append("%-10s %-16.6f %-16.6f" % (
        "SVD (d=%d)" % svd_report_dim, svd_recall, svd_ndcg))
    lines.append("-" * 60)
    lines.append("")
    lines.append("Smallest 50 eigenvalue range: %.8f to %.8f" % (
        eigvals[0], eigvals[-1]))
    lines.append("")
    lines.append("=" * 60)
    lines.append("Fiedler Vector Analysis (2nd smallest eigenvalue, index = 1)")
    lines.append("=" * 60)
    for label, indices in (("Positive Extreme", movie_fiedler["high"]),
                           ("Negative Extreme", movie_fiedler["low"])):
        lines.append("")
        lines.append(label + " Top %d Movies:" % TOP_N)
        for rank, (item_idx, score) in enumerate(indices, start=1):
            lines.append("%2d. %s Score = %+.8f" % (
                rank, movie_display_name(item_ids[item_idx], movie_info), score))
    lines.append("")
    lines.append("Note: the sign direction of the positive/negative extremes is determined by the Fiedler vector;")
    lines.append("only the highest and lowest scoring ends are shown here.")
    if cold_results is not None:
        lines.append("")
        lines.append("=" * 60)
        lines.append("Cold-Start Comparison (Users with Few Training Ratings)")
        lines.append("=" * 60)
        strict_cold = cold_results.get("strict")
        low_cold = cold_results.get("low")
        if strict_cold is not None:
            lines.append("")
            lines.append("Strict cold users (< %d training ratings)" % COLD_START_THRESHOLD)
            lines.append("Cold users with test items: %d" % strict_cold["count"])
            lines.append("Spectral Method (K=%d) Recall@10 = %.6f" % (
                SPECTRAL_EMBEDDING_K, strict_cold["spectral_recall"]))
            lines.append("SVD (d=%d) Recall@10 = %.6f" % (
                SVD_REPORT_DIM, strict_cold["svd_recall"]))
            lines.append("Spectral Method (K=%d) NDCG@10 = %.6f" % (
                SPECTRAL_EMBEDDING_K, strict_cold["spectral_ndcg"]))
            lines.append("SVD (d=%d) NDCG@10 = %.6f" % (
                SVD_REPORT_DIM, strict_cold["svd_ndcg"]))
            if strict_cold["count"] == 1:
                lines.append("Note: only one strict user exists, so this single-user result is highly noisy;")
                lines.append("use the low-activity group below for a more stable comparison.")
        if low_cold is not None:
            lines.append("")
            lines.append("Low-activity group (< %d training ratings)" % COLD_LOW_THRESHOLD)
            lines.append("Cold users with test items: %d" % low_cold["count"])
            lines.append("Spectral Method (K=%d) Recall@10 = %.6f" % (
                SPECTRAL_EMBEDDING_K, low_cold["spectral_recall"]))
            lines.append("SVD (d=%d) Recall@10 = %.6f" % (
                SVD_REPORT_DIM, low_cold["svd_recall"]))
            lines.append("Spectral relative retention of full-data Recall@10: %.1f%%" % (
                low_cold["spectral_retention"] * 100.0))
            lines.append("SVD relative retention of full-data Recall@10: %.1f%%" % (
                low_cold["svd_retention"] * 100.0))
            lines.append("Higher relative retention means slower degradation on cold users.")
    return lines


def save_plot(k_values, spectral_results, svd_dimensions, svd_results,
              baseline_recall, baseline_ndcg, image_path):
    """Draw and save the comparison figure (Recall on the left, NDCG on the right)."""
    try:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC"
        ]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    recall_values = [item[0] for item in spectral_results]
    ndcg_values = [item[1] for item in spectral_results]
    svd_recall_values = [item[0] for item in svd_results]
    svd_ndcg_values = [item[1] for item in svd_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(k_values, recall_values, "o-", color="#1f77b4",
                 label="Spectral Method")
    axes[0].axhline(baseline_recall, color="#d62728", linestyle="--",
                    label="Popularity Baseline")
    axes[0].plot(svd_dimensions, svd_recall_values, "s-", color="#2ca02c",
                 label="SVD")
    axes[0].set_xlabel("Dimension (K / d)")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_title("Recall@10 Comparison")
    axes[0].set_xticks(k_values)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_values, ndcg_values, "o-", color="#1f77b4",
                 label="Spectral Method")
    axes[1].axhline(baseline_ndcg, color="#d62728", linestyle="--",
                    label="Popularity Baseline")
    axes[1].plot(svd_dimensions, svd_ndcg_values, "s-", color="#2ca02c",
                 label="SVD")
    axes[1].set_xlabel("Dimension (K / d)")
    axes[1].set_ylabel("NDCG@10")
    axes[1].set_title("NDCG@10 Comparison")
    axes[1].set_xticks(k_values)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Spectral Method vs SVD vs Popularity Baseline", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("Spectral Movie Recommendation System (MovieLens 100K)")
    print("=" * 60)

    os.makedirs(DESKTOP_DIR, exist_ok=True)

    ensure_dataset()

    print("Reading u1.base (training set) and u1.test (test set)...")
    train_ratings = load_ratings(TRAIN_FILE)
    test_ratings = load_ratings(TEST_FILE)

    print("Loading movie information from u.item...")
    movie_info = load_movie_info(MOVIE_INFO_FILE)
    print("Training samples: %d, Test samples: %d" % (len(train_ratings),
                                                     len(test_ratings)))

    user_ids, item_ids, user_index, item_index = build_id_maps(
        train_ratings, test_ratings
    )
    num_users = len(user_ids)
    num_items = len(item_ids)
    print("Users: %d, Movies: %d" % (num_users, num_items))

    print("Building the user-item bipartite graph adjacency matrix...")
    adjacency, train_bool, lap_dense = build_graph_matrices(
        train_ratings, user_index, item_index
    )

    train_items = {idx: set() for idx in range(num_users)}
    test_items = {idx: set() for idx in range(num_users)}
    for user_id, item_id, _ in train_ratings:
        train_items[user_index[user_id]].add(item_index[item_id])
    for user_id, item_id, _ in test_ratings:
        test_items[user_index[user_id]].add(item_index[item_id])

    print("Computing the 50 smallest eigenvalues/eigenvectors of the normalized Laplacian...")
    print("This may take a few minutes; please wait...")
    try:
        eigvals, eigvecs = eigh(lap_dense, subset_by_index=(0, 49))
    except TypeError:
        # Compatibility with older scipy parameter names
        eigvals, eigvecs = eigh(lap_dense, eigvals=(0, 49))
    print("Eigenvalue computation complete; smallest eigenvalue = %.8f" % eigvals[0])

    spectral_results = []
    for k_value in K_VALUES:
        print("Computing Recall@10 and NDCG@10 for K = %d ..." % k_value)
        recall, ndcg = evaluate_spectral(
            k_value, eigvecs, num_users, train_bool, test_items
        )
        spectral_results.append((recall, ndcg))
        print("K = %d  ->  Recall@10 = %.6f, NDCG@10 = %.6f" % (
            k_value, recall, ndcg))

    print("Computing popularity baseline (sorted by total rating count)...")
    baseline_recall, baseline_ndcg = evaluate_popularity_baseline(
        adjacency, num_users, train_bool, test_items
    )
    print("Popularity Baseline -> Recall@10 = %.6f, NDCG@10 = %.6f" % (
        baseline_recall, baseline_ndcg))

    print("Computing SVD baseline (TruncatedSVD)...")
    svd_results = evaluate_svd(adjacency, train_bool, test_items)
    for d_value, (recall, ndcg) in zip(SVD_DIMENSIONS, svd_results):
        print("SVD d = %d  ->  Recall@10 = %.6f, NDCG@10 = %.6f" % (
            d_value, recall, ndcg))
    svd_best_index = SVD_DIMENSIONS.index(SVD_REPORT_DIM)
    svd_best_recall, svd_best_ndcg = svd_results[svd_best_index]
    print("SVD report row (d = %d) -> Recall@10 = %.6f, NDCG@10 = %.6f" % (
        SVD_REPORT_DIM, svd_best_recall, svd_best_ndcg))

    print("Building K=%d spectral embeddings..." % SPECTRAL_EMBEDDING_K)
    user_emb_k30, item_emb_k30 = build_spectral_embeddings(
        eigvecs, num_users, SPECTRAL_EMBEDDING_K
    )

    print("Evaluating cold-start users (fewer than %d training ratings)..." % COLD_START_THRESHOLD)
    full_spectral_recall = spectral_results[
        K_VALUES.index(SPECTRAL_EMBEDDING_K)][0]
    full_svd_recall = svd_results[
        SVD_DIMENSIONS.index(SVD_REPORT_DIM)][0]
    cold_results = evaluate_cold_start(
        adjacency, eigvecs, num_users, train_bool, train_items, test_items,
        full_spectral_recall, full_svd_recall
    )
    if cold_results is not None:
        strict_cold = cold_results.get("strict")
        low_cold = cold_results.get("low")
        if strict_cold is not None:
            print("Strict cold users (<%d training ratings) with test items: %d" % (
                COLD_START_THRESHOLD, strict_cold["count"]))
            print("Spectral strict cold Recall@10 = %.6f, SVD = %.6f" % (
                strict_cold["spectral_recall"], strict_cold["svd_recall"]))
        if low_cold is not None:
            print("Low-activity users (<%d training ratings): %d" % (
                COLD_LOW_THRESHOLD, low_cold["count"]))
            print("Spectral retains %.1f%% of full Recall@10, SVD retains %.1f%%" % (
                low_cold["spectral_retention"] * 100.0,
                low_cold["svd_retention"] * 100.0))

    print("Running Fiedler vector analysis (bonus)...")
    fiedler_vector = eigvecs[:, 1]
    movie_part = fiedler_vector[num_users:]
    sorted_indices = np.argsort(movie_part)
    high_indices = sorted_indices[-TOP_N:][::-1]
    low_indices = sorted_indices[:TOP_N]
    movie_fiedler = {
        "high": [(idx, movie_part[idx]) for idx in high_indices],
        "low": [(idx, movie_part[idx]) for idx in low_indices],
    }

    print("Selecting representative Action/Romance fans...")
    action_user_id, romance_user_id, user_counts = select_representative_users(
        train_ratings, movie_info
    )
    print("Action fan user: %s | Romance fan user: %s" % (
        action_user_id, romance_user_id))
    print("Generating 推荐示例.txt ...")
    recommendation_lines = build_recommendation_file_lines(
        user_emb_k30, item_emb_k30, train_bool, user_index, item_ids,
        movie_info, action_user_id, romance_user_id, user_counts
    )
    with open(RECOMMENDATION_FILE, "w", encoding="utf-8-sig") as recommendation_file:
        recommendation_file.write("\n".join(recommendation_lines) + "\n")
    print("Recommendation examples saved to:", RECOMMENDATION_FILE)

    print("Generating metrics_report.txt ...")
    report_lines = build_report_lines(
        spectral_results, svd_results, SVD_REPORT_DIM,
        baseline_recall, baseline_ndcg,
        len(train_ratings), len(test_ratings), eigvals, item_ids,
        movie_fiedler, movie_info, cold_results
    )
    with open(REPORT_FILE, "w", encoding="utf-8-sig") as report_file:
        report_file.write("\n".join(report_lines) + "\n")
    print("Report saved to:", REPORT_FILE)

    print("Rendering spectral_result.png ...")
    save_plot(
        K_VALUES, spectral_results, SVD_DIMENSIONS, svd_results,
        baseline_recall, baseline_ndcg, IMAGE_FILE
    )
    print("Image saved to:", IMAGE_FILE)

    print("Rendering embedding_map.png ...")
    build_embedding_map(
        eigvecs, num_users, item_ids, movie_info,
        high_indices, low_indices, EMBEDDING_IMAGE_FILE
    )
    print("Embedding map saved to:", EMBEDDING_IMAGE_FILE)

    print("")
    print("All done! Desktop files saved:")
    print(" -", IMAGE_FILE)
    print(" -", REPORT_FILE)
    print(" -", EMBEDDING_IMAGE_FILE)
    print(" -", RECOMMENDATION_FILE)
    print("")
    pause_if_interactive()
    sys.exit(0)


if __name__ == "__main__":
    main()
