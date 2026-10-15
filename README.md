# Spectral Collaborative Filtering for Movie Recommendations

This project applies spectral graph theory to collaborative filtering on the MovieLens 100K dataset. It builds a user-item bipartite graph, computes the normalized Laplacian, extracts the 50 smallest eigenvalues and eigenvectors, and evaluates spectral embeddings against SVD and popularity baselines.

## File Structure

| File | Description |
| --- | --- |
| `movie_recommender.py` | Main program: automatic data download, bipartite graph construction, normalized Laplacian, top-50 eigenvectors, K-value scan, SVD and popularity baselines, Fiedler vector analysis, cold-start evaluation, and all output generation. |
| `app.py` | Flask-based web recommender that lets a user enter a MovieLens user ID and view generated recommendations. |
| `spectral_result.png` | Recall@10 and NDCG@10 comparison curves for the spectral method, SVD, and popularity baseline. |
| `embedding_map.png` | 2D spectral movie embedding; red and blue points mark Fiedler positive and negative extremes, with representative movie labels. |
| `metrics_report.txt` | Full numeric report: K-value scan, baselines, Fiedler movie lists, and cold-start comparison. |
| `recommendation_cards.txt` | Top-5 recommendation cards for a representative Action fan and a Romance fan, including genres. |
| `README.md` | Project overview and setup instructions. |
| `SUBMISSION.md` | Submission file list and requirement checklist. |
| `PROJECT_SUMMARY.md` | One-page project summary for review or presentation. |

The `ml-100k/` directory is downloaded automatically on first run and should not be manually submitted.

## Requirements

- Python 3.x
- numpy
- scipy
- matplotlib
- scikit-learn
- flask

## How to Run

Install dependencies:

```bash
pip install numpy scipy matplotlib scikit-learn flask
```

Run the main experiment:

```bash
python movie_recommender.py
```

The script downloads MovieLens 100K if needed and writes the following files to the desktop:

- `spectral_result.png`
- `metrics_report.txt`
- `embedding_map.png`
- `recommendation_cards.txt`

Start the web interface:

```bash
python app.py
```

Open the URL printed in the terminal (the Flask default is `http://127.0.0.1:5000`), enter a user ID, and view the top recommendations.

## Core Results

| Method | Setting | Recall@10 | NDCG@10 |
| --- | --- | --- | --- |
| Spectral Method | K = 30 | 0.111 | 0.291 |
| SVD Baseline | d = 30 | 0.185 | 0.535 |
| Popularity Baseline | - | 0.097 | 0.325 |

Among spectral settings, K = 30 provides the best Recall@10. The SVD baseline achieves higher absolute metrics on the full test set, but the spectral method degrades more slowly under cold-start conditions: for users with fewer than 10 training ratings, spectral keeps 86.3% of its full-data Recall@10 while SVD keeps 82.8%.

## Notes

- No ratings, predictions, or code logic were changed during final packaging.
- `app.py` should be placed in the same submission folder as `movie_recommender.py` before final submission.
