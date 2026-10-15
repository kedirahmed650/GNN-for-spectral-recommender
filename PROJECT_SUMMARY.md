# Project Summary

## What I Did

I implemented a spectral collaborative filtering recommender on the MovieLens 100K dataset. The system downloads the data automatically, builds a user-item bipartite graph, and computes the normalized Laplacian. I calculated the 50 smallest eigenvalues and eigenvectors with `scipy.linalg.eigh`, then used the resulting spectral embeddings to generate recommendations at K = 5, 10, 20, 30, and 50. I evaluated every setting with Recall@10 and NDCG@10.

To make the results interpretable, I added three reference points. The popularity baseline recommends the most frequently rated movies. The SVD baseline applies `TruncatedSVD` to the user-item matrix at the same five dimensions. I also analyzed the Fiedler vector, loaded real movie titles and genres from `u.item`, generated recommendation cards for representative Action and Romance fans, visualized the spectral embedding space, and compared both matrix factorization methods under cold-start conditions.

## What I Found

The spectral method reached its best Recall@10 at K = 30 (0.111) and its best NDCG@10 at K = 30 (0.291). Performance rose from K = 5 through K = 30 and then fell at K = 50, forming an inverted U-shaped curve that supports the hypothesis that only the low-frequency spectral signals carry most of the recommendation structure. The popularity baseline scored 0.097 on Recall@10, and SVD with d = 30 scored 0.185. SVD achieved higher absolute accuracy on the full test set.

The more interesting result came from cold-start evaluation. For users with fewer than 10 training ratings, the spectral method retained 86.3% of its full-data Recall@10, while SVD retained 82.8%. This indicates that spectral embeddings degrade more slowly when user history is sparse. The Fiedler vector also produced an interpretable split: the positive extreme included Titanic, while the negative extreme included classics such as Terminator and Back to the Future, suggesting that the spectral structure automatically captures genre-like divisions in the movie space.

## Conclusion

Spectral collaborative filtering works well as a graph-based recommender and is especially valuable in cold-start scenarios. The combination of spectral embeddings, SVD, and popularity baselines gives a clear picture of trade-offs between accuracy, explainability, and robustness to sparse user data.
