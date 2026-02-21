# Recommender System

Personal recommender system leveraging a manual algorithm to generate anime recommendations according to personal taste, leveraging my own ratings done on MyAnimeList over the years. This would make use of Jaccard index for genre set similarities and numerical comparison for ratings and other metadata. Vectorization is a potential option for cosine similarity comparison of synopses but would not be a priority

## Things to note:

- Given the single user situation, system will be a content-based filtering approach rather than a collaborative-based one
- Initially, the plan was to make use of ML models (feature vector embedding + logistic regression training), but similarly, the single user and the lack of proper GPU compute resources (Google Colab is never enough D:) makes it difficult to implement well
- Should there ever be a situation where I find multiple more users to justify it, a switch to a hybrid approach is possible; Should I randomly get a GPU upgrade, a deeper ML model may also be worked on
