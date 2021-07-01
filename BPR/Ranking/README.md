# Ranking Algorithms

In this directory you can find all the code needed to implement the different ranking algorithms. 
In particular, we implement the BPR algorithm using two different underlying models:
-  Matrix Factorization ([MF][MF]): These techniques generally learn a low-dimensional representation of users and items by mapping them into a joint latent space consisting of latent factors. Recommendations are then generated based on the similarity of user and item factors.
- Factorization Machines ([FM][FM]): This algorithm is a general factorization model that not only learns user and item latent factors, but also the relation between users and items with any auxiliary features. This is done by also factorizing these features to the same joint latent space.  This creates a great flexibility by allowing the algorithm to incorporate any additional information in terms of these auxiliary features.

Also, you can find a notebook which takes the categorized data from the [Simplified-OC][S-OC] and processes it to be used in the ranking algorithm.

[MF]: https://github.com/alexmartinezmiguel/TFM-Travels-Offers-Classification/tree/main/BPR/Ranking/MF
[FM]: https://github.com/alexmartinezmiguel/TFM-Travels-Offers-Classification/tree/main/BPR/Ranking/FM
[S-OC]: https://github.com/alexmartinezmiguel/TFM-Travels-Offers-Classification/tree/main/BPR/Simplified-OC
