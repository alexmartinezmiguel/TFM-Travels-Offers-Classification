# Simplified Offer Categorizer

In this directory you can find all the code for implementing the simplified version of the Offer Categorizer:
- To run the script you need to have a Redis Docker instance up and running.
- The main script is [categorizer.py][category_script], which reads the data from the cache, calls the necessary functions to compute 
  all the determinant factors, and finally aggregates the results to compute a score for each category of all offers.
- Inside the [categories][category_folder] folder, you can find all the functions needed to compute the categorizarion.

[category_script]: https://github.com/alexmartinezmiguel/TFM-Travels-Offers-Classification/blob/main/BPR/Simplified-OC/categorizer.py
[category_folder]: https://github.com/alexmartinezmiguel/TFM-Travels-Offers-Classification/tree/main/BPR/Simplified-OC/categories