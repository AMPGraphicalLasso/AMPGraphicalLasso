# Sparse Estimation of Inverse Covariance Matrix using Approximate Message Passing (AMP)

This is the Github repository for hosting the codes and relevant materials for the ongoing publication work titled "Sparse Estimation of Inverse Covariance Matrix using Approximate Message Passing". The objective is to improve the Graphical Lasso Algorithm of Friedman, Hastie and Tibshirani based on the use of AMP algorithm and its variants to solve the lasso regression problems arising at the intermediate steps of the Graphicla Lasso Algorithm.

This is Python based implementation and borrows elements from the Python based implementations of the original Graphical lasso algorithm available at https://github.com/CamDavidsonPilon/Graphical-Lasso-in-Finance.git and https://github.com/takashi-takahashi/approximate_message_passing.git. The original repositories have been modified to reflect the changes incorporated in our proposed algorithms based on AMP and Vector AMP, and the modified repositories are available here. 

Although this is primarily a Python based implementation, the notebooks interchangeably use R and Python through the use of the R magic interface. This non-standard solution was necessary to make use of the highly optimized Rglasso package. We hope to come up with a more elegant solution for the final version of the code.
