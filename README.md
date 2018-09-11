# Semi-Supervised-Projected-Clustering
Semi-Supervised Projected Clustering according to 'On Discovery of Extremely Low-Dimensional Clusters using Semi-Supervised Projected Clustering' by Kevin Y. Yip et. al.

Dataset used: 

- https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

- Forina, M., Lanteri, S. Armanino, C., Casolino, C., Casale, M., Oliveri, P. (2008). V-PARVUS. An Extendible Package of programs for explorative data analysis, classification and regression analysis. Dip. Chimica e Tecnologie Farmaceutiche ed Alimentari, Università di Genova. Freely available at http://www.parvus.unige.it.

# Overview
SSPC algorithm aims to apply projected clustering with semi-supervised heuristics onto high-dimensional clustering data, to improve on current unsupervised clustering algorithms.

We will briefly discuss the algorithm as suggested in the paper and compare it with K-Means.

## Partial labels
The term semi-supervised refers to the case where we have partial labels for our datasets. SSPC use the partial labels to create 'seed groups' for each cluster, and draw medoids from these seed groups to act as the centroid as in K-Means.

## Projected clustering
For high-dimensional data clustering, it is often the case where only a subset of the features (dimensions) are relevant for a specific cluster, so we can create a mechanism (select_dim) to find those relevant features for a cluster, according to some metric (details in the paper).

Similarly, we can incorporate this idea into the objective function so that when looking for the 'optimal' clustering, we only consider the dimensions which are most relevant to a cluster.

# Complexity
'It can be shown that SSPC has a time and space complexity of O(knd) and O(nd) respectively.'

Our implementation: O(ckd * n^2) where c = convergence length

## select_dim
### calc_stats
Loop over all clusters, call calc_stats_i, which is O(nd) so total complexity: O(k * nd)
### find relevant dimensions
Loop over all dimensions to find relevant ones: O(d)
### overall
O(knd + d) = O(knd)

## Score functions
### score_function_ij
Calculate stats of j-th dimension: O(n)
### score_function_i
Loop over all selected dimensions, do score_function_ij: O(nd)
### score_function_all
Loop over clusters to find all scores, O(knd)

## initialize
### bin_cell(data)
O(size(data))
### hill_climb
Number of dimensions = building_dim_num

Grid search space volume ~ O((chosen dimensional values range / std / climb_step_size)^3) = O_hc
### private_seeds_for_labeled_objects
Find median -> select_dim -> score_function_i -> hill_climb: O(nd + knd + nd + O_hc) = O(max(knd, O_hc))

For simplicity, consider the case where O_hc < knd.
### get_peak
O(size(data)) ~ O(nd)
### max_min_dist
Loop over private seed groups and then private seeds within the group {ungrouped points {selected_dims}}: O(n * n * d) = O(d * n^2)
### overall
O(k * max(knd, n^2 * d)) = O(kd * n^2) (for n > k)

## fit_and_predict
### draw_medoids
Randomly permutate all seeds: O(nd)

### assign_max
Loop over all points {all clusters {calculate score_function_i}}: O(n * k * nd) = O(kd * n^2)

### replace_cluster_rep
Loop over all clusters {find the median of the cluster}: O(k * nd) = O(knd)

### overall
initialize -> draw_medoids -> loop till convergence {

  assign_max -> score_function_all -> replace_cluster_rep: O(kd * n^2 + knd + knd) = O(kd * n^2)

}: O(kd * n^2 + nd + c * kd * n^2) = O(ckd * n^2) 

where c = convergence length

Our implementation does not meet the complexity expectation from the original paper, so we will be looking for ways of improvement.
