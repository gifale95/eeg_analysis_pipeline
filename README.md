# EEG Analysis Pipeline

The present code provides a pipeline for the analysis of EEG data. The individual steps of this pipeline are:

  - **Preprocessing**
  - **Multivariate Noise Normalization**
  - **Event Related Potentials Analysis**
  - **Multivariate Pattern Analysis**


  
# Preprocessing

Preprocessing steps include:

 - Time-locking the data trials into epochs
 - Performing a baseline correction of the epochs
 - Downsampling the data to the desired frequency



# Multivariate Noise Normalization

Multivariate Noise Normalization ([Guggenmos et al., 2018][mvnn]) is a method aimed at downweighting EEG sensors with high noise levels and emphasizing sensors with low noise levels. Furthermore, it emphasizes or deemphasizes specific spatial frequencies of EEG patterns. This results in cleaner data for subsequent analyses.

[mvnn]: https://doi.org/10.1016/j.neuroimage.2018.02.044



# Event Related Potentials Analysis

Event Related Potentials ([Woodman, 2010][erps]) are obtained by averaging the time-locked EEG data across all its trials. This will results in the sensor-wise mean activity over time.

[erps]: https://link.springer.com/article/10.3758%2FBF03196680



# Multivariate Pattern Analysis

Multivariate Pattern Analysis ([Mur et al., 2009][mvpa]) is a method aimed at extracting condition-related information from the multidimensional sensor space. This analysis, performed independently at each time point, results in the Pairwise Dissimilarity Matrices ([Kriegeskorte et al., 2008][rdms]) of the experimental conditions.

[mvpa]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2656880/
[rdms]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2605405/