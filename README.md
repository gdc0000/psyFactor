# Factor and Dimensionality Reduction Explorer

## Overview

The **Factor and Dimensionality Reduction Explorer** is an interactive Streamlit application designed for social scientists and students with limited statistical background. The app integrates advanced statistical methods with clear, educational explanations to help users perform:

- **Reliability Analysis**
- **Principal Component Analysis (PCA)**
- **Exploratory Factor Analysis (EFA)**

By simplifying complex datasets through dimensionality reduction and uncovering latent constructs, the tool supports rigorous yet accessible data analysis.

## Features

### Reliability Analysis
- **Cronbach’s α:** Estimates internal consistency of items.
- **McDonald’s ω:** Provides a nuanced reliability estimate considering the factor structure.
- **Item-Rest Correlations:** Examines how well each item correlates with the overall scale.
- **Alpha if Item Dropped:** Shows the impact on reliability when an item is removed.
- **Composite Score Computation:** Option to compute mean or sum scores with a preview and manual appending.

### Principal Component Analysis (PCA)
- **Extraction Methods:**
  - **Eigenvalue > 1:** Retains components with eigenvalues greater than 1.
  - **Fixed Number:** Manually specify the number of components.
  - **Parallel Analysis:** Compares observed eigenvalues with those from random data.
- **Rotation Options:**
  - **None:** No rotation applied.
  - **Varimax:** Orthogonal rotation that simplifies loadings and keeps components uncorrelated.
  - **Oblimin:** Oblique rotation that allows components to correlate, which can be more realistic in social science data.
- **Visualizations:** Scree plot and component loadings table.
- **Score Appending:** Option to manually append PCA scores to your dataset.

### Exploratory Factor Analysis (EFA)
- **Extraction Methods:** Similar options as PCA – Eigenvalue > 1, Fixed Number, or Parallel Analysis.
- **Rotation Options:**
  - **None:** No rotation applied.
  - **Varimax:** Orthogonal rotation keeping factors uncorrelated.
  - **Oblimin:** Oblique rotation allowing factors to correlate.
  - **Promax:** A faster approximate oblique rotation.
- **Visualizations:** Scree plot, factor loadings, and factor correlations.
- **Score Appending:** Option to manually append EFA factor scores to your dataset.

### Sample Datasets
Use preloaded datasets from `sklearn.datasets` including:
- Wine
- Iris (Numeric Only)
- Diabetes
- California Housing

### Data Upload
Supports uploading your own data in CSV, Excel, or SPSS (.sav) formats. Only numeric columns are used for analysis.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the application using Streamlit:
```bash
streamlit run main.py
```
Then, open the provided local URL in your web browser to interact with the app.

## Educational Components

- **In-App Explanations:** Expandable sections provide detailed, plain language explanations of statistical methods, including differences among extraction methods and rotation options.
- **Tooltips and Examples:** Each selectable option comes with a clear explanation to help you understand its purpose and when to use it.
- **Interactive Visualizations:** Scree plots, correlation heatmaps, and loadings tables guide your interpretation of the results.

## License

This project is licensed under the MIT License.

## Author

**Gabriele Di Cicco, PhD in Social Psychology**  
[ORCID](https://orcid.org/0000-0002-1439-5790) | [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)

## Acknowledgements

Thank you to the open-source community for providing the libraries and tools that made this project possible.
```
