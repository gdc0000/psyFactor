import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
import pyreadstat
from io import BytesIO
import base64

st.set_page_config(page_title="Factor and Dimensionality Reduction Explorer", layout="wide")

# ----- Utility Functions -----
def cronbach_alpha(df):
    try:
        df = df.dropna()
        items = df.values.astype(float)
        item_vars = np.var(items, axis=0, ddof=1)
        total_var = np.var(np.sum(items, axis=1), ddof=1)
        n_items = items.shape[1]
        alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_vars) / total_var)
        return alpha
    except Exception as e:
        st.error(f"Error computing Cronbach's alpha: {e}")
        return np.nan

def mcdonald_omega(df):
    try:
        fa = FactorAnalyzer(n_factors=1, rotation=None)
        fa.fit(df)
        loadings = fa.loadings_[:, 0]
        communalities = loadings**2
        unique_vars = 1 - communalities
        omega = (np.sum(loadings)**2) / ((np.sum(loadings)**2) + np.sum(unique_vars))
        return omega
    except Exception as e:
        st.error(f"Error computing McDonald's omega: {e}")
        return np.nan

def item_rest_correlations(df):
    correlations = {}
    try:
        for col in df.columns:
            rest = df.drop(columns=col).sum(axis=1)
            correlations[col] = df[col].corr(rest)
    except Exception as e:
        st.error(f"Error computing item-rest correlations: {e}")
    return correlations

def alpha_if_item_dropped(df):
    alphas = {}
    try:
        for col in df.columns:
            alphas[col] = cronbach_alpha(df.drop(columns=col))
    except Exception as e:
        st.error(f"Error computing 'alpha if item dropped': {e}")
    return alphas

def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(
            np.dot(Phi.T, np.asarray(Lambda)**3 - (gamma / p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda)))))
        )
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d/d_old < 1 + tol:
            break
    return np.dot(Phi, R)

def parallel_analysis(data, n_iter=100, random_state=42):
    n_samples, n_vars = data.shape
    rng = np.random.RandomState(random_state)
    eigenvalues = np.zeros((n_iter, n_vars))
    for i in range(n_iter):
        random_data = rng.normal(size=(n_samples, n_vars))
        pca = PCA()
        pca.fit(random_data)
        eigenvalues[i, :] = pca.explained_variance_
    mean_eigenvalues = np.mean(eigenvalues, axis=0)
    return mean_eigenvalues

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# ----- App Title and Introduction -----
st.title("Factor and Dimensionality Reduction Explorer")
st.markdown(
    """
**Dimensionality reduction** simplifies complex datasets by reducing the number of variables while retaining essential information.  
In social sciences, **factor analysis** uncovers latent constructs underlying observed data, enabling researchers to interpret patterns and relationships with clarity and rigor.
    """
)

# ----- Sidebar: Data Upload and Sample Dataset Selection -----
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV, Excel, or SPSS (.sav) file", type=["csv", "xlsx", "xls", "sav"])
df = None
if uploaded_file is not None:
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith('.sav'):
            df, meta = pyreadstat.read_sav(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV, Excel, or SPSS (.sav) file.")
        st.sidebar.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    sample_option = st.sidebar.radio(
        "No file uploaded. Choose a sample regression dataset:",
        options=["Diabetes", "California Housing", "Linnerud", "Boston Housing", "Make Regression"],
        key="sample_option"
    )
    if sample_option == "Diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        st.sidebar.success("Diabetes dataset loaded!")
    elif sample_option == "California Housing":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        st.sidebar.success("California Housing dataset loaded!")
    elif sample_option == "Linnerud":
        from sklearn.datasets import load_linnerud
        data = load_linnerud()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        st.sidebar.success("Linnerud dataset loaded!")
    elif sample_option == "Boston Housing":
        # Note: Boston Housing dataset is deprecated in recent versions of scikit-learn.
        from sklearn.datasets import load_boston
        data = load_boston()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        st.sidebar.success("Boston Housing dataset loaded!")
    elif sample_option == "Make Regression":
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
        st.sidebar.success("Synthetic regression dataset loaded!")

if df is not None:
    # Only keep numeric columns for analysis
    numeric_df = df.select_dtypes(include=[np.number])
    non_numeric = df.columns.difference(numeric_df.columns)
    if len(non_numeric) > 0:
        st.info(f"The following non-numeric columns have been excluded from analysis: {', '.join(non_numeric)}")
else:
    st.error("No data available.")

st.subheader("Dataset Preview (Numeric Columns)")
if df is not None:
    st.dataframe(numeric_df.head())

# ----- Tabs for Analyses -----
tabs = st.tabs(["Reliability Analysis", "Principal Component Analysis (PCA)", "Exploratory Factor Analysis (EFA)"])

# ----- Reliability Analysis Tab -----
with tabs[0]:
    st.header("Reliability Analysis")
    st.markdown(
        """
**Reliability Analysis** assesses the internal consistency of a set of items (e.g., survey questions).  
- **Cronbach’s α** estimates the degree to which items measure the same construct.  
- **McDonald’s ω** provides a more nuanced reliability estimate by considering the factor structure.  
- **Item-rest correlations** reveal how well each item correlates with the overall scale.  
- **Alpha if item dropped** shows the impact on reliability when an item is removed.
        """
    )
    with st.expander("Learn more about Reliability Analysis"):
        st.markdown(
            """
**What is Reliability Analysis?**  
It evaluates whether a set of items consistently reflects an underlying construct. High reliability implies that items are coherent and measure the same phenomenon.

**Key Metrics Explained:**  
- **Cronbach’s α:** Values above 0.7 are typically acceptable, indicating good internal consistency.  
- **McDonald’s ω:** Often more robust, especially when the assumption of equal factor loadings (tau-equivalence) is violated.  
- **Item-rest correlations and Alpha if item dropped:** Help identify items that may detract from the overall scale's consistency.
            """
        )
    rel_items = st.multiselect("Select items for analysis (numeric columns only)", options=list(numeric_df.columns), default=list(numeric_df.columns))
    if rel_items:
        try:
            rel_data = numeric_df[rel_items].copy()
            reverse_items = st.multiselect("Select reverse-coded items", options=rel_items, help="Choose items where higher scores indicate a lower level of the construct; they will be recoded accordingly.")
            if reverse_items:
                scale_min = st.number_input("Enter scale minimum", value=1, key="rel_scale_min")
                scale_max = st.number_input("Enter scale maximum", value=5, key="rel_scale_max")
                for item in reverse_items:
                    rel_data[item] = scale_max + scale_min - rel_data[item]

            alpha_val = cronbach_alpha(rel_data)
            omega_val = mcdonald_omega(rel_data)
            ir_corr = item_rest_correlations(rel_data)
            alpha_drop = alpha_if_item_dropped(rel_data)
            item_stats = pd.DataFrame({
                "Mean": rel_data.mean(),
                "Std Dev": rel_data.std(),
                "Item-Rest Corr": pd.Series(ir_corr),
                "Alpha if Dropped": pd.Series(alpha_drop)
            })

            st.subheader("Reliability Metrics")
            st.write(f"Cronbach's α: **{alpha_val:.3f}**")
            st.write(f"McDonald's ω: **{omega_val:.3f}**")
            st.subheader("Item Statistics")
            st.dataframe(item_stats)

            st.subheader("Correlation Heatmap")
            corr = rel_data.corr()
            fig, ax = plt.subplots()
            cax = ax.imshow(corr, aspect="auto", interpolation="none")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(corr.index)
            fig.colorbar(cax)
            st.pyplot(fig)

            score_method = st.radio("Compute composite score by:", options=["Mean", "Sum", "None"], key="score_method")
            if score_method != "None":
                if score_method == "Mean":
                    numeric_df["Reliability_MeanScore"] = rel_data.mean(axis=1)
                else:
                    numeric_df["Reliability_SumScore"] = rel_data.sum(axis=1)
                st.success(f"{score_method} score computed and added to the dataset.")
        except Exception as e:
            st.error(f"An error occurred during Reliability Analysis: {e}")

# ----- Principal Component Analysis (PCA) Tab -----
with tabs[1]:
    st.header("Principal Component Analysis (PCA)")
    st.markdown(
        """
**PCA** reduces the dimensionality of your data by creating new, uncorrelated components that capture most of the variance.  
This technique simplifies complex datasets and helps reveal underlying patterns by transforming correlated variables into a smaller set of components.
    """
    )
    with st.expander("Learn more about PCA"):
        st.markdown(
            """
**What is PCA?**  
Principal Component Analysis (PCA) is a method used to reduce the number of variables while retaining as much variability as possible.

**Extraction Methods:**  
- **Eigenvalue > 1:** Retains components with eigenvalues greater than 1, under the assumption that a component should capture more variance than an individual variable.
- **Fixed number:** Allows you to specify a predetermined number of components.
- **Parallel Analysis:** Compares the data's eigenvalues to those from random data and retains only those components that exceed the random thresholds.

**Rotation Options:**  
- **None:** No rotation is applied.
- **Varimax:** An orthogonal rotation that simplifies interpretation by forcing variables to load highly on one component while minimizing cross-loadings.
            """
        )
    pca_vars = st.multiselect("Select variables for PCA (numeric columns only)", options=list(numeric_df.columns), key="pca_vars")
    if pca_vars:
        try:
            pca_data = numeric_df[pca_vars].dropna()
            extraction_method = st.radio("Extraction method", options=["Eigenvalue > 1", "Fixed number", "Parallel Analysis"], key="pca_extraction")
            st.markdown(
                """
                - **Eigenvalue > 1:** Retains components with eigenvalues greater than 1.
                - **Fixed number:** Manually select the number of components.
                - **Parallel Analysis:** Compares your eigenvalues to those from random data.
                """
            )
            fixed_components = None
            if extraction_method == "Fixed number":
                fixed_components = st.number_input("Enter number of components", min_value=1, max_value=len(pca_vars), value=2, step=1, key="pca_fixed")
            rotation_method = st.radio("Rotation", options=["None", "Varimax"], key="pca_rotation")
            st.markdown(
                """
                - **None:** No rotation applied.
                - **Varimax:** Simplifies interpretation by rotating components so that each variable loads highly on one component.
                """
            )
            loading_cutoff = st.number_input("Factor loading cutoff", value=0.3, step=0.1, key="pca_cutoff")
            sort_loadings = st.checkbox("Sort loadings", value=True, key="pca_sort")
            if st.button("Run PCA", key="pca_run"):
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(pca_data)
                pca_full = PCA(n_components=len(pca_vars))
                pca_full.fit(data_scaled)
                eigenvalues = pca_full.explained_variance_
                n_components = len(pca_vars)
                if extraction_method == "Eigenvalue > 1":
                    n_components = np.sum(eigenvalues > 1)
                elif extraction_method == "Parallel Analysis":
                    par_eigs = parallel_analysis(data_scaled)
                    n_components = np.sum(eigenvalues > par_eigs)
                elif extraction_method == "Fixed number" and fixed_components is not None:
                    n_components = int(fixed_components)
                st.write(f"Number of components selected: **{n_components}**")
                pca = PCA(n_components=n_components)
                pca_scores = pca.fit_transform(data_scaled)
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                if rotation_method == "Varimax":
                    loadings = varimax(loadings)
                loadings_display = pd.DataFrame(loadings, index=pca_vars,
                                                columns=[f"Component {i+1}" for i in range(n_components)])
                loadings_display = loadings_display.where(loadings_display.abs() >= loading_cutoff, "")
                if sort_loadings:
                    loadings_display = loadings_display.reindex(loadings_display.abs().max(axis=1).sort_values(ascending=False).index)
                st.subheader("Scree Plot")
                fig, ax = plt.subplots()
                ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker="o")
                ax.set_xlabel("Component number")
                ax.set_ylabel("Eigenvalue")
                st.pyplot(fig)
                st.subheader("Component Loadings")
                st.dataframe(loadings_display)
                for i in range(n_components):
                    numeric_df[f"PCA_Component_{i+1}"] = pca_scores[:, i]
                st.success("PCA scores have been added to the dataset.")
        except Exception as e:
            st.error(f"An error occurred during PCA: {e}")

# ----- Exploratory Factor Analysis (EFA) Tab -----
with tabs[2]:
    st.header("Exploratory Factor Analysis (EFA)")
    st.markdown(
        """
**EFA** uncovers the underlying structure of a set of variables by identifying latent factors that explain observed correlations.  
This method is particularly useful in social sciences for revealing hidden constructs (such as attitudes or personality traits) that influence measured responses.
    """
    )
    with st.expander("Learn more about EFA"):
        st.markdown(
            """
**What is EFA?**  
Exploratory Factor Analysis (EFA) is used to explore potential factor structures without imposing a preconceived model.

**Extraction Methods:**  
- **Eigenvalue > 1:** Retains factors with eigenvalues > 1, assuming each factor should explain more variance than a single variable.
- **Fixed number:** Manually specify the number of factors to extract.
- **Parallel Analysis:** Retains factors whose eigenvalues exceed those derived from random data.

**Rotation Options:**  
- **None:** No rotation is applied.
- **Varimax:** An orthogonal rotation that keeps factors uncorrelated while simplifying interpretation.
- **Oblimin:** An oblique rotation allowing factors to correlate; useful when underlying constructs are expected to be interrelated.
- **Promax:** A computationally faster, approximate oblique rotation similar to Oblimin.
            """
        )
    efa_vars = st.multiselect("Select variables for EFA (numeric columns only)", options=list(numeric_df.columns), key="efa_vars")
    if efa_vars:
        try:
            efa_data = numeric_df[efa_vars].dropna()
            efa_extraction = st.radio("Extraction method", options=["Eigenvalue > 1", "Fixed number", "Parallel Analysis"], key="efa_extraction")
            st.markdown(
                """
                - **Eigenvalue > 1:** Retains factors with eigenvalues > 1.
                - **Fixed number:** Manually select the number of factors.
                - **Parallel Analysis:** Uses random data comparisons to determine factor retention.
                """
            )
            efa_fixed = None
            if efa_extraction == "Fixed number":
                efa_fixed = st.number_input("Enter number of factors", min_value=1, max_value=len(efa_vars), value=2, step=1, key="efa_fixed")
            efa_rotation = st.selectbox("Rotation", options=["None", "Varimax", "Oblimin", "Promax"], key="efa_rotation")
            st.markdown(
                """
                - **None:** No rotation applied.
                - **Varimax:** Orthogonal rotation that keeps factors uncorrelated.
                - **Oblimin:** Oblique rotation that allows factors to correlate; ideal when constructs are expected to interact.
                - **Promax:** A faster, approximate oblique rotation similar to Oblimin.
                """
            )
            if st.button("Run EFA", key="efa_run"):
                scaler = StandardScaler()
                efa_scaled = scaler.fit_transform(efa_data)
                pca_full = PCA(n_components=len(efa_vars))
                pca_full.fit(efa_scaled)
                eigenvalues = pca_full.explained_variance_
                n_factors = len(efa_vars)
                if efa_extraction == "Eigenvalue > 1":
                    n_factors = np.sum(eigenvalues > 1)
                elif efa_extraction == "Parallel Analysis":
                    par_eigs = parallel_analysis(efa_scaled)
                    n_factors = np.sum(eigenvalues > par_eigs)
                elif efa_extraction == "Fixed number" and efa_fixed is not None:
                    n_factors = int(efa_fixed)
                st.write(f"Number of factors selected: **{n_factors}**")
                rotation_param = None if efa_rotation == "None" else efa_rotation.lower()
                fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation_param)
                fa.fit(efa_data)
                loadings = pd.DataFrame(fa.loadings_, index=efa_vars,
                                        columns=[f"Factor {i+1}" for i in range(n_factors)])
                st.subheader("Factor Loadings")
                st.dataframe(loadings.style.applymap(lambda v: "color: red" if abs(v) < 0.3 else ""))
                ev, _ = fa.get_eigenvalues()
                fig, ax = plt.subplots()
                ax.plot(range(1, len(ev) + 1), ev, marker="o")
                ax.set_xlabel("Factor number")
                ax.set_ylabel("Eigenvalue")
                st.subheader("Scree Plot")
                st.pyplot(fig)
                if efa_rotation in ["Oblimin", "Promax"]:
                    if hasattr(fa, "phi_"):
                        phi = pd.DataFrame(fa.phi_, columns=[f"Factor {i+1}" for i in range(n_factors)],
                                           index=[f"Factor {i+1}" for i in range(n_factors)])
                        st.subheader("Factor Correlations")
                        st.dataframe(phi)
                variance, proportion, cumulative = fa.get_factor_variance()
                fit_df = pd.DataFrame({
                    "Variance": variance,
                    "Proportion": proportion,
                    "Cumulative": cumulative
                }, index=[f"Factor {i+1}" for i in range(n_factors)])
                st.subheader("Variance Explained by Factors")
                st.dataframe(fit_df)
                try:
                    efa_scores = fa.transform(efa_data)
                    for i in range(n_factors):
                        numeric_df[f"EFA_Factor_{i+1}"] = efa_scores[:, i]
                    st.success("EFA factor scores have been added to the dataset.")
                except Exception as e:
                    st.error(f"Could not compute factor scores: {e}")
        except Exception as e:
            st.error(f"An error occurred during EFA: {e}")

# ----- Export Enriched Dataset -----
st.markdown("---")
st.header("Export Enriched Dataset")
csv = convert_df_to_csv(numeric_df)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="enriched_dataset.csv",
    mime="text/csv"
)

# ----- Footer -----
st.markdown("---")
st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
st.markdown(
    """
[GitHub](https://github.com/gdc0000) |  
[ORCID](https://orcid.org/0000-0002-1439-5790) |  
[LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
    """
)
