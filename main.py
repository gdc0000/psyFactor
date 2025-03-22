import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from io import BytesIO
import base64

st.set_page_config(page_title="Social Science Analysis Tool", layout="wide")

# ----- Utility Functions -----
def cronbach_alpha(df):
    df = df.dropna()
    items = df.values
    item_vars = np.var(items, axis=0, ddof=1)
    total_var = np.var(np.sum(items, axis=1), ddof=1)
    n_items = items.shape[1]
    alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_vars) / total_var)
    return alpha

def mcdonald_omega(df):
    # One-factor factor analysis for omega estimation
    fa = FactorAnalyzer(n_factors=1, rotation=None)
    fa.fit(df)
    loadings = fa.loadings_[:, 0]
    communalities = loadings**2
    unique_vars = 1 - communalities
    # Omega formula: (sum(loadings))^2 / ((sum(loadings))^2 + sum(unique variances))
    omega = (np.sum(loadings)**2) / ((np.sum(loadings)**2) + np.sum(unique_vars))
    return omega

def item_rest_correlations(df):
    correlations = {}
    for col in df.columns:
        rest = df.drop(columns=col).sum(axis=1)
        correlations[col] = df[col].corr(rest)
    return correlations

def alpha_if_item_dropped(df):
    alphas = {}
    for col in df.columns:
        alphas[col] = cronbach_alpha(df.drop(columns=col))
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

# ----- App Title and Data Upload -----
st.title("Social Science Analysis Tool")
st.markdown(
    """
This application supports **Reliability Analysis**, **Principal Component Analysis (PCA)**, and **Exploratory Factor Analysis (EFA)** with clear, educational explanations. Use the sidebar to upload your dataset or try the demo data.
    """
)

# Sidebar: Data Upload
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Data successfully loaded!")
else:
    st.sidebar.info("Using demo dataset (100 cases, 10 items).")
    np.random.seed(0)
    df = pd.DataFrame(
        np.random.randint(1, 6, size=(100, 10)),
        columns=[f"Q{i}" for i in range(1, 11)]
    )

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ----- Tabs for Analyses -----
tabs = st.tabs(["Reliability Analysis", "Principal Component Analysis (PCA)", "Exploratory Factor Analysis (EFA)"])

# ----- Reliability Analysis Tab -----
with tabs[0]:
    st.header("Reliability Analysis")
    st.markdown(
        """
**Reliability Analysis** helps you assess the internal consistency of a set of items (e.g., survey questions).
- **Cronbach’s α** estimates how closely related a set of items are.
- **McDonald’s ω** is another reliability estimate.
- **Item-rest correlations** show how each item correlates with the sum of the others.
- **Alpha if item dropped** indicates how reliability changes if an item is removed.
        """
    )
    with st.expander("Learn more about Reliability Analysis"):
        st.markdown(
            """
- **Cronbach’s Alpha (α):** Values closer to 1 indicate high internal consistency.
- **McDonald’s Omega (ω):** Considers the factor structure of the items.
- **Reverse-coded Items:** Sometimes items are phrased oppositely; they must be recoded.
            """
        )
    rel_items = st.multiselect("Select items for analysis", options=list(df.columns), default=list(df.columns))
    if rel_items:
        rel_data = df[rel_items].copy()
        reverse_items = st.multiselect("Select reverse-coded items", options=rel_items)
        if reverse_items:
            scale_min = st.number_input("Enter scale minimum", value=1)
            scale_max = st.number_input("Enter scale maximum", value=5)
            for item in reverse_items:
                rel_data[item] = scale_max + scale_min - rel_data[item]

        # Compute reliability metrics
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

        # Correlation heatmap
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

        # Option to compute and add mean or sum score
        score_method = st.radio("Compute composite score by:", options=["Mean", "Sum", "None"])
        if score_method != "None":
            if score_method == "Mean":
                df["Reliability_MeanScore"] = rel_data.mean(axis=1)
            else:
                df["Reliability_SumScore"] = rel_data.sum(axis=1)
            st.success(f"{score_method} score computed and added to the dataset.")

# ----- Principal Component Analysis (PCA) Tab -----
with tabs[1]:
    st.header("Principal Component Analysis (PCA)")
    st.markdown(
        """
**PCA** reduces the dimensionality of your data while preserving as much variability as possible.
- It creates new components that are linear combinations of your variables.
- **Rotation (e.g., Varimax)** can make the components easier to interpret.
        """
    )
    with st.expander("Learn more about PCA"):
        st.markdown(
            """
- **Extraction Methods:**
    - **Eigenvalue > 1:** Only components with eigenvalues greater than 1 are retained.
    - **Fixed Number:** You specify the number of components.
    - **Parallel Analysis:** Compares eigenvalues with those from random data.
- **Factor Loading Cutoff:** Only loadings above a threshold are shown.
        """
        )
    pca_vars = st.multiselect("Select variables for PCA", options=list(df.columns))
    if pca_vars:
        pca_data = df[pca_vars].dropna()
        extraction_method = st.radio("Extraction method", options=["Eigenvalue > 1", "Fixed number", "Parallel Analysis"])
        fixed_components = None
        if extraction_method == "Fixed number":
            fixed_components = st.number_input("Enter number of components", min_value=1, max_value=len(pca_vars), value=2, step=1)
        rotation_method = st.radio("Rotation", options=["None", "Varimax"])
        loading_cutoff = st.number_input("Factor loading cutoff", value=0.3, step=0.1)
        sort_loadings = st.checkbox("Sort loadings", value=True)
        if st.button("Run PCA"):
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(pca_data)
            # Determine number of components
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
            # Run PCA with determined number of components
            pca = PCA(n_components=n_components)
            pca_scores = pca.fit_transform(data_scaled)
            # Compute loadings: components_.T * sqrt(eigenvalues)
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            if rotation_method == "Varimax":
                loadings = varimax(loadings)
            # Apply cutoff
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
            # Save PCA scores to dataset
            for i in range(n_components):
                df[f"PCA_Component_{i+1}"] = pca_scores[:, i]
            st.success("PCA scores have been added to the dataset.")

# ----- Exploratory Factor Analysis (EFA) Tab -----
with tabs[2]:
    st.header("Exploratory Factor Analysis (EFA)")
    st.markdown(
        """
**EFA** seeks to uncover the underlying structure of a set of variables.
- It identifies latent factors that explain the observed correlations.
- **Rotation** (e.g., Oblimin, Promax) can help in interpreting factors.
        """
    )
    with st.expander("Learn more about EFA"):
        st.markdown(
            """
- **Extraction Methods:**
    - **Eigenvalue > 1:** Retain factors with eigenvalues greater than 1.
    - **Fixed Number:** Specify the number of factors.
    - **Parallel Analysis:** Compare with random data eigenvalues.
- **Model Fit:** Measures how well the factor model represents the data.
        """
        )
    efa_vars = st.multiselect("Select variables for EFA", options=list(df.columns))
    if efa_vars:
        efa_data = df[efa_vars].dropna()
        efa_extraction = st.radio("Extraction method", options=["Eigenvalue > 1", "Fixed number", "Parallel Analysis"])
        efa_fixed = None
        if efa_extraction == "Fixed number":
            efa_fixed = st.number_input("Enter number of factors", min_value=1, max_value=len(efa_vars), value=2, step=1)
        efa_rotation = st.selectbox("Rotation", options=["None", "Varimax", "Oblimin", "Promax"])
        if st.button("Run EFA"):
            # Standardize data
            scaler = StandardScaler()
            efa_scaled = scaler.fit_transform(efa_data)
            # Determine number of factors using PCA on correlation matrix
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
            # Set rotation parameter
            rotation_param = None if efa_rotation == "None" else efa_rotation.lower()
            fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation_param)
            fa.fit(efa_data)
            loadings = pd.DataFrame(fa.loadings_, index=efa_vars,
                                    columns=[f"Factor {i+1}" for i in range(n_factors)])
            st.subheader("Factor Loadings")
            st.dataframe(loadings.style.applymap(lambda v: "color: red" if abs(v) < 0.3 else ""))
            # Scree Plot
            ev, _ = fa.get_eigenvalues()
            fig, ax = plt.subplots()
            ax.plot(range(1, len(ev) + 1), ev, marker="o")
            ax.set_xlabel("Factor number")
            ax.set_ylabel("Eigenvalue")
            st.subheader("Scree Plot")
            st.pyplot(fig)
            # Factor correlations (if available)
            if efa_rotation in ["Oblimin", "Promax"]:
                if hasattr(fa, "phi_"):
                    phi = pd.DataFrame(fa.phi_, columns=[f"Factor {i+1}" for i in range(n_factors)],
                                       index=[f"Factor {i+1}" for i in range(n_factors)])
                    st.subheader("Factor Correlations")
                    st.dataframe(phi)
            # Model Fit Measures
            variance, proportion, cumulative = fa.get_factor_variance()
            fit_df = pd.DataFrame({
                "Variance": variance,
                "Proportion": proportion,
                "Cumulative": cumulative
            }, index=[f"Factor {i+1}" for i in range(n_factors)])
            st.subheader("Variance Explained by Factors")
            st.dataframe(fit_df)
            # Save EFA factor scores to dataset
            try:
                efa_scores = fa.transform(efa_data)
                for i in range(n_factors):
                    df[f"EFA_Factor_{i+1}"] = efa_scores[:, i]
                st.success("EFA factor scores have been added to the dataset.")
            except Exception as e:
                st.error(f"Could not compute factor scores: {e}")

# ----- Export Enriched Dataset -----
st.markdown("---")
st.header("Export Enriched Dataset")
csv = convert_df_to_csv(df)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="enriched_dataset.csv",
    mime="text/csv"
)

# ----- Footer -----
st.markdown("---")
st.markdown("### **Dr. Gabriele Di Cicco, PhD in Social Psychology**")
st.markdown(
    """
[GitHub](https://github.com/gdc0000) |  
[ORCID](https://orcid.org/0000-0002-1439-5790) |  
[LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
    """
)
