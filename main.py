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
**Dimensionality reduction** is the process of simplifying complex datasets by reducing the number of variables while retaining essential information.  
In social sciences, **factor analysis** is a key technique that uncovers latent constructs (factors) underlying observed variables, helping researchers interpret and understand data patterns.
    """
)

# ----- Sidebar: Data Upload -----
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
    st.sidebar.info("Using demo dataset (100 cases, 10 items).")
    np.random.seed(0)
    df = pd.DataFrame(
        np.random.randint(1, 6, size=(100, 10)),
        columns=[f"Q{i}" for i in range(1, 11)]
    )

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
- **Cronbach’s α** estimates how closely related the items are.  
- **McDonald’s ω** considers the factor structure and provides an alternative measure.  
- **Item-rest correlations** reveal how each item correlates with the total score from other items.  
- **Alpha if item dropped** shows how the overall reliability would change if an item were removed.  
    """
    )
    with st.expander("Learn more about Reliability Analysis"):
        st.markdown(
            """
**What is Reliability Analysis?**  
Reliability analysis examines the extent to which a set of items measures a single unidimensional latent construct. High reliability indicates that the items consistently reflect the underlying construct.  
        
**Key Methods:**  
- **Cronbach’s α:** A widely used metric where values above 0.7 are generally considered acceptable.  
- **McDonald’s ω:** Often seen as a more robust alternative, especially when the assumption of tau-equivalence is violated.  
- **Item-rest correlations and Alpha if item dropped:** These help identify problematic items that may not align well with the overall scale.
            """
        )
    rel_items = st.multiselect("Select items for analysis (numeric columns only)", options=list(numeric_df.columns), default=list(numeric_df.columns))
    if rel_items:
        try:
            rel_data = numeric_df[rel_items].copy()
            reverse_items = st.multiselect("Select reverse-coded items", options=rel_items, help="Choose items where higher scores indicate a lower level of the construct; they will be recoded accordingly.")
            if reverse_items:
                scale_min = st.number_input("Enter scale minimum", value=1)
                scale_max = st.number_input("Enter scale maximum", value=5)
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

            score_method = st.radio("Compute composite score by:", options=["Mean", "Sum", "None"])
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
Principal Component Analysis (PCA) is a mathematical technique used to reduce the number of variables in a dataset while retaining as much information as possible.  
        
**Key Points:**  
- **Extraction Methods:**  
  - **Eigenvalue > 1:** Only keep components with eigenvalues greater than 1.  
  - **Fixed Number:** Specify the number of components to retain.  
  - **Parallel Analysis:** Compare the eigenvalues with those obtained from random data.  
- **Rotation (e.g., Varimax):** Helps simplify the interpretation by maximizing the variance of the loadings.
- **Purpose:** PCA is ideal when you want to simplify data for visualization or further analysis without losing critical information.
            """
        )
    pca_vars = st.multiselect("Select variables for PCA (numeric columns only)", options=list(numeric_df.columns))
    if pca_vars:
        try:
            pca_data = numeric_df[pca_vars].dropna()
            extraction_method = st.radio("Extraction method", options=["Eigenvalue > 1", "Fixed number", "Parallel Analysis"])
            fixed_components = None
            if extraction_method == "Fixed number":
                fixed_components = st.number_input("Enter number of components", min_value=1, max_value=len(pca_vars), value=2, step=1)
            rotation_method = st.radio("Rotation", options=["None", "Varimax"])
            loading_cutoff = st.number_input("Factor loading cutoff", value=0.3, step=0.1)
            sort_loadings = st.checkbox("Sort loadings", value=True)
            if st.button("Run PCA"):
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
This method is particularly useful in social sciences for revealing hidden constructs (like attitudes or personality traits) that influence measured responses.
    """
    )
    with st.expander("Learn more about EFA"):
        st.markdown(
            """
**What is EFA?**  
Exploratory Factor Analysis (EFA) is used to explore the possible underlying factor structure of a set of observed variables without imposing a preconceived structure on the outcome.  
        
**Key Points:**  
- **Extraction Methods:**  
  - **Eigenvalue > 1:** Retain factors with eigenvalues greater than 1.  
  - **Fixed Number:** Specify the number of factors.  
  - **Parallel Analysis:** Use random data to help decide the number of factors.
- **Rotation Options:** Such as Oblimin or Promax, which allow factors to correlate, or Varimax, which assumes orthogonality.
- **Purpose:** EFA is ideal when you want to explore and identify latent constructs that explain the relationships among variables.
            """
        )
    efa_vars = st.multiselect("Select variables for EFA (numeric columns only)", options=list(numeric_df.columns))
    if efa_vars:
        try:
            efa_data = numeric_df[efa_vars].dropna()
            efa_extraction = st.radio("Extraction method", options=["Eigenvalue > 1", "Fixed number", "Parallel Analysis"])
            efa_fixed = None
            if efa_extraction == "Fixed number":
                efa_fixed = st.number_input("Enter number of factors", min_value=1, max_value=len(efa_vars), value=2, step=1)
            efa_rotation = st.selectbox("Rotation", options=["None", "Varimax", "Oblimin", "Promax"])
            if st.button("Run EFA"):
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
