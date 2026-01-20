import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# It's recommended to install fpdf2 before running the script:
# pip install fpdf2
try:
    from fpdf import FPDF
except ImportError:
    print("fpdf2 is not installed. Please run 'pip install fpdf2' to generate the PDF report.")
    FPDF = None

# --- Configuration ---
DATA_PATH = Path(__file__).parent.parent / "data" / "raw"
REPORTS_PATH = Path(__file__).parent
FIGURES_PATH = REPORTS_PATH / "figures"
OUTPUTS_PATH = REPORTS_PATH / "outputs"

# Create directories if they don't exist
FIGURES_PATH.mkdir(exist_ok=True)
OUTPUTS_PATH.mkdir(exist_ok=True)

TARGET = 'exam_score'
ID_COL = 'id'

# --- Plotting Functions ---

def plot_numerical_distributions(df, features, figures_path):
    """Plots and saves histograms for numerical features."""
    image_paths = []
    print("  Plotting numerical distributions...")
    for col in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        image_path = figures_path / f"numerical_dist_{col}.png"
        plt.savefig(image_path)
        plt.close()
        image_paths.append(image_path)
    return image_paths

def plot_categorical_distributions(df, features, figures_path):
    """Plots and saves count plots for categorical features."""
    image_paths = []
    print("  Plotting categorical distributions...")
    for col in features:
        plt.figure(figsize=(12, 8))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        image_path = figures_path / f"categorical_dist_{col}.png"
        plt.savefig(image_path)
        plt.close()
        image_paths.append(image_path)
    return image_paths

def plot_numerical_vs_target(df, features, target, figures_path):
    """Plots and saves joint plots for numerical features vs. target."""
    image_paths = []
    print("  Plotting numerical vs. target...")
    # jointplot creates its own figure, so we handle it differently
    for col in features:
        g = sns.jointplot(x=target, y=col, data=df, kind="hex", height=8)
        g.fig.suptitle(f'{col} vs. {target}', y=1.02)
        image_path = figures_path / f"numerical_vs_target_{col}.png"
        g.savefig(image_path)
        plt.close(g.fig)
        image_paths.append(image_path)
    return image_paths

def plot_categorical_vs_target(df, features, target, figures_path):
    """Plots and saves box plots for categorical features vs. target."""
    image_paths = []
    print("  Plotting categorical vs. target...")
    for col in features:
        plt.figure(figsize=(12, 8))
        sns.boxplot(y=col, x=target, data=df, order=df[col].value_counts().index)
        plt.title(f'{col} vs. {target} Distribution')
        plt.xlabel(f'Valor do {target}')
        plt.ylabel(col)
        plt.tight_layout()
        image_path = figures_path / f"categorical_vs_target_{col}.png"
        plt.savefig(image_path)
        plt.close()
        image_paths.append(image_path)
    return image_paths

def plot_correlation_matrix(df, features, target, figures_path):
    """Plots and saves the correlation matrix."""
    print("  Plotting correlation matrix...")
    df_corr = df.copy()
    corr_features = features + [target]
    
    corr_matrix = df_corr[corr_features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix of Numerical Features and Target')
    image_path = figures_path / "correlation_matrix.png"
    plt.savefig(image_path)
    plt.close()
    return [image_path]

# --- PDF Generation ---

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Exploratory Data Analysis Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, images):
        for image_path in images:
            self.set_font('Arial', '', 10)
            # Add a small line break before the figure name
            self.ln(5)
            self.cell(0, 10, f"Figure: {image_path.name.replace('_', ' ').replace('.png', '')}", 0, 1, 'L')
            # A4 width is 210mm, leaving 10mm margin on each side
            self.image(str(image_path), x=10, w=190)

def generate_pdf_report(image_sections, output_path):
    """Generates a PDF report from the saved figures."""
    if FPDF is None:
        return

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    for title, images in image_sections.items():
        pdf.chapter_title(title)
        pdf.chapter_body(images)
        # Add a new page for the next section if it's not the last one
        if title != list(image_sections.keys())[-1]:
            pdf.add_page()
            
    pdf.output(output_path / "eda_report.pdf")

# --- Main Execution ---

def main():
    """Main function to run the EDA and generate the report."""
    print("Starting EDA process...")
    train_df = pd.read_csv(DATA_PATH / 'train.csv')
    
    features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
    numerical_features = train_df[features].select_dtypes(include=np.number).columns.tolist()
    categorical_features = train_df[features].select_dtypes(include=['object', 'category']).columns.tolist()

    image_sections = {}

    print("Generating analysis plots...")
    image_sections["Numerical Feature Distributions"] = plot_numerical_distributions(train_df, numerical_features, FIGURES_PATH)
    image_sections["Categorical Feature Distributions"] = plot_categorical_distributions(train_df, categorical_features, FIGURES_PATH)
    image_sections["Numerical Features vs. Target"] = plot_numerical_vs_target(train_df, numerical_features, TARGET, FIGURES_PATH)
    image_sections["Categorical Features vs. Target"] = plot_categorical_vs_target(train_df, categorical_features, TARGET, FIGURES_PATH)
    image_sections["Correlation Matrix"] = plot_correlation_matrix(train_df, numerical_features, TARGET, FIGURES_PATH)

    if FPDF:
        print("Generating PDF report...")
        generate_pdf_report(image_sections, OUTPUTS_PATH)
        print(f"EDA report generated at: {OUTPUTS_PATH / 'eda_report.pdf'}")
    else:
        print("Skipping PDF generation as fpdf2 is not installed.")
    
    print("EDA process finished.")

if __name__ == '__main__':
    main()
