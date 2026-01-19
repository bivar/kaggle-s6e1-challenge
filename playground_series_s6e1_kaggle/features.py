import pandas as pd
from pathlib import Path
from loguru import logger
import typer

from playground_series_s6e1_kaggle.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features for the model from the raw dataframe.
    """
    logger.info("Starting feature creation...")
    df_featured = df.copy()

    # 1. Ordinal Feature Encoding
    logger.info("Encoding ordinal features...")
    sleep_quality_mapping = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_mapping = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_mapping = {'easy': 0, 'moderate': 1, 'hard': 2}

    df_featured['sleep_quality'] = df_featured['sleep_quality'].map(sleep_quality_mapping)
    df_featured['facility_rating'] = df_featured['facility_rating'].map(facility_rating_mapping)
    df_featured['exam_difficulty'] = df_featured['exam_difficulty'].map(exam_difficulty_mapping)

    # 2. Nominal Feature Encoding (One-Hot)
    logger.info("One-hot encoding nominal features...")
    nominal_cols = ['gender', 'course', 'internet_access', 'study_method']
    df_featured = pd.get_dummies(df_featured, columns=nominal_cols, prefix=nominal_cols, drop_first=True)

    # 3. Interaction and Polynomial Features
    logger.info("Creating interaction and polynomial features...")
    # Handle potential division by zero
    df_featured['study_efficiency'] = df_featured['class_attendance'] / (df_featured['study_hours'] + 1e-6)
    df_featured['study_sleep_interaction'] = df_featured['study_hours'] * df_featured['sleep_hours']

    df_featured['study_hours_sq'] = df_featured['study_hours'] ** 2
    df_featured['class_attendance_sq'] = df_featured['class_attendance'] ** 2

    logger.info("Feature creation finished.")
    return df_featured


@app.command()
def main(
    train_input_path: Path = RAW_DATA_DIR / "train.csv",
    test_input_path: Path = RAW_DATA_DIR / "test.csv",
    train_output_path: Path = PROCESSED_DATA_DIR / "train_featured.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "test_featured.csv",
):
    """
    Loads raw data, creates features, and saves the processed data.
    """
    logger.info("Loading raw data...")
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)

    # Combine train and test sets for consistent encoding
    logger.info("Combining train and test sets for processing...")
    target = train_df['exam_score']

    train_df_processed = train_df.drop(columns=['exam_score'])
    
    combined_df = pd.concat([train_df_processed, test_df], ignore_index=True)

    # Create features
    featured_df = create_features(combined_df)

    # Separate back into train and test sets
    logger.info("Splitting back into train and test sets...")
    train_featured = featured_df.iloc[:len(train_df)]
    test_featured = featured_df.iloc[len(train_df):]

    # Add target back to the training set
    train_featured['exam_score'] = target.values

    # Ensure the output directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving featured training data to {train_output_path}")
    train_featured.to_csv(train_output_path, index=False)

    logger.info(f"Saving featured test data to {test_output_path}")
    test_featured.to_csv(test_output_path, index=False)

    logger.success("Feature generation process complete.")


if __name__ == "__main__":
    app()
