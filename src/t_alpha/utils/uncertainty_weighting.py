import numpy as np


def apply_uncertainty_weighting(df, scale=10, output_csv=None):
    """
    Apply weighting to the MC dropout results based on uncertainty.

    Args:
        df (pd.DataFrame): DataFrame containing 'prediction' and 'uncertainty' columns.
        scale (float, optional): Scale parameter for the weighting transformation. Defaults to 10.
        output_csv (str, optional): Path to save the weighted DataFrame as a CSV file.
                                    If None, the CSV is not saved. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'weight' column.
    """
    # Ensure required columns are present
    if not {"prediction", "uncertainty"}.issubset(df.columns):
        raise ValueError(
            "Input DataFrame must contain 'prediction' and 'uncertainty' columns."
        )

    # Drop rows with missing uncertainty values
    df = df.dropna(subset=["uncertainty"]).copy()

    # Check if uncertainty has variation to avoid division by zero
    if df["uncertainty"].nunique() > 1:
        # Normalize the uncertainty values to range between 0 and 1
        min_uncertainty = df["uncertainty"].min()
        max_uncertainty = df["uncertainty"].max()
        df["normalized_uncertainty"] = (df["uncertainty"] - min_uncertainty) / (
            max_uncertainty - min_uncertainty
        )
    else:
        # If all uncertainty values are the same, set normalized_uncertainty to 0.5
        df["normalized_uncertainty"] = 0.5

    # Apply the negative sigmoid transformation with the given scale
    df["weight"] = 1 - 1 / (1 + np.exp(-scale * (df["normalized_uncertainty"] - 0.5)))

    # Drop the 'normalized_uncertainty' column as it's no longer needed
    df = df.drop(columns=["normalized_uncertainty"])

    # Save the weighted results to a CSV file if output_csv is provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Weighted results saved to {output_csv}")

    return df
