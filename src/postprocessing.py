"""
Post-processing: Apply TargetPred_To_Keep mask and prepare final submission.
This is CRITICAL - the competition says entries without reference values must be 0.
"""
import pandas as pd
import numpy as np
from config import TARGETS, SUBMISSION_DIR, TARGET_KEEP_FILE, SAMPLE_SUB_FILE
import os


def apply_target_mask(predictions_df, target_keep_path=TARGET_KEEP_FILE):
    """
    Zero out predictions where TargetPred_To_Keep is 0.
    This is a MUST - failing to do this will inflate RMSE.
    """
    target_keep = pd.read_csv(target_keep_path)

    # Get the ID column name
    id_col = target_keep.columns[0]

    # Ensure predictions have the same ID column
    pred_id_col = predictions_df.columns[0]

    for target in TARGETS:
        if target in target_keep.columns and target in predictions_df.columns:
            # Create mask: 0 means no reference value, prediction should be 0
            mask_map = target_keep.set_index(id_col)[target]
            mask = predictions_df[pred_id_col].map(mask_map).fillna(0)
            predictions_df[target] = predictions_df[target] * mask

    return predictions_df


def clip_predictions(predictions_df, lower=0, upper=None):
    """Clip predictions to valid range (nutrients can't be negative)."""
    for target in TARGETS:
        if target in predictions_df.columns:
            predictions_df[target] = predictions_df[target].clip(lower=lower, upper=upper)
    return predictions_df


def create_submission(predictions_df, filename="submission.csv"):
    """Create a properly formatted submission file."""
    sample_sub = pd.read_csv(SAMPLE_SUB_FILE)
    id_col = sample_sub.columns[0]

    # Ensure same format as sample submission
    submission = sample_sub[[id_col]].copy()

    for target in TARGETS:
        if target in predictions_df.columns:
            pred_id = predictions_df.columns[0]
            pred_map = predictions_df.set_index(pred_id)[target]
            submission[target] = submission[id_col].map(pred_map).fillna(0)
        else:
            submission[target] = 0

    # Apply mask
    submission = apply_target_mask(submission)

    # Clip to non-negative
    submission = clip_predictions(submission)

    # Save
    filepath = os.path.join(SUBMISSION_DIR, filename)
    submission.to_csv(filepath, index=False)
    print(f"\nSubmission saved to: {filepath}")
    print(f"Shape: {submission.shape}")
    print(f"Non-zero predictions per target:")
    for target in TARGETS:
        nonzero = (submission[target] > 0).sum()
        print(f"  {target}: {nonzero}")

    return submission


def validate_submission(submission_path):
    """Validate submission format against sample."""
    sample_sub = pd.read_csv(SAMPLE_SUB_FILE)
    submission = pd.read_csv(submission_path)

    # Check shape
    assert submission.shape == sample_sub.shape, \
        f"Shape mismatch: {submission.shape} vs {sample_sub.shape}"

    # Check columns
    assert list(submission.columns) == list(sample_sub.columns), \
        f"Column mismatch: {list(submission.columns)} vs {list(sample_sub.columns)}"

    # Check IDs
    id_col = submission.columns[0]
    assert set(submission[id_col]) == set(sample_sub[id_col]), \
        "ID mismatch between submission and sample"

    # Check no negatives
    for target in TARGETS:
        if target in submission.columns:
            assert (submission[target] >= 0).all(), f"Negative values found in {target}"

    # Check mask
    target_keep = pd.read_csv(TARGET_KEEP_FILE)
    tk_id = target_keep.columns[0]
    for target in TARGETS:
        if target in target_keep.columns and target in submission.columns:
            merged = submission.merge(target_keep[[tk_id, target]], on=tk_id,
                                       suffixes=("_pred", "_mask"))
            should_be_zero = merged[merged[f"{target}_mask"] == 0][f"{target}_pred"]
            assert (should_be_zero == 0).all(), \
                f"Non-zero predictions found where mask is 0 for {target}"

    print("Submission validation PASSED!")
    return True
