"""
Tests for constraint satisfaction metrics.
"""
import pytest
import pandas as pd
import numpy as np
from sdeval.metrics.constraints import (
    parse_constraint,
    compute_constraint_satisfaction_rate,
    compute_constraint_support
)


class TestConstraintParsing:
    """Test constraint string parsing."""

    def test_parse_single_categorical_constraint(self):
        """Test parsing a single categorical constraint."""
        constraint = "education=11th"
        parsed = parse_constraint(constraint)
        assert parsed == [("education", "11th")]

    def test_parse_single_categorical_constraint_with_spaces(self):
        """Test parsing constraint with spaces in value."""
        constraint = "workclass=State-gov"
        parsed = parse_constraint(constraint)
        assert parsed == [("workclass", "State-gov")]

    def test_parse_multiple_categorical_constraints(self):
        """Test parsing multiple categorical constraints."""
        constraint = "workclass=State-gov,education=Bachelors"
        parsed = parse_constraint(constraint)
        assert len(parsed) == 2
        assert ("workclass", "State-gov") in parsed
        assert ("education", "Bachelors") in parsed

    def test_parse_constraint_with_whitespace(self):
        """Test parsing constraint with extra whitespace."""
        constraint = " workclass = State-gov , education = Bachelors "
        parsed = parse_constraint(constraint)
        assert len(parsed) == 2
        assert ("workclass", "State-gov") in parsed
        assert ("education", "Bachelors") in parsed

    def test_parse_empty_constraint(self):
        """Test parsing empty constraint."""
        constraint = ""
        parsed = parse_constraint(constraint)
        assert parsed == []

    def test_parse_none_constraint(self):
        """Test parsing None constraint."""
        parsed = parse_constraint(None)
        assert parsed == []


class TestConstraintSatisfactionRate:
    """Test constraint satisfaction rate computation."""

    def test_single_constraint_fully_satisfied(self):
        """Test when all samples satisfy constraint."""
        df = pd.DataFrame({
            'education': ['11th', '11th', '11th'],
            'age': [25, 30, 35]
        })
        constraint = "education=11th"
        rate = compute_constraint_satisfaction_rate(df, constraint)
        assert rate == 1.0

    def test_single_constraint_partially_satisfied(self):
        """Test when some samples satisfy constraint."""
        df = pd.DataFrame({
            'education': ['11th', 'Bachelors', '11th', 'Masters'],
            'age': [25, 30, 35, 40]
        })
        constraint = "education=11th"
        rate = compute_constraint_satisfaction_rate(df, constraint)
        assert rate == 0.5  # 2 out of 4

    def test_single_constraint_not_satisfied(self):
        """Test when no samples satisfy constraint."""
        df = pd.DataFrame({
            'education': ['Bachelors', 'Masters', 'Doctorate'],
            'age': [25, 30, 35]
        })
        constraint = "education=11th"
        rate = compute_constraint_satisfaction_rate(df, constraint)
        assert rate == 0.0

    def test_multiple_constraints_all_satisfied(self):
        """Test when all samples satisfy multiple constraints."""
        df = pd.DataFrame({
            'workclass': ['State-gov', 'State-gov', 'State-gov'],
            'education': ['Bachelors', 'Bachelors', 'Bachelors'],
            'age': [25, 30, 35]
        })
        constraint = "workclass=State-gov,education=Bachelors"
        rate = compute_constraint_satisfaction_rate(df, constraint)
        assert rate == 1.0

    def test_multiple_constraints_partially_satisfied(self):
        """Test when some samples satisfy multiple constraints."""
        df = pd.DataFrame({
            'workclass': ['State-gov', 'State-gov', 'Private', 'State-gov'],
            'education': ['Bachelors', 'Masters', 'Bachelors', 'Bachelors'],
            'age': [25, 30, 35, 40]
        })
        constraint = "workclass=State-gov,education=Bachelors"
        rate = compute_constraint_satisfaction_rate(df, constraint)
        assert rate == 0.5  # 2 out of 4 (rows 0 and 3)

    def test_constraint_with_missing_column(self):
        """Test constraint with column not in DataFrame."""
        df = pd.DataFrame({
            'education': ['11th', 'Bachelors'],
            'age': [25, 30]
        })
        constraint = "workclass=State-gov"
        with pytest.raises(KeyError):
            compute_constraint_satisfaction_rate(df, constraint)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({'education': [], 'age': []})
        constraint = "education=11th"
        rate = compute_constraint_satisfaction_rate(df, constraint)
        assert rate == 0.0

    def test_empty_constraint(self):
        """Test with empty constraint (should return 1.0 - all satisfy vacuous truth)."""
        df = pd.DataFrame({
            'education': ['11th', 'Bachelors'],
            'age': [25, 30]
        })
        constraint = ""
        rate = compute_constraint_satisfaction_rate(df, constraint)
        assert rate == 1.0  # Vacuous truth: all rows satisfy empty constraint

    def test_constraint_with_special_characters(self):
        """Test constraint with special characters in values."""
        df = pd.DataFrame({
            'workclass': ['Self-emp-not-inc', 'Private', 'Self-emp-not-inc'],
            'age': [25, 30, 35]
        })
        constraint = "workclass=Self-emp-not-inc"
        rate = compute_constraint_satisfaction_rate(df, constraint)
        assert abs(rate - 0.6667) < 0.001  # 2 out of 3


class TestConstraintSupport:
    """Test the main constraint support/satisfaction metrics function."""

    def test_constraint_support_both_datasets(self):
        """Test computing constraint satisfaction for both real and synthetic."""
        real_df = pd.DataFrame({
            'education': ['11th', 'Bachelors', '11th', 'Masters'],
            'age': [25, 30, 35, 40]
        })
        synthetic_df = pd.DataFrame({
            'education': ['11th', '11th', '11th', 'Bachelors'],
            'age': [26, 31, 36, 41]
        })
        constraint = "education=11th"

        metrics = compute_constraint_support(real_df, synthetic_df, constraint)

        assert 'real_satisfaction_rate' in metrics
        assert 'synthetic_satisfaction_rate' in metrics
        assert 'satisfaction_rate_diff' in metrics
        assert metrics['real_satisfaction_rate'] == 0.5
        assert metrics['synthetic_satisfaction_rate'] == 0.75
        assert abs(metrics['satisfaction_rate_diff'] - 0.25) < 0.001

    def test_constraint_support_perfect_match(self):
        """Test when real and synthetic have same satisfaction rate."""
        real_df = pd.DataFrame({
            'workclass': ['State-gov', 'Private', 'State-gov'],
            'age': [25, 30, 35]
        })
        synthetic_df = pd.DataFrame({
            'workclass': ['State-gov', 'Private', 'State-gov'],
            'age': [26, 31, 36]
        })
        constraint = "workclass=State-gov"

        metrics = compute_constraint_support(real_df, synthetic_df, constraint)

        assert abs(metrics['real_satisfaction_rate'] - 0.6667) < 0.001
        assert abs(metrics['synthetic_satisfaction_rate'] - 0.6667) < 0.001
        assert abs(metrics['satisfaction_rate_diff']) < 0.001

    def test_constraint_support_multiple_constraints(self):
        """Test with multiple constraints."""
        real_df = pd.DataFrame({
            'workclass': ['State-gov', 'State-gov', 'Private', 'State-gov'],
            'education': ['Bachelors', 'Masters', 'Bachelors', 'Bachelors'],
            'age': [25, 30, 35, 40]
        })
        synthetic_df = pd.DataFrame({
            'workclass': ['State-gov', 'State-gov', 'State-gov', 'Private'],
            'education': ['Bachelors', 'Bachelors', 'Masters', 'Bachelors'],
            'age': [26, 31, 36, 41]
        })
        constraint = "workclass=State-gov,education=Bachelors"

        metrics = compute_constraint_support(real_df, synthetic_df, constraint)

        assert metrics['real_satisfaction_rate'] == 0.5  # 2 out of 4
        assert metrics['synthetic_satisfaction_rate'] == 0.5  # 2 out of 4
        assert abs(metrics['satisfaction_rate_diff']) < 0.001

    def test_constraint_support_empty_constraint(self):
        """Test with empty constraint."""
        real_df = pd.DataFrame({
            'education': ['11th', 'Bachelors'],
            'age': [25, 30]
        })
        synthetic_df = pd.DataFrame({
            'education': ['Masters', 'Doctorate'],
            'age': [35, 40]
        })
        constraint = ""

        metrics = compute_constraint_support(real_df, synthetic_df, constraint)

        assert metrics['real_satisfaction_rate'] == 1.0
        assert metrics['synthetic_satisfaction_rate'] == 1.0
        assert metrics['satisfaction_rate_diff'] == 0.0
