import pandas as pd
import pytest


@pytest.fixture(scope="module")
def clean():
    return pd.read_csv("data/churn_clean.csv")


def test_no_identifier_column(clean):
    assert "customerID" not in clean.columns


def test_totalcharges_is_numeric(clean):
    assert pd.api.types.is_numeric_dtype(clean["TotalCharges"])


def test_no_missing_values(clean):
    assert clean.isnull().sum().sum() == 0


def test_target_is_binary(clean):
    assert set(clean["Churn"].unique()) == {0, 1}


def test_new_customers_have_zero_charges(clean):
    """The 11 tenure=0 rows must be 0, not median-imputed."""
    new_customers = clean[clean["tenure"] == 0]
    assert (new_customers["TotalCharges"] == 0).all()


def test_row_count_preserved(clean):
    assert len(clean) == 7043


def test_no_negative_charges(clean):
    assert (clean["TotalCharges"] >= 0).all()
    assert (clean["MonthlyCharges"] >= 0).all()