"""Reporting helpers for Synthetic Data Evaluator."""

from .html_report import generate_html_report
from .writer import write_summary

__all__ = ["write_summary", "generate_html_report"]
