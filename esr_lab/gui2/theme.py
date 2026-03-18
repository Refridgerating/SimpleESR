"""Shared style constants for the Qt GUI scaffold."""

from __future__ import annotations


THEME_STYLESHEETS: dict[str, str] = {
    "dark": """
QMainWindow {
    background-color: #111317;
}
QWidget {
    color: #d8dee9;
    font-size: 10pt;
}
QPushButton {
    background-color: #2c89c6;
    border: 1px solid #3aa1e0;
    border-radius: 6px;
    padding: 6px 10px;
}
QPushButton:hover {
    background-color: #37a0e6;
}
QPushButton:disabled {
    background-color: #2b3640;
    color: #8392a2;
    border-color: #33424f;
}
QListWidget, QTableView {
    background-color: #171a1f;
    gridline-color: #2a313a;
    selection-background-color: #2d5f87;
}
QHeaderView::section {
    background-color: #20252c;
    color: #ced7e0;
    border: 0px;
    padding: 5px;
}
QProgressBar {
    border: 1px solid #33424f;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #2c89c6;
}
""",
    "light": """
QMainWindow {
    background-color: #f3f6fa;
}
QWidget {
    color: #1a2b3a;
    font-size: 10pt;
}
QPushButton {
    background-color: #2f86c3;
    color: #ffffff;
    border: 1px solid #2a79af;
    border-radius: 6px;
    padding: 6px 10px;
}
QPushButton:hover {
    background-color: #4b97ca;
}
QPushButton:disabled {
    background-color: #d3dce6;
    color: #8a98a6;
    border-color: #c3ced9;
}
QListWidget, QTableView {
    background-color: #ffffff;
    gridline-color: #d9e1ea;
    selection-background-color: #b9d7f1;
}
QHeaderView::section {
    background-color: #ecf1f6;
    color: #1f2d3b;
    border: 0px;
    padding: 5px;
}
QProgressBar {
    border: 1px solid #bfd0e1;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #2f86c3;
}
""",
}


def normalize_theme(theme: str) -> str:
    return "light" if theme == "light" else "dark"


def stylesheet_for(theme: str) -> str:
    return THEME_STYLESHEETS[normalize_theme(theme)]

