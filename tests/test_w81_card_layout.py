"""W8.1 card containment regression: no research content flush against borders.

Uses stdlib HTMLParser and CSS invariants so the normal Python CI matrix
checks structural changes even without Chromium. Browser visual QA remains
a separate release requirement.
"""
from html.parser import HTMLParser
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1] / "frontend"


class LayoutTree(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.scroll_regions = []
        self.label_targets = []
        self.select_ids = []
        self.nested_tables = 0

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        cls = attr.get("class", "").split()
        if tag == "div" and "table-wrap" in cls:
            self.scroll_regions.append({
                "role": attr.get("role"),
                "tabindex": attr.get("tabindex"),
                "aria-label": attr.get("aria-label"),
                "tables": 0,
            })
        if tag == "table" and self.scroll_regions:
            self.scroll_regions[-1]["tables"] += 1
        if tag == "label" and attr.get("for", "").startswith("policy"):
            self.label_targets.append(attr["for"])
        if tag == "select" and attr.get("id", "").startswith("policy"):
            self.select_ids.append(attr["id"])


def test_research_card_padding_and_table_containment_are_explicit():
    css = (ROOT / "styles.css").read_text()
    assert "--panel-inset" in css
    for name in (
        ".research-evidence-card", ".model-explanations-card",
        ".policy-explorer-card",
    ):
        assert name in css
    card_section = css.split("/* W8.1 · card typography and responsive containment.")[-1]
    assert "padding: var(--panel-inset)" in card_section
    assert "overflow-x: auto" in card_section
    assert "max-width: 100%" in card_section
    assert "min-width: 950px" in card_section
    assert "scrollbar-width: thin" in card_section
    # Never globally clip the card or body: overflow should remain discoverable.
    assert not re.search(r"\.card\s*\{[^}]*overflow:\s*hidden", card_section)
    assert not re.search(r"body\s*\{[^}]*overflow-x:\s*hidden", card_section)


def test_tables_remain_accessible_inside_scrollable_cards():
    html = (ROOT / "index.html").read_text()
    parser = LayoutTree()
    parser.feed(html)
    assert len(parser.scroll_regions) >= 2
    for region in parser.scroll_regions:
        assert region["role"] == "region"
        assert region["tabindex"] == "0"
        assert region["aria-label"]
        assert region["tables"] >= 1
    assert "Scroll within the table" in html
    assert "Ridge interval coverage" in html
    assert "95% descriptive CI" in html


def test_policy_controls_keep_correct_explicit_label_connections():
    html = (ROOT / "index.html").read_text()
    parser = LayoutTree()
    parser.feed(html)
    expected = {
        "policyAsset", "policyModel", "policyHorizon", "policyWidth"
    }
    assert expected <= set(parser.label_targets)
    assert expected <= set(parser.select_ids)
    assert html.count('class="policy-field"') == 4
    assert 'id="policyRiskChart"' in html


def test_mobile_layout_respects_card_and_chart_minimum_sizes():
    css = (ROOT / "styles.css").read_text().split(
        "/* W8.1 · card typography and responsive containment."
    )[-1]
    assert "@media (max-width: 980px)" in css
    assert "@media (max-width: 680px)" in css
    assert "grid-template-areas:" in css
    assert '"feature value"' in css and '"track track"' in css
    assert "grid-template-columns: 1fr;" in css
    assert "overflow-wrap: anywhere" in css


def test_release_cache_tag_changes_after_spacing_fix():
    html = (ROOT / "index.html").read_text()
    assert 'href="./styles.css?v=w81-card-fit"' in html
    assert 'src="./app.js?v=w81-card-fit"' in html
