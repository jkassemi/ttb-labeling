from cola_label_verification.models import LabelInfo
from cola_label_verification.rules import evaluate_checklist


def _find_finding(result, rule_id: str):
    for finding in result.findings:
        if finding.rule_id == rule_id:
            return finding
    raise AssertionError(f"{rule_id} finding not found")


def test_alcohol_content_extraction(
    crystal_springs_extraction: LabelInfo,
) -> None:
    result = evaluate_checklist(
        crystal_springs_extraction,
    )
    finding = _find_finding(result, "alcohol_content_presence")
    assert finding.status == "pass"
