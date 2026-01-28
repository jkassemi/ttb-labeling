import pytest

from cola_label_verification.models import LabelInfo
from cola_label_verification.rules import ApplicationFields, RulesConfig
from cola_label_verification.rules.models import Finding
import cola_label_verification.rules.engine as engine


def _make_finding(rule_id: str) -> Finding:
    return Finding(
        rule_id=rule_id,
        status="pass",
        message="ok",
        severity="info",
    )


def test_evaluate_checklist_builds_context_and_runs_rules_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label_info = LabelInfo()
    application_fields = ApplicationFields(brand_name="Test Brand")
    rules_config = RulesConfig(verification_threshold=0.42)
    spans = (object(),)
    images = (object(),)

    contexts: list[object] = []

    def _rule_one(context: engine.RuleContext) -> Finding:
        contexts.append(context)
        return _make_finding("rule_one")

    def _rule_two(context: engine.RuleContext) -> Finding:
        contexts.append(context)
        return _make_finding("rule_two")

    monkeypatch.setattr(engine, "ALL_RULES", (_rule_one, _rule_two))

    result = engine.evaluate_checklist(
        label_info,
        application_fields=application_fields,
        rules_config=rules_config,
        images=images,
        spans=spans,
    )

    assert [finding.rule_id for finding in result.findings] == [
        "rule_one",
        "rule_two",
    ]
    assert len(contexts) == 2
    assert contexts[0] is contexts[1]

    context = contexts[0]
    assert context.label_info is label_info
    assert context.application_fields is application_fields
    assert context.rules_config is rules_config
    assert context.spans is spans
    assert context.images is images


def test_evaluate_checklist_uses_default_config_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[engine.RuleContext] = []

    def _rule(context: engine.RuleContext) -> Finding:
        captured.append(context)
        return _make_finding("default_config")

    monkeypatch.setattr(engine, "ALL_RULES", (_rule,))

    engine.evaluate_checklist(LabelInfo())

    assert len(captured) == 1
    context = captured[0]
    assert isinstance(context.rules_config, RulesConfig)
    assert context.rules_config.verification_threshold == 0.65


def test_evaluate_checklist_propagates_rule_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _rule(context: engine.RuleContext) -> Finding:
        raise ValueError("boom")

    monkeypatch.setattr(engine, "ALL_RULES", (_rule,))

    with pytest.raises(ValueError, match="boom"):
        engine.evaluate_checklist(LabelInfo())
