"""
Decision Fusion Layer
---------------------
Combines Rule Engine output + ML Anomaly score into a
final Risk Score and Action Decision.

Risk Score: 0.0 (safe) → 1.0 (critical threat)

Final Decision:
  ALLOW  : risk_score < 0.35
  ALERT  : 0.35 ≤ risk_score < 0.70
  BLOCK  : risk_score ≥ 0.70
"""

from dataclasses import dataclass, field
from typing import List, Optional
from engine.rule_engine import RuleEngineOutput, SEVERITY_WEIGHT
from engine.ml_detector import MLResult


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class FusionDecision:
    # Final verdict
    action: str             # ALLOW | ALERT | BLOCK
    risk_score: float       # 0.0 → 1.0

    # Component scores
    rule_score: float       # contribution from rules
    ml_score: float         # contribution from ML
    rule_aware_ml_score: float

    # Explainability
    rule_matched: bool
    matched_rule_names: List[str]
    rule_severity: str
    ml_anomaly_score: float
    reasoning: str          # human-readable explanation

    # Color for UI
    @property
    def color(self) -> str:
        return {"ALLOW": "green", "ALERT": "orange", "BLOCK": "red"}.get(self.action, "gray")

    @property
    def icon(self) -> str:
        return {"ALLOW": "✅", "ALERT": "⚠️", "BLOCK": "🚫"}.get(self.action, "❓")


# ── Fusion Engine ─────────────────────────────────────────────────────────────

class FusionLayer:
    """
    Usage
    -----
    fusion = FusionLayer()
    decision = fusion.decide(rule_output, ml_result)
    """

    def __init__(
        self,
        rule_weight: float = 0.55,   # rules are more precise → higher weight
        ml_weight:   float = 0.45,
        allow_threshold: float = 0.35,
        block_threshold: float = 0.70,
    ):
        assert abs(rule_weight + ml_weight - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.rule_weight     = rule_weight
        self.ml_weight       = ml_weight
        self.allow_threshold = allow_threshold
        self.block_threshold = block_threshold

    def decide(
        self,
        rule_output: RuleEngineOutput,
        ml_result: MLResult,
    ) -> FusionDecision:

        # ── Rule score ────────────────────────────────────────────────────────
        rule_score = 0.0
        if rule_output.any_match:
            # Score the strongest matched rule. A high-confidence LOW rule should
            # not inflate a lower-confidence CRITICAL rule, so evaluate per rule.
            rule_score = max(
                SEVERITY_WEIGHT.get(rule.severity, 0) * rule.confidence
                for rule in rule_output.matched_rules
            )

        # ── ML score ──────────────────────────────────────────────────────────
        raw_ml_score = ml_result.anomaly_score  # pure model score, already 0..1
        ml_score = self._rule_aware_ml_score(raw_ml_score, rule_output)

        # ── Weighted fusion ───────────────────────────────────────────────────
        risk_score = round(
            (self.rule_weight * rule_score) + (self.ml_weight * ml_score),
            4
        )
        risk_score = float(min(risk_score, 1.0))

        # ── Boost: if both agree it's bad, amplify ────────────────────────────
        if rule_output.any_match and ml_result.is_anomaly:
            boost = min(0.15, (1.0 - risk_score) * 0.3)
            risk_score = round(risk_score + boost, 4)

        # Signature rules are deterministic evidence. Do not let a low ML score
        # dilute medium/high/critical rule hits into ALLOW.
        risk_score = max(
            risk_score,
            self._rule_risk_floor(rule_output),
        )
        risk_score = round(float(min(risk_score, 1.0)), 4)

        # ── Decision ──────────────────────────────────────────────────────────
        if risk_score >= self.block_threshold:
            action = "BLOCK"
        elif risk_score >= self.allow_threshold:
            action = "ALERT"
        else:
            action = "ALLOW"

        # Rule policy override: ML can reduce uncertainty, but it must not
        # downgrade deterministic medium-or-higher signatures to ALLOW.
        action = self._apply_rule_action_floor(action, rule_output)

        # ── Reasoning ─────────────────────────────────────────────────────────
        matched_names = [r.rule_name for r in rule_output.matched_rules]
        reasoning = self._build_reasoning(
            action, risk_score, rule_output, ml_result, matched_names, ml_score
        )

        return FusionDecision(
            action=action,
            risk_score=risk_score,
            rule_score=round(rule_score, 4),
            ml_score=round(ml_score, 4),
            rule_aware_ml_score=round(ml_score, 4),
            rule_matched=rule_output.any_match,
            matched_rule_names=matched_names,
            rule_severity=rule_output.highest_severity,
            ml_anomaly_score=raw_ml_score,
            reasoning=reasoning,
        )

    def _rule_aware_ml_score(self, raw_ml_score: float, rule_output: RuleEngineOutput) -> float:
        if not rule_output.any_match:
            return raw_ml_score

        floors = {
            "LOW": 0.0,
            "MEDIUM": 0.45,
            "HIGH": 0.65,
            "CRITICAL": 0.85,
        }
        return max(raw_ml_score, floors.get(rule_output.highest_severity, 0.0))

    def _rule_risk_floor(self, rule_output: RuleEngineOutput) -> float:
        if not rule_output.any_match:
            return 0.0

        floors = {
            "LOW": 0.0,
            "MEDIUM": self.allow_threshold,
            "HIGH": max(self.allow_threshold, 0.55),
            "CRITICAL": self.block_threshold,
        }
        return floors.get(rule_output.highest_severity, 0.0)

    def _apply_rule_action_floor(self, action: str, rule_output: RuleEngineOutput) -> str:
        if not rule_output.any_match:
            return action

        if rule_output.highest_severity == "CRITICAL":
            return "BLOCK"

        if rule_output.highest_severity in {"MEDIUM", "HIGH"} and action == "ALLOW":
            return "ALERT"

        return action

    def _build_reasoning(
        self,
        action: str,
        risk_score: float,
        rule_output: RuleEngineOutput,
        ml_result: MLResult,
        matched_names: List[str],
        rule_aware_ml_score: float,
    ) -> str:
        parts = []

        if rule_output.any_match:
            parts.append(
                f"Rule engine matched {len(matched_names)} rule(s): "
                f"{', '.join(matched_names)} "
                f"[Severity: {rule_output.highest_severity}]"
            )
        else:
            parts.append("No signature rules matched.")

        if ml_result.is_anomaly:
            parts.append(
                f"ML model predicted {ml_result.predicted_label or 'anomalous traffic'} "
                f"(attack score: {ml_result.anomaly_score:.2f}, confidence: {ml_result.confidence:.2f})"
            )
        else:
            parts.append(
                f"ML model predicted {ml_result.predicted_label or 'Benign'} "
                f"(attack score: {ml_result.anomaly_score:.2f}, confidence: {ml_result.confidence:.2f})"
            )

        if (
            rule_output.any_match
            and rule_output.highest_severity in {"MEDIUM", "HIGH", "CRITICAL"}
            and ml_result.anomaly_score < 0.35
        ):
            parts.append(
                "Rule-only detection: ML score is low because the classifier did not see enough attack evidence "
                "in its selected features/preprocessing"
            )

        if rule_aware_ml_score > ml_result.anomaly_score:
            parts.append(f"Rule-aware ML score raised to {rule_aware_ml_score:.2f}")

        parts.append(f"Final risk score: {risk_score:.2f} → Action: {action}")
        return " | ".join(parts)
