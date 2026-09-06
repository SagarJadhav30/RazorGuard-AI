"""
RazorGuard AI - SHAP Explainability Demonstration Script
"""

import sys
import os
import json
from datetime import datetime 

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

from ml.utils.data_generator import get_preset_scenarios
from backend.app.services.explanation_service import ExplanationService


def main():
    print("==================================================================")
    print("  RazorGuard AI - SHAP Explainability & Risk Assessment Demo")
    print("==================================================================")

    presets = get_preset_scenarios()
    service = ExplanationService(models_dir="models")

    for key, scenario in presets.items():
        print(f"\n--- Testing Scenario: {scenario['name']} ---")
        print(f"Description: {scenario['description']}")

        result = service.explain_transaction(scenario["data"])

        print(f"\n  Final Decision:        [{result['decision']}]")
        print(f"  Risk Level:            {result['risk_level']}")
        print(f"  Final Risk Score:      {result['final_risk_score']} / 100")
        print(f"  Fraud Probability:     {result['fraud_probability']*100:.1f}%")
        print(f"  Triggered Rules:       {result['triggered_rules']}")

        print("\n  Top Risk Factors (Pushing Score UP):")
        if result["risk_factors"]:
            for rf in result["risk_factors"]:
                print(f"    - {rf['feature_name_human']}: SHAP Impact = +{rf['shap_value']:.4f} (Value: {rf['feature_value']})")
        else:
            print("    None detected")

        print("\n  Top Protective Factors (Pushing Score DOWN):")
        if result["protective_factors"]:
            for pf in result["protective_factors"]:
                print(f"    - {pf['feature_name_human']}: SHAP Impact = {pf['shap_value']:.4f} (Value: {pf['feature_value']})")
        else:
            print("    None detected")

        print("------------------------------------------------------------------")

    print("\n[+] Structured Explanation JSON Sample:")
    print(json.dumps(result, indent=2))
    print("==================================================================")


if __name__ == "__main__":
    main()
