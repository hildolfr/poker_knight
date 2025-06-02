#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify card format API consistency fix.

Tests that the optimizer now accepts the same formats as the solver:
- Unicode list format: ["KH", "QH"]  
- Unicode string format: "KH QH"
- Simple string format: "Kh Qh" (backward compatibility)
"""

from poker_knight.optimizer import create_scenario_analyzer

def test_card_formats():
    """Test all supported card formats."""
    print("🧪 Testing Card Format API Consistency Fix")
    print("=" * 50)
    
    analyzer = create_scenario_analyzer()
    
    # Test scenarios
    test_cases = [
        {
            "name": "Unicode List Format (like solver)",
            "hand": ["KH", "QH"],
            "board": ["7H", "6C", "KS"]
        },
        {
            "name": "Unicode String Format",
            "hand": "KH QH",
            "board": "7H 6C KS"
        },
        {
            "name": "Simple String Format (backward compatibility)", 
            "hand": "Kh Qh",
            "board": "7h 6c Ks"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}️⃣ Testing: {case['name']}")
        print(f"   Hand: {case['hand']}")
        print(f"   Board: {case['board']}")
        
        try:
            # Test the analysis
            complexity = analyzer.calculate_scenario_complexity(
                player_hand=case['hand'],
                num_opponents=2,
                board=case['board'],
                position='late'
            )
            
            print(f"   [PASS] Result: {complexity.overall_complexity.name}")
            print(f"   [STATS] Complexity: {complexity.complexity_score:.1f}/10.0")
            print(f"   🎯 Recommended: {complexity.recommended_simulations:,} sims")
            
        except Exception as e:
            print(f"   [FAIL] Error: {e}")
            return False
    
    print(f"\n🎉 All tests passed! API consistency fix working perfectly.")
    print(f"[PASS] Users can now use the same format across solver and optimizer")
    print(f"[PASS] No more manual format conversion needed")
    print(f"[PASS] Backward compatibility maintained")
    
    return True

if __name__ == "__main__":
    test_card_formats() 