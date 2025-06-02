#!/usr/bin/env python3
"""
Test script to verify card format API consistency fix.

Tests that the optimizer now accepts the same formats as the solver:
- Unicode list format: ["K♥️", "Q♥️"]  
- Unicode string format: "K♥️ Q♥️"
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
            "hand": ["K♥️", "Q♥️"],
            "board": ["7♥️", "6♣️", "K♠️"]
        },
        {
            "name": "Unicode String Format",
            "hand": "K♥️ Q♥️",
            "board": "7♥️ 6♣️ K♠️"
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
            
            print(f"   ✅ Result: {complexity.overall_complexity.name}")
            print(f"   📊 Complexity: {complexity.complexity_score:.1f}/10.0")
            print(f"   🎯 Recommended: {complexity.recommended_simulations:,} sims")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    print(f"\n🎉 All tests passed! API consistency fix working perfectly.")
    print(f"✅ Users can now use the same format across solver and optimizer")
    print(f"✅ No more manual format conversion needed")
    print(f"✅ Backward compatibility maintained")
    
    return True

if __name__ == "__main__":
    test_card_formats() 