from core.jaghut_core import JaghutCore
import os

def test_scope():
    jaghut = JaghutCore()
    user_id = "test_user"
    
    # Pre-load context
    print("\n--- Context: ESG ---")
    jaghut.ask("analisis skor ForestIQ Wilmar", user_id)
    
    test_cases = [
        "bagaimana perbandingan dengan APRIL?",
        "apa komitmen deforestasi mereka?",
        "siapa kamu ?",
        "apa itu jaghut?"
    ]
    
    for q in test_cases:
        print(f"\nTesting: '{q}'")
        response = jaghut.ask(q, user_id)
        if response == jaghut.refusal_msg:
            print(f"❌ REJECTED: {q}")
        else:
            print(f"✅ ALLOWED: {q}")
            # print(f"Response snippet: {response[:100]}...")

if __name__ == "__main__":
    test_scope()
