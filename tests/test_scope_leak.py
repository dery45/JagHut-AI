import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

from core.jaghut_core import JaghutCore

def test():
    jaghut = JaghutCore()
    uid  = "test_scope_fix"

    print("="*60)
    print("TEST 1: Normal ESG query")
    print("="*60)
    q1 = "analisis skor ESG Wilmar dan APRIL"
    print(f"User: {q1}")
    r1 = jaghut.ask(uid, q1)
    print(f"Jaghut: {r1[:200]}...\n")

    print("="*60)
    print("TEST 2: Referential followup (suffix-based rewrite)")
    print("="*60)
    q2 = "bagaimana perbandingan metrik deforestasi mereka"
    print(f"User: {q2}")
    r2 = jaghut.ask(uid, q2)
    print(f"Jaghut: {r2[:200]}...\n")

    print("="*60)
    print("TEST 3: Out-of-scope with 'itu' (definitional)")
    print("="*60)
    q3 = "apa itu machine learning"
    print(f"User: {q3}")
    r3 = jaghut.ask(uid, q3)
    is_refused = (r3 == jaghut.refusal_msg)
    print(f"Jaghut: {r3[:200]}")
    if is_refused:
        print("\n✅ TEST 3 PASSED: Correctly blocked out-of-scope query!")
    else:
        print("\n❌ TEST 3 FAILED: Should have been blocked!")

    print("\n" + "="*60)
    print("TEST 4: 'what is EUDR' (definitional, in-scope)")
    print("="*60)
    q4 = "apa itu EUDR"
    print(f"User: {q4}")
    r4 = jaghut.ask(uid, q4)
    is_refused4 = (r4 == jaghut.refusal_msg)
    print(f"Jaghut: {r4[:200]}")
    if not is_refused4:
        print("\n✅ TEST 4 PASSED: Allowed in-scope definitional query!")
    else:
        print("\n⚠️  TEST 4: Blocked — check if 'EUDR' is in scope keywords")

if __name__ == "__main__":
    test()
