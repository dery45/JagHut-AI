import sys
import os
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

from core.jaghut_core import JaghutCore

def test_query_flow():
    jaghut = JaghutCore()
    user_id = "test_user_fix"
    
    print("\n--- Testing Query 1 ---")
    q1 = "analisis skor ForestIQ Wilmar"
    print(f"User: {q1}")
    a1 = jaghut.ask(user_id, q1)
    # print(f"Jaghut: {a1[:100]}...")
    
    print("\n--- Testing Query 2 (Referential) ---")
    q2 = "bandingkan dengan skor APRIL"
    print(f"User: {q2}")
    a2 = jaghut.ask(user_id, q2)
    # print(f"Jaghut: {a2[:100]}...")
    
    # Check if we got an error message or a real answer
    if "Maaf, terjadi kesalahan" in a2:
        print("\n❌ FAILED: Still getting error in Query 2")
    else:
        print("\n✅ SUCCESS: No crash in Query 2")

if __name__ == "__main__":
    test_query_flow()
