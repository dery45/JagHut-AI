import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / "config" / ".env")
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    print("No MONGO_URI found")
    sys.exit(1)

import certifi
try:
    print("Testing connection with certifi...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    client.admin.command('ping')
    print("Connection successful with certifi!")
except Exception as e:
    print(f"Error with certifi: {e}")

try:
    print("Testing connection with tlsAllowInvalidCertificates=True and no certifi...")
    client2 = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tls=True, tlsAllowInvalidCertificates=True)
    client2.admin.command('ping')
    print("Connection successful with tlsAllowInvalidCertificates=True!")
except Exception as e:
    print(f"Error with tlsAllowInvalidCertificates: {e}")
