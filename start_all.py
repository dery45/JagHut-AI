import os
import subprocess
import sys
import time
import signal
from pathlib import Path

# Suppress pydantic V1 deprecation warnings on Python 3.14+
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")

# Script to launch all Jaghut Background Services + Telegram Bot
# Run with: `python start_all.py`

_HEALTH_CHECK_DELAY  = 5    # seconds after launch before first health check
_MONITOR_INTERVAL    = 10   # seconds between health checks
_MAX_RESTART_ATTEMPTS = 3   # max auto-restarts per service before giving up

def run_service(name: str, cmd: list[str]) -> subprocess.Popen:
    print(f"🚀 Starting {name}...")
    try:
        return subprocess.Popen(cmd)
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None


class ServiceManager:
    def __init__(self):
        self.services: list[dict] = []   # {name, cmd, process, restarts}

    def add(self, name: str, cmd: list[str]):
        proc = run_service(name, cmd)
        self.services.append({
            "name":     name,
            "cmd":      cmd,
            "process":  proc,
            "restarts": 0,
        })

    def check_health(self):
        """Poll each service; restart if crashed and below restart limit."""
        for svc in self.services:
            proc = svc["process"]
            if proc is None:
                continue
            rc = proc.poll()
            if rc is not None:
                # Process has exited
                if svc["restarts"] < _MAX_RESTART_ATTEMPTS:
                    svc["restarts"] += 1
                    print(f"⚠️  [{svc['name']}] exited (code={rc}) — "
                          f"restarting ({svc['restarts']}/{_MAX_RESTART_ATTEMPTS})...")
                    try:
                        svc["process"] = subprocess.Popen(svc["cmd"])
                    except Exception as e:
                        print(f"❌  [{svc['name']}] restart failed: {e}")
                        svc["process"] = None
                else:
                    if proc is not None:
                        print(f"🛑  [{svc['name']}] reached max restarts ({_MAX_RESTART_ATTEMPTS}). "
                              f"Not restarting. Check logs.")
                        svc["process"] = None   # mark as permanently down

    def terminate_all(self):
        for svc in self.services:
            if svc["process"]:
                svc["process"].terminate()
        for svc in self.services:
            if svc["process"]:
                svc["process"].wait()


def main():
    print("==================================================")
    print("          Jaghut v0.1L Master Startup Script        ")
    print("==================================================")

    import urllib.request
    import urllib.error
    import time
    print("⏳ Waiting for ChromaDB to be ready on port 8000...")
    chroma_ready = False
    for _ in range(30):
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/api/v1", method="GET")
            urllib.request.urlopen(req, timeout=1)
            chroma_ready = True
            break
        except urllib.error.HTTPError:
            # Server responded but returned HTTP error (e.g., 404, 501) - meaning it is online!
            chroma_ready = True
            break
        except Exception:
            try:
                req2 = urllib.request.Request("http://localhost:8000/api/v1", method="GET")
                urllib.request.urlopen(req2, timeout=1)
                chroma_ready = True
                break
            except urllib.error.HTTPError:
                chroma_ready = True
                break
            except Exception:
                pass
        time.sleep(1)
        
    if not chroma_ready:
        print("❌ ChromaDB not reachable after 30 seconds. Exiting...")
        sys.exit(1)
        
    print("✅ ChromaDB is ready!")

    mgr = ServiceManager()

    # 1. Start Vector CSV/XLSX Service
    mgr.add("Vector CSV Watcher",    [sys.executable, str(Path("services/vectorCSV.py"))])
    # 2. Start Vector PDF Service
    mgr.add("Vector PDF Watcher",    [sys.executable, str(Path("services/vectorpdf.py"))])
    # 3. Start Weather Service
    mgr.add("Weather Insight Service",[sys.executable, str(Path("services/vectorWeather.py"))])
    # 4. Start Daily Insight (MongoDB Async Pushing)
    mgr.add("Daily Insight Cron",    [sys.executable, str(Path("services/daily_insight.py"))])
    # 5. Start Company Insight Engine (ForestIQ assessment → LLM insight)
    mgr.add("Company Insight Engine",[sys.executable, str(Path("services/company_insight_engine.py"))])
    # 6. Start Telegram Bot
    mgr.add("Telegram Bot",          [sys.executable, str(Path("interfaces/telegram/telegram_bot.py"))])

    print("\n✅ All background services initiated! (Press Ctrl+C to stop all)")
    print("   Note: Ensure 'chroma run --path data/db --port 8000' is running in a separate terminal!")
    print("==================================================\n")

    # Initial health check after a short delay
    print(f"   ⏳ Health check in {_HEALTH_CHECK_DELAY}s...")
    time.sleep(_HEALTH_CHECK_DELAY)
    mgr.check_health()

    try:
        while True:
            time.sleep(_MONITOR_INTERVAL)
            mgr.check_health()
    except KeyboardInterrupt:
        print("\nShutting down all Jaghut services...")
        mgr.terminate_all()
        print("✅ Shutdown complete.")


if __name__ == "__main__":
    main()
