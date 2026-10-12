"""Download Ukrainian Air Force (kpszsu) posts from the public t.me web preview."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertcast.telegram import scrape_channel  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="kpszsu")
    ap.add_argument("--since", default="2026-05-01")
    ap.add_argument("--out", default="data/raw/telegram_kpszsu.csv")
    a = ap.parse_args()
    df = scrape_channel(a.channel, a.since, Path(a.out))
    print(f"{len(df)} posts, {df['published_at'].min()} .. {df['published_at'].max()}")
