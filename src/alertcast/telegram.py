"""Scrape the public web preview (t.me/s/<channel>) of a Telegram channel.

No login / API keys are needed. Each page holds ~20 posts; we paginate
backwards with ?before=<post_id> until we pass `since`.

NOTE on leakage: Telegram shows the *current* text of a post, so an edited
post could carry information added after publication. We key every post by
its original publication timestamp and only use posts published strictly
before the forecast time. The Air Force channel rarely edits threat posts,
but this is a documented residual risk (see README).
"""
from __future__ import annotations

import html
import re
import time
from pathlib import Path

import pandas as pd
import requests

POST_RE = re.compile(
    r'data-post="[^/]+/(\d+)".*?<time datetime="([^"]+)"', re.S
)
MSG_SPLIT = 'class="tgme_widget_message_wrap'
TEXT_RE = re.compile(r'<div class="tgme_widget_message_text[^>]*>(.*?)</div>', re.S)


def _clean(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment)
    return html.unescape(re.sub(r"<[^>]+>", " ", fragment)).strip()


def parse_page(page: str) -> list[dict]:
    rows = []
    for chunk in page.split(MSG_SPLIT)[1:]:
        m = POST_RE.search(chunk)
        if not m:
            continue
        t = TEXT_RE.search(chunk)
        rows.append(
            {
                "post_id": int(m.group(1)),
                "published_at": m.group(2),
                "text": _clean(t.group(1)) if t else "",
            }
        )
    return rows


def scrape_channel(channel: str, since: str, out: Path, pause: float = 0.4) -> pd.DataFrame:
    since_ts = pd.Timestamp(since, tz="UTC")
    existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
    known = set(existing["post_id"]) if len(existing) else set()
    rows, before = [], None
    session = requests.Session()
    while True:
        url = f"https://t.me/s/{channel}" + (f"?before={before}" if before else "")
        for attempt in range(4):
            try:
                r = session.get(url, timeout=30)
                r.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(2 ** (attempt + 1))
        else:
            raise RuntimeError(f"failed to fetch {url}")
        page = parse_page(r.text)
        if not page:
            break
        rows.extend(p for p in page if p["post_id"] not in known)
        oldest = min(page, key=lambda p: p["post_id"])
        if pd.Timestamp(oldest["published_at"]) < since_ts:
            break
        if known and oldest["post_id"] <= max(known):
            break  # reached already-scraped history (incremental update)
        before = oldest["post_id"]
        time.sleep(pause)
    df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    df = df.drop_duplicates("post_id").sort_values("post_id")
    df = df[pd.to_datetime(df["published_at"], utc=True) >= since_ts]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
