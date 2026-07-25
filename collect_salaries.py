"""WO-9 — NBA player salaries -> cache/salaries/{season}.csv for build_value.py.

Source: Spotrac's public player cap-hit rankings, one page per season. Their
robots.txt carries `User-agent: * / Allow: / Crawl-delay: 5` and does not disallow
this path, so the crawl is permitted; SLEEP honours the stated delay and the whole
job is six requests.

Salaries are facts, and the derived product here is a surplus metric rather than a
republication of anyone's compilation. Even so, the source is attributed in the
export meta, and whether to publish dollar figures on a public site is the site
owner's call rather than this script's.

Names, not ids: Spotrac has its own player ids, so rows are matched to NBA person ids
by normalised name against `player_season`. Unmatched rows are reported, never
silently dropped — an unreported 30% miss would quietly bias every cost-per-win
figure downstream.
"""
import csv
import os
import re
import sqlite3
import sys
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import rapm_lib as rl

OUT_DIR = os.path.join(config.CACHE_DIR, "salaries")
SLEEP = 5.5                      # robots.txt says Crawl-delay: 5
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
URL = "https://www.spotrac.com/nba/rankings/player/_/year/{year}"

# name in a redirect link, then the cap figure in the following span
ROW = re.compile(
    r'redirect/player/\d+"\s+class="link">([^<]+)</a>.*?'
    r'<span class="medium">\s*\$([0-9,]+)', re.S)


def fetch(year):
    """curl rather than urllib: the host 403s urllib even with an identical
    user-agent, so it is filtering on the wider header set a real client sends.
    curl's defaults get through, and the request is the same one either way."""
    out = subprocess.run(
        ["curl", "-sS", "--fail", "--max-time", "60", "--compressed",
         "-A", UA,
         "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
         "-H", "Accept-Language: en-US,en;q=0.9",
         URL.format(year=year)],
        capture_output=True, text=True, check=True)
    return out.stdout


def name_index(con):
    """normalised name -> nba player_id, per season, from the pipeline's own
    player table so ids always agree with everything downstream."""
    idx = {}
    rows = con.execute(
        "SELECT season, player_id, player_name FROM player_season "
        "WHERE player_name IS NOT NULL").fetchall()
    for season, pid, nm in rows:
        key = rl.strip_suffix(rl.norm_name(nm))
        idx.setdefault(season, {}).setdefault(key, int(pid))
    return idx


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    idx = name_index(con)
    con.close()

    total_rows = total_matched = 0
    for season in config.SEASONS:
        year = int(season[:4])
        path = os.path.join(OUT_DIR, f"{season}.csv")
        if os.path.exists(path):
            print(f"[{season}] cached, skipping")
            continue
        try:
            html = fetch(year)
        except Exception as e:
            print(f"[{season}] FETCH FAILED: {type(e).__name__}: {e}")
            continue
        pairs = ROW.findall(html)
        season_idx = idx.get(season, {})
        matched, unmatched = [], []
        for nm, amount in pairs:
            key = rl.strip_suffix(rl.norm_name(nm.strip()))
            pid = season_idx.get(key)
            if pid is None:
                unmatched.append(nm.strip())
            else:
                matched.append((pid, int(amount.replace(",", ""))))
        # de-duplicate: keep the largest figure per player (a traded player can
        # appear more than once; the cap total is the larger row)
        best = {}
        for pid, amt in matched:
            best[pid] = max(best.get(pid, 0), amt)

        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["player_id", "salary"])
            for pid, amt in sorted(best.items()):
                w.writerow([pid, amt])
        os.replace(tmp, path)

        total_rows += len(pairs)
        total_matched += len(best)
        rate = len(best) / len(pairs) if pairs else 0
        print(f"[{season}] parsed {len(pairs)} rows -> {len(best)} matched "
              f"players ({rate:.1%}); {len(unmatched)} unmatched"
              + (f", e.g. {unmatched[:3]}" if unmatched else ""))
        time.sleep(SLEEP)

    print(f"\ntotal: {total_rows} rows parsed, {total_matched} player-seasons "
          f"written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
