"""Shared loaders for the DraftDB evaluation lab.

Reads only the public model repo (draftdb/) and the private data repo (DraftDB-Data/) that sit
next to this folder. Everything is keyed by pid; names come from DraftDB-Data/identity and are
used for display only.
"""
import os, json, csv, re, unicodedata, collections
import numpy as np, pandas as pd

LAB = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(LAB))
# Private data repo (player names, dated multi-site consensus, v4.22 engine inputs); not part of the public repo.
DATA = os.environ.get("DRAFTDB_DATA", os.path.join(os.path.dirname(REPO), "DraftDB-Data"))
V4 = f"{DATA}/v4_build/staging_v422"          # latest full input matrix (1,673 inputs)
OUT = os.environ.get("DRAFTDB_LAB_OUT", f"{REPO}/experiments/lab_2026-10-19")
os.makedirs(OUT, exist_ok=True)

TEST_YEARS = list(range(2019, 2026))
S = [f"y_s{i}_war" for i in range(1, 6)]
DRAFT_DATE = {2000:"06-28",2001:"06-27",2002:"06-26",2003:"06-26",2004:"06-24",2005:"06-28",2006:"06-28",
    2007:"06-28",2008:"06-26",2009:"06-25",2010:"06-24",2011:"06-23",2012:"06-28",2013:"06-27",2014:"06-26",
    2015:"06-25",2016:"06-23",2017:"06-22",2018:"06-21",2019:"06-20",2020:"11-18",2021:"07-29",2022:"06-23",
    2023:"06-22",2024:"06-26",2025:"06-25",2026:"06-24"}


def norm_name(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace(".", " ").replace("-", " ").replace("'", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def identity():
    d = pd.read_csv(f"{DATA}/identity/tabular_names.csv")
    d["draft_year"] = pd.to_numeric(d["draft_year"], errors="coerce")
    return d


def names():
    return dict(zip(identity().pid, identity().player_name))


def answers(year):
    a = pd.read_csv(f"{REPO}/data/answers/answers_{year}.csv")
    for c in S + ["actual_pick", "y_early_war", "y_minutes_5yr", "y_early_tier_code", "y_second_contract_aav",
                  "y_earnings_first5", "y_games_5yr"] + [f"y_s{i}_minutes" for i in range(1, 6)] + [f"y_s{i}_impact" for i in range(1, 6)]:
        if c in a.columns:
            a[c] = pd.to_numeric(a[c], errors="coerce")
    a["was_drafted"] = pd.to_numeric(a["was_drafted"], errors="coerce").fillna(0).astype(int)
    return a


def horizon(year):
    """Seasons completed so far for a test class, capped at 5 (same rule as vault.horizons)."""
    a = answers(year)
    return int(np.clip(a[S].notna().sum(axis=1).max(), 1, 5))


def war_k(a, k):
    """k-season WAR per row (missing seasons count 0, like vault.score)."""
    return a[S[:k]].fillna(0).sum(axis=1).values


def pool(year, k=None, drafted_only=True):
    """Scoring pool for one class: pid, actual_pick, war (k seasons), plus alternate outcomes."""
    a = answers(year)
    k = k or horizon(year)
    d = a.copy()
    d["war"] = war_k(a, k)
    d["war_pos"] = np.clip(d["war"], 0, None)
    d["minutes"] = a[[f"y_s{i}_minutes" for i in range(1, k + 1)]].fillna(0).sum(axis=1).values
    d["k"] = k
    if drafted_only:
        d = d[(d.was_drafted == 1) & d.actual_pick.notna()].copy()
    return d[["pid", "draft_year", "was_drafted", "actual_pick", "war", "war_pos", "minutes", "k",
              "y_early_tier_code", "y_second_contract_aav", "y_games_5yr"]].reset_index(drop=True)


def draft_history(year):
    rs = json.load(open(f"{REPO}/data/draft_history/draft_{year}.json"))["resultSets"][0]
    H = {h: i for i, h in enumerate(rs["headers"])}
    rows = [{"nba_id": int(r[H["PERSON_ID"]]), "name": r[H["PLAYER_NAME"]], "pick": int(r[H["OVERALL_PICK"]]),
             "round": int(r[H["ROUND_NUMBER"]]), "team_id": r[H["TEAM_ID"]], "abbr": r[H["TEAM_ABBREVIATION"]],
             "team": f'{r[H["TEAM_CITY"]]} {r[H["TEAM_NAME"]]}', "org": r[H["ORGANIZATION"]]} for r in rs["rowSet"]]
    d = pd.DataFrame(rows).sort_values("pick").reset_index(drop=True)
    idn = identity()
    by_nba = {int(v): p for p, v in zip(idn.pid, idn.nba_id) if pd.notna(v)}
    d["pid"] = d.nba_id.map(by_nba)
    d["year"] = year
    return d


def train():
    t = pd.read_csv(f"{REPO}/data/train_2000_2018.csv", low_memory=False)
    if "actual_pick" not in t.columns:            # picks live in the identity file, not the matrix
        idn = identity()[["pid", "actual_pick"]]
        t = t.merge(idn, on="pid", how="left")
    for c in S + ["actual_pick", "y_early_war", "bio_age_at_draft", "col_impact", "hs_rsci_rank",
                  "cons_mock_consensus_rank"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    return t


def test_inputs(year, version="public"):
    p = f"{REPO}/data/tests/test_{year}_inputs.csv" if version == "public" else f"{V4}/tests/test_{year}_inputs.csv"
    return pd.read_csv(p, low_memory=False)


# ------------------------------------------------------------------ boards
def board_current_strict():
    """The current model's strict board (trained 2007-2018), names included; higher score = better."""
    b = pd.read_csv(f"{REPO}/model/board_current_2019_2025.csv")
    return b.rename(columns={"season": "year"})


def board_current_expanding():
    """Expanding-window ranks reconstructed from the team-card audit (boards_used = expanding)."""
    a = pd.read_csv(f"{REPO}/data/team_cards_audit.csv")
    a = a[a.actual_model_rank.notna() & (a.actual_pid != "")]
    b = a[["year", "actual_pid", "actual_model_rank", "model_score"]].drop_duplicates("actual_pid")
    b = b.rename(columns={"actual_pid": "pid", "actual_model_rank": "model_rank"})
    b["score"] = -b.model_rank.astype(float)       # higher = better
    return b[["year", "pid", "model_rank", "score"]].reset_index(drop=True)


def board_colin():
    b = pd.read_csv(f"{LAB}/data/colin_board.csv")
    b["score"] = -b.colin_rank.astype(float)
    return b[["year", "pid", "colin_rank", "score"]]


def consensus_multisite():
    """Dated multi-publisher mock consensus (Wayback, last capture before draft), 2015-2026.
    Returns long frame pid, year, site, rank and a wide mean-rank consensus per pid-year."""
    m = pd.read_csv(f"{DATA}/mock_consensus2_ranks_merged.csv")
    m = m.rename(columns={"draft_year": "year"})
    return m


# ------------------------------------------------------------------ pick value curve
def pick_value_curve(max_pick=60, years=(2000, 2018), smooth=True):
    """Expected 5-season WAR by overall pick, from drafted players in the training classes.
    Isotonic (non-increasing) fit of mean WAR on pick, then a light smoothing. Returns array
    v[1..max_pick] (index 0 unused)."""
    from sklearn.isotonic import IsotonicRegression
    t = train()
    t = t[(t.draft_year >= years[0]) & (t.draft_year <= years[1]) & (t.actual_pick.notna())]
    w = t[S].fillna(0).sum(axis=1).values
    p = t.actual_pick.values.astype(float)
    ir = IsotonicRegression(increasing=False, y_min=0.0).fit(p, w)
    grid = np.arange(1, max_pick + 1, dtype=float)
    v = ir.predict(grid)
    if smooth:
        # 5-point moving average, keeps monotone shape, removes isotonic plateaus
        vs = np.convolve(np.pad(v, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
        v = np.minimum.accumulate(np.maximum(vs, 0))
    return np.concatenate([[np.nan], v])     # v[pick]


def write_json(obj, name):
    with open(f"{OUT}/{name}", "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
