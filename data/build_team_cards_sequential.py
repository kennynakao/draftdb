#!/usr/bin/env python3
"""Build team_cards.json + team_cards_audit.csv by a SEQUENTIAL redraft.

The real draft 2019-2025 is replayed in overall-pick order. At each slot the
team on the clock takes the model's highest-ranked player still available;
every player is taken exactly once, so a player can appear as the model's
choice on only one card. (Replaces data/build_team_cards.py, which compared
each pick independently with "best player still available at that slot" and
let the same player appear on many cards.)

Inputs (all inside the repo):
  data/draft_history/draft_YYYY.json   stats.nba.com DraftHistory
  model/board_expanding_2019_2025.csv  year, model_rank, pid, player, actual_pick, score
  data/answers/answers_YYYY.csv        per-pid WAR to date and games/minutes
  data/pick_value_curve.json           expected 5-season WAR by overall pick
  data/team_cards.json                 (previous output, read only for slugs)

Outputs:
  data/team_cards.json
  data/team_cards_audit.csv
"""
import csv, collections, datetime, json, os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = HERE                                         # REPO/data
REPO = os.path.dirname(DATA)
DRAFT_DIR = os.path.join(DATA, 'draft_history')
ANSWERS = os.path.join(DATA, 'answers', 'answers_{}.csv')
BOARD = os.path.join(REPO, 'model', 'board_expanding_2019_2025.csv')
CURVE = os.path.join(DATA, 'pick_value_curve.json')
OLD_JSON = os.path.join(DATA, 'team_cards.json')
OUT_JSON = os.path.join(DATA, 'team_cards.json')
OUT_CSV = os.path.join(DATA, 'team_cards_audit.csv')
YEARS = list(range(2019, 2026))
EPS = 0.05
METHOD = ('sequential redraft: the real draft replayed in pick order, each player '
          'taken once (replaces best-available-with-repeats, 2026-10-19)')
log = collections.defaultdict(list)


def fnum(v, default=0.0):
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- inputs
curve = json.load(open(CURVE))['expected_war_by_pick']


def slot_value(pick):
    return float(curve[min(max(pick, 1), 60)])


board = pd.read_csv(BOARD)
board['year'] = board['year'].astype(int)
# the open pool: the strict engine's ranking of the clean pre-draft universe (H7), tools/lab/export_open_pool_board.py
OPEN = os.path.join(REPO, 'data', 'open_pool_board_2019_2025.csv')
open_board = pd.read_csv(OPEN) if os.path.exists(OPEN) else None
if open_board is not None:
    open_board['year'] = open_board['year'].astype(int)

answers = {}                                        # (year, pid) -> dict
pid_by_pick = {}                                    # (year, pick) -> pid
for y in YEARS:
    a = pd.read_csv(ANSWERS.format(y))
    for r in a.to_dict('records'):
        answers[(y, r['pid'])] = {
            'war': fnum(r.get('y_early_war')),
            'seasons': int(fnum(r.get('y_partial_seasons'))),
            'complete': bool(r.get('y_window_complete')),
            'games': fnum(r.get('y_games_5yr')),
            'minutes': fnum(r.get('y_minutes_5yr')),
        }
        if fnum(r.get('was_drafted')) == 1 and not pd.isna(r.get('actual_pick')):
            pid_by_pick[(y, int(r['actual_pick']))] = r['pid']

brow = {}                                           # (year, pid) -> board row
for r in board.to_dict('records'):
    brow[(r['year'], r['pid'])] = r
    pid_by_pick[(r['year'], int(r['actual_pick']))] = r['pid']   # board mapping wins

# slugs: keep identical to the existing file
slug_of = {}
if os.path.exists(OLD_JSON):
    for t in json.load(open(OLD_JSON))['teams']:
        slug_of[t['team_id']] = t['slug']

# ---------------------------------------------------------------- draft history
teams = {}                                          # team_id -> (city, name, abbr)
picks = collections.defaultdict(list)
for y in YEARS:
    rs = json.load(open(os.path.join(DRAFT_DIR, f'draft_{y}.json')))['resultSets'][0]
    H = {h: i for i, h in enumerate(rs['headers'])}
    for row in rs['rowSet']:
        tid = row[H['TEAM_ID']]
        teams[tid] = (row[H['TEAM_CITY']], row[H['TEAM_NAME']], row[H['TEAM_ABBREVIATION']])
        pk = int(row[H['OVERALL_PICK']])
        picks[y].append({
            'pick': pk, 'round': int(row[H['ROUND_NUMBER']]), 'team_id': tid,
            'abbr': row[H['TEAM_ABBREVIATION']],
            'team_name': f'{row[H["TEAM_CITY"]]} {row[H["TEAM_NAME"]]}',
            'name': row[H['PLAYER_NAME']], 'org': row[H['ORGANIZATION']],
            'pid': pid_by_pick.get((y, pk)),
        })
    picks[y].sort(key=lambda p: p['pick'])
for tid in teams:
    if tid not in slug_of:
        city, name, _ = teams[tid]
        slug_of[tid] = re.sub(r'[^a-z0-9]+', '-', f'{city} {name}'.lower()).strip('-')
        log['slug_generated'].append(f'{tid} {slug_of[tid]}')

pick_by_pid = {(y, p['pid']): p for y in YEARS for p in picks[y] if p['pid']}


def war_rec(y, pid):
    return answers.get((y, pid))


def war_of(y, pid):
    r = war_rec(y, pid) if pid else None
    return r['war'] if r else 0.0


# ---------------------------------------------------------------- sequential redraft
cards_by_team = collections.defaultdict(lambda: collections.defaultdict(list))
audit = []
choice_count = collections.Counter()

for y in YEARS:
    yb = board[board['year'] == y].sort_values('model_rank')
    available = list(yb['pid'])                     # already in rank order
    open_avail = ([r for r in open_board[open_board['year'] == y].sort_values('model_rank').to_dict('records')]
                  if open_board is not None else [])
    drafted = [p for p in picks[y] if p['pid'] and (y, p['pid']) in answers]
    for p in picks[y]:                              # overall-pick order
        pk = p['pick']
        cp = None
        if available:
            cpid = available.pop(0)                 # lowest model_rank still available
            cp = pick_by_pid.get((y, cpid))
            br = brow[(y, cpid)]
            choice_count[(y, cpid)] += 1
        aw = war_rec(y, p['pid']) if p['pid'] else None
        a_war = aw['war'] if aw else 0.0
        if p['pid'] is None:
            log['actual_no_outcome_record'].append(f'{y} #{pk} {p["name"]} ({p["abbr"]})')
        elif aw is None:
            log['actual_no_answers_row'].append(f'{y} #{pk} {p["name"]} ({p["abbr"]})')
        a_brow = brow.get((y, p['pid'])) if p['pid'] else None

        hind = max((h for h in drafted if h['pick'] > pk and h['pid'] != p['pid']),
                   key=lambda h: (war_of(y, h['pid']), -h['pick']), default=None)

        if cp is None:
            log['no_model_choice'].append(f'{y} #{pk} {p["abbr"]}')
            model_choice, c_war, delta, verdict, cw = None, None, None, None, {}
        else:
            c_war = war_of(y, cpid)
            cw = war_rec(y, cpid) or {}
            if cpid == p['pid']:
                delta, verdict = 0.0, 'tie'
                log['same_player'].append(f'{y} #{pk}')
            else:
                delta = c_war - a_war
                verdict = 'model' if delta > EPS else ('team' if delta < -EPS else 'tie')
            model_choice = {
                'pid': cpid, 'name': cp['name'] if cp else br['player'],
                'org': cp['org'] if cp else '',
                'war': round(c_war, 1),
                'seasons_scored': cw.get('seasons', 0),
                'model_rank': int(br['model_rank']),
                'actual_pick': int(br['actual_pick']),
                'drafted_by': cp['team_name'] if cp else '',
            }
        # open pool: the model's pick if it could take anyone in the clean pre-draft universe, each taken once
        open_choice, o_war = None, None
        if open_avail:
            orow = open_avail.pop(0)
            o_war = fnum(orow.get('war'))
            open_choice = {
                'pid': orow['pid'], 'name': orow.get('player') or '',
                'war': round(o_war, 1), 'was_drafted': int(orow['was_drafted']),
                'actual_pick': (int(orow['actual_pick']) if str(orow.get('actual_pick', '')).strip() not in ('', 'nan') else None),
                'source': orow.get('source', ''), 'model_rank': int(orow['model_rank']),
                'same_as_team': orow['pid'] == p['pid'],
            }
        sv = slot_value(pk)
        card = {
            'pick': pk, 'round': p['round'],
            'actual': {
                'pid': p['pid'], 'name': p['name'], 'org': p['org'],
                'war': round(a_war, 1),
                'seasons_scored': aw['seasons'] if aw else 0,
                'model_rank': int(a_brow['model_rank']) if a_brow else None,
                'games': round(aw['games'], 1) if aw else 0.0,
                'minutes': round(aw['minutes'], 1) if aw else 0.0,
            },
            'model_choice': model_choice,
            'best_hindsight': ({
                'pid': hind['pid'], 'name': hind['name'],
                'war': round(war_of(y, hind['pid']), 1),
                'actual_pick': hind['pick'], 'drafted_by': hind['team_name'],
            } if hind else None),
            'delta_war': round(delta, 1) if delta is not None else None,
            'verdict': verdict,
            'slot_value': round(sv, 2),
            'open_pool_choice': open_choice,
            '_raw': (a_war, c_war, sv, o_war),      # unrounded, for summary sums; removed below
        }
        cards_by_team[p['team_id']][y].append(card)
        audit.append({
            'slug': slug_of[p['team_id']], 'team': p['team_name'], 'abbr': p['abbr'],
            'year': y, 'pick': pk, 'round': p['round'],
            'actual_name': p['name'], 'actual_pid': p['pid'] or '',
            'actual_org': p['org'], 'actual_war': round(a_war, 3),
            'actual_seasons_scored': aw['seasons'] if aw else 0,
            'actual_model_rank': int(a_brow['model_rank']) if a_brow else '',
            'actual_on_board': a_brow is not None,
            'actual_join': 'year+actual_pick',
            'model_name': model_choice['name'] if model_choice else '',
            'model_pid': cpid if cp else '',
            'model_org': model_choice['org'] if model_choice else '',
            'model_war': round(c_war, 3) if cp else '',
            'model_seasons_scored': cw.get('seasons', 0) if cp else '',
            'model_rank': model_choice['model_rank'] if model_choice else '',
            'model_score': round(float(br['score']), 6) if cp else '',
            'model_actual_pick': model_choice['actual_pick'] if model_choice else '',
            'model_drafted_by': cp['abbr'] if cp else '',
            'hindsight_name': hind['name'] if hind else '',
            'hindsight_war': round(war_of(y, hind['pid']), 3) if hind else '',
            'hindsight_actual_pick': hind['pick'] if hind else '',
            'hindsight_drafted_by': hind['abbr'] if hind else '',
            'delta_war': round(delta, 3) if delta is not None else '',
            'verdict': verdict or '',
            'war_note': ('actual pick has no pre-draft/outcome record; WAR imputed 0'
                         if not p['pid'] else ''),
            'drafting_team_note': 'team that made the selection on draft night; '
                                  'draft-night trades are not reflected',
            'slot_value': round(sv, 3),
            'method': 'sequential',
            'open_pool_name': open_choice['name'] if open_choice else '',
            'open_pool_pid': open_choice['pid'] if open_choice else '',
            'open_pool_war': round(o_war, 3) if open_choice else '',
            'open_pool_was_drafted': open_choice['was_drafted'] if open_choice else '',
            'open_pool_actual_pick': open_choice['actual_pick'] if open_choice and open_choice['actual_pick'] else '',
        })

# ---------------------------------------------------------------- team summaries
r1 = lambda x: round(x, 1)


def summarize(cs):
    # Totals cover only cards where the model still had a player to take, so
    # actual / model / slot value are summed over the same set of slots.
    cs = [c for c in cs if c['model_choice'] is not None]
    a = sum(c['_raw'][0] for c in cs)
    m = sum(c['_raw'][1] for c in cs)
    sv = sum(c['_raw'][2] for c in cs)
    oc = [c for c in cs if c.get('open_pool_choice')]
    o = sum(c['_raw'][3] for c in oc)
    o_und = [c for c in oc if c['open_pool_choice']['was_drafted'] == 0]
    return {
        'open_pool_war_total': r1(o), 'open_pool_cards': len(oc),
        'open_pool_undrafted_picks': len(o_und),
        'open_pool_undrafted_war': r1(sum(c['_raw'][3] for c in o_und)),
        'war_actual_total': r1(a), 'war_model_total': r1(m),
        'model_better': sum(1 for c in cs if c['verdict'] == 'model'),
        'team_better': sum(1 for c in cs if c['verdict'] == 'team'),
        'ties': sum(1 for c in cs if c['verdict'] == 'tie'),
        'slot_value_total': r1(sv),
        'war_actual_surplus': r1(a - sv), 'war_model_surplus': r1(m - sv),
    }


team_out = {}
for tid, (city, name, abbr) in teams.items():
    drafts, allc = [], []
    for y in YEARS:
        cs = sorted(cards_by_team[tid][y], key=lambda c: c['pick'])
        drafts.append({'year': y, 'no_pick': not cs, 'picks': cs})
        allc += cs
    s = summarize(allc)
    n_picks = sum(1 for y in YEARS for p in picks[y] if p['team_id'] == tid)
    fr = summarize([c for c in allc if c['round'] == 1])
    summary = {'n_picks': n_picks,
               'n_cards': sum(1 for c in allc if c['model_choice'] is not None)}
    summary.update({k: s[k] for k in ('war_actual_total', 'war_model_total',
                                      'model_better', 'team_better', 'ties',
                                      'slot_value_total', 'war_actual_surplus',
                                      'war_model_surplus', 'open_pool_war_total',
                                      'open_pool_cards', 'open_pool_undrafted_picks',
                                      'open_pool_undrafted_war')})
    summary['first_round_only'] = {k: fr[k] for k in (
        'war_actual_total', 'war_model_total', 'model_better', 'team_better',
        'slot_value_total', 'war_actual_surplus', 'war_model_surplus')}
    team_out[tid] = {'slug': slug_of[tid], 'name': f'{city} {name}', 'abbr': abbr,
                     'team_id': tid, 'summary': summary, 'drafts': drafts}

for t in team_out.values():
    for d in t['drafts']:
        for c in d['picks']:
            del c['_raw']

out = {
    'generated': datetime.date.today().isoformat(),
    'method': METHOD,
    'open_pool': ('open_pool_choice per card: the strict engine ranking the clean pre-draft universe of that class '
                  '(early-entry list, Division-I seniors at 40%+ of team minutes, EuroLeague 22-year-olds; built '
                  '2026-10-20 with no outcome information), replayed in pick order with each player taken once. '
                  'An undrafted pick scores 0 WAR unless he later reached the NBA. Board: '
                  'data/open_pool_board_2019_2025.csv (tools/lab/export_open_pool_board.py).'),
    'boards_used': 'expanding',
    'war_definition': (
        'Realized WAR to date: y_early_war from the answers files, summed over the '
        "player's NBA seasons scored so far, at most five (2019-2021 classes have "
        'complete five-season windows; 2022-2025 are partial and still accruing, so '
        'compare within a draft year, not across). Players who never appeared in the '
        'NBA score 0. A handful of draft-and-stash picks have no pre-draft or outcome '
        'record in our data at all; they are shown with WAR 0 and flagged in '
        'team_cards_audit.csv (war_note).'),
    'model': {
        'label': 'EVO gen11 + international-line mask + NBADraft.net grades + '
                 'Torvik league context + ridge on top columns',
        'expanding_accuracy': 50.6, 'strict_accuracy': 41.8,
        'draft_accuracy_expanding': 24.0, 'draft_accuracy_strict': 26.3,
    },
    'teams': sorted(team_out.values(), key=lambda t: t['name']),
}
with open(OUT_JSON, 'w') as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
    f.write('\n')

cols = list(audit[0].keys())
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in sorted(audit, key=lambda r: (r['team'], r['year'], r['pick'])):
        w.writerow(r)

# ---------------------------------------------------------------- report
print(f'method                 : {METHOD}')
print(f'teams                  : {len(out["teams"])}')
print(f'total cards            : {len(audit)}')
print('cards per year         : ' + ', '.join(
    f'{y}: {sum(1 for r in audit if r["year"] == y)}' for y in YEARS))
dups = [k for k, v in choice_count.items() if v > 1]
print(f'players chosen twice   : {len(dups)} (must be 0)')
for k in ('actual_no_outcome_record', 'actual_no_answers_row', 'no_model_choice',
          'same_player', 'slug_generated'):
    if log[k]:
        print(f'  {k}: {len(log[k])}  {log[k][:8]}')

scored = [r for r in audit if r['model_war'] != '']
tot_a = sum(r['actual_war'] for r in scored)
tot_a_all = sum(r['actual_war'] for r in audit)
tot_m = sum(r['model_war'] for r in scored)
tot_sv = sum(r['slot_value'] for r in scored)
print(f'\nactual WAR incl. cards with no model pick: {tot_a_all:.1f}')
print(f'league WAR (cards with a model pick): actual {tot_a:.1f}   model {tot_m:.1f}   '
      f'slot value {tot_sv:.1f}   (model cards {len(scored)})')
vc = collections.Counter(r['verdict'] for r in audit)
print(f'verdicts               : model {vc["model"]}  team {vc["team"]}  tie {vc["tie"]}  '
      f'none {vc[""]}')
for rd in (1, 2):
    sub = [r for r in audit if r['round'] == rd]
    c = collections.Counter(r['verdict'] for r in sub)
    print(f'  round {rd} (n={len(sub)}): model {c["model"]}  team {c["team"]}  tie {c["tie"]}')
for t in out['teams']:
    if t['abbr'] == 'GSW':
        print('\nGolden State summary:')
        print(json.dumps(t['summary'], indent=1))
print(f'\nwrote {OUT_JSON}\nwrote {OUT_CSV}')
