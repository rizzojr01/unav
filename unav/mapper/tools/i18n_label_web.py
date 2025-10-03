# unav/mapper/tools/i18n_label_web.py
# -*- coding: utf-8 -*-
"""
UNav i18n Label Editor (Web) — Auto-derived structure with destinations

What’s new in this revision:
- FIX: Correct place/building/floor extraction in disk scan (was using wrong Path.parents indexes).
- Tree now renders DESTINATION display names using labels.json first (target lang -> English -> fallback).
- After saving labels or changing target language, the tree updates to show localized destination names.

Behavior summary
----------------
- Structure (Place → Building → Floor → Destination) is derived from data:
  A) Preferred: FacilityNavigator (floors that have boundaries.json).
  B) Fallback: disk scan + per-floor destinations.json.
- No manual structure editing. Only labeling (English + target language) and aliases.
- Labels are saved to: <DATA_FINAL_ROOT>/_i18n/labels.json

Run
---
  python -m unav.mapper.tools.i18n_label_web \
      --data-final-root /path/to/data \
      --use-nav \
      --host 127.0.0.1 --port 5009
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

from flask import Flask, request, jsonify, Response

# Optional UNav imports (only used in --use-nav mode)
try:
    from unav.config import UNavConfig  # type: ignore
    from unav.navigator.multifloor import FacilityNavigator  # type: ignore
except Exception:  # pragma: no cover
    UNavConfig = None  # type: ignore
    FacilityNavigator = None  # type: ignore

# Broad language list (ISO 639-1 / BCP47) for the target dropdown
LANG_AUTONYMS: Dict[str, str] = {
    "en": "English",
    "zh": "中文", "zh-Hans": "简体中文", "zh-Hant": "繁體中文",
    "ja": "日本語", "ko": "한국어", "th": "ไทย", "vi": "Tiếng Việt",
    "id": "Bahasa Indonesia", "ms": "Bahasa Melayu", "fil": "Filipino",
    "hi": "हिन्दी", "bn": "বাংলা", "ta": "தமிழ்", "te": "తెలుగు",
    "kn": "ಕನ್ನಡ", "ml": "മലയാളം", "mr": "मराठी", "gu": "ગુજરાતી",
    "pa": "ਪੰਜਾਬੀ", "ur": "اردو",
    "ar": "العربية", "fa": "فارسی", "tr": "Türkçe", "he": "עברית",
    "ru": "Русский", "uk": "Українська", "kk": "Қазақ тілі",
    "el": "Ελληνικά", "bg": "Български", "ro": "Română", "hu": "Magyar",
    "cs": "Čeština", "sk": "Slovenčina", "sl": "Slovenščina", "pl": "Polski",
    "de": "Deutsch", "fr": "Français", "es": "Español", "pt": "Português",
    "pt-BR": "Português (Brasil)", "it": "Italiano", "nl": "Nederlands",
    "sv": "Svenska", "no": "Norsk", "da": "Dansk", "fi": "Suomi",
    "et": "Eesti", "lv": "Latviešu", "lt": "Lietuvių",
    "sq": "Shqip", "sr": "Српски", "hr": "Hrvatski", "bs": "Bosanski", "mk": "Македонски",
    "sw": "Kiswahili", "am": "አማርኛ", "af": "Afrikaans", "zu": "isiZulu",
    "xh": "isiXhosa", "yo": "Yorùbá",
}

DEFAULT_EN = "en"
LABELS_REL = "_i18n/labels.json"

app = Flask(__name__)

# --------------------------- labels I/O ---------------------------

def load_labels(root: Path) -> Dict[str, Any]:
    p = root / LABELS_REL
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
    else:
        raw = {}
    data = {
        "places": raw.get("places", {}) if isinstance(raw.get("places", {}), dict) else {},
        "buildings": raw.get("buildings", {}) if isinstance(raw.get("buildings", {}), dict) else {},
        "floors": raw.get("floors", {}) if isinstance(raw.get("floors", {}), dict) else {},
        "destinations": raw.get("destinations", {}) if isinstance(raw.get("destinations", {}), dict) else {},
        "aliases": raw.get("aliases", {}) if isinstance(raw.get("aliases", {}), dict) else {},
    }
    save_labels(root, data)  # ensure file exists
    return data

def save_labels(root: Path, data: Dict[str, Any]) -> None:
    p = root / LABELS_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# --------------------------- structure derivation ---------------------------

def has_nav_assets(root: Path, place: str, building: str, floor: str) -> bool:
    """Minimal requirement for nav to load a floor (PathFinder needs boundaries.json)."""
    return (root / place / building / floor / "boundaries.json").exists()

def scan_floors_from_disk(root: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Build places mapping by scanning:
    - Any floor that has boundaries.json (for nav)
    - Or any floor that has destinations.json (for file fallback)
    Result: place -> building -> [floors]
    """
    places: Dict[str, Dict[str, List[str]]] = {}
    for floor_dir in root.glob("*/*/*/"):
        if not floor_dir.is_dir():
            continue
        # Correct extraction: place = parent.parent, building = parent, floor = name
        building_dir = floor_dir.parent
        place_dir = building_dir.parent
        place = place_dir.name
        building = building_dir.name
        floor = floor_dir.name
        if not ((floor_dir / "boundaries.json").exists() or (floor_dir / "destinations.json").exists()):
            continue
        bmap = places.setdefault(place, {})
        flist = bmap.setdefault(building, [])
        if floor not in flist:
            flist.append(floor)
    # Sort floors for better UI
    for bmap in places.values():
        for b in bmap:
            bmap[b].sort()
    return places

def derive_from_nav(root: Path) -> Tuple[Dict[str, Dict[str, List[str]]],
                                         Dict[str, Dict[str, Dict[str, List[Tuple[str, str]]]]]]:
    """
    Use FacilityNavigator to derive full structure and destinations.
    Returns:
      places: place -> building -> [floors]
      dests:  place -> building -> floor -> [ (dest_id, fallback_name) ]
    """
    if UNavConfig is None or FacilityNavigator is None:
        raise RuntimeError("UNav is not available for nav-based derivation.")

    # Whitelist floors that have nav assets
    whitelist: Dict[str, Dict[str, List[str]]] = {}
    for place, bmap in scan_floors_from_disk(root).items():
        for building, floors in bmap.items():
            usable = [f for f in floors if has_nav_assets(root, place, building, f)]
            if usable:
                whitelist.setdefault(place, {}).setdefault(building, []).extend(usable)

    if not whitelist:
        # Nothing to feed to nav
        return {}, {}

    cfg = UNavConfig(data_final_root=str(root), places=whitelist)
    nav = FacilityNavigator(cfg.navigator_config)

    # Derive places from nav.pf_map keys
    places: Dict[str, Dict[str, List[str]]] = {}
    for (place, building, floor) in nav.pf_map.keys():
        bmap = places.setdefault(place, {})
        flist = bmap.setdefault(building, [])
        if floor not in flist:
            flist.append(floor)
    for bmap in places.values():
        for b in bmap:
            bmap[b].sort()

    # Derive destinations from nav pf objects
    dests: Dict[str, Dict[str, Dict[str, List[Tuple[str, str]]]]] = {}
    for key, pf in nav.pf_map.items():
        place, building, floor = key
        rows = dests.setdefault(place, {}).setdefault(building, {}).setdefault(floor, [])
        for did in pf.dest_ids:
            did_str = str(did)
            fallback = pf.labels[did]
            rows.append((did_str, fallback))
    return places, dests

def derive_from_files(root: Path) -> Tuple[Dict[str, Dict[str, List[str]]],
                                           Dict[str, Dict[str, Dict[str, List[Tuple[str, str]]]]]]:
    """
    Fallback derivation from disk only:
      - Build places via scan_floors_from_disk()
      - For each floor, read destinations.json if present (list of {id, name})
    """
    places = scan_floors_from_disk(root)
    dests: Dict[str, Dict[str, Dict[str, List[Tuple[str, str]]]]] = {}
    for place, bmap in places.items():
        for building, floors in bmap.items():
            for floor in floors:
                floor_dir = root / place / building / floor
                rows: List[Tuple[str, str]] = []
                fp = floor_dir / "destinations.json"
                if fp.exists():
                    try:
                        arr = json.loads(fp.read_text(encoding="utf-8"))
                        if isinstance(arr, list):
                            for item in arr:
                                if not isinstance(item, dict):
                                    continue
                                did = str(item.get("id", ""))
                                name = str(item.get("name", did))
                                if did:
                                    rows.append((did, name))
                    except Exception:
                        pass
                dests.setdefault(place, {}).setdefault(building, {})[floor] = rows
    return places, dests

def build_tree(root: Path, use_nav: bool) -> Tuple[Dict[str, Dict[str, List[str]]],
                                                   Dict[str, Dict[str, Dict[str, List[Tuple[str, str]]]]],
                                                   bool]:
    """
    Returns (places, dests, used_nav)
    """
    if use_nav:
        try:
            places, dests = derive_from_nav(root)
            if places:  # nav succeeded with usable data
                return places, dests, True
        except Exception as e:
            print(f"[i18n-web] derive_from_nav failed: {e}. Falling back to files.")
    # Fallback
    places, dests = derive_from_files(root)
    return places, dests, False

# --------------------------- runtime config ---------------------------

DATA_ROOT: Path
USE_NAV: bool

# --------------------------- API ---------------------------

@app.get("/api/meta")
def api_meta() -> Response:
    langs = [{"code": c, "name": LANG_AUTONYMS.get(c, c)} for c in LANG_AUTONYMS.keys()]
    return jsonify({
        "data_root": str(DATA_ROOT),
        "use_nav_requested": USE_NAV,
        "default_en": DEFAULT_EN,
        "langs": langs,
    })

@app.get("/api/tree")
def api_tree() -> Response:
    places, dests, used_nav = build_tree(DATA_ROOT, USE_NAV)
    return jsonify({"places": places, "dests": dests, "used_nav": used_nav})

@app.get("/api/labels")
def api_labels_get() -> Response:
    return jsonify(load_labels(DATA_ROOT))

@app.post("/api/labels")
def api_labels_set() -> Response:
    """
    Body:
    {
      "section": "places|buildings|floors|destinations",
      "key": "New_York_City/LightHouse/6_floor/07993",
      "labels": {"en": "...", "<target>": "..."}
    }
    """
    payload = request.get_json(force=True)
    section = str(payload.get("section", ""))
    key = str(payload.get("key", ""))
    labels = payload.get("labels", {})
    if section not in ("places","buildings","floors","destinations"):
        return jsonify({"error": "Invalid section"}), 400
    if not key or not isinstance(labels, dict):
        return jsonify({"error": "Invalid key/labels"}), 400

    data = load_labels(DATA_ROOT)
    entry = data.setdefault(section, {}).setdefault(key, {})
    for lang, text in labels.items():
        text = (text or "").strip()
        if text:
            entry[lang] = text
        elif lang in entry:
            del entry[lang]
    save_labels(DATA_ROOT, data)
    return jsonify({"ok": True})

@app.get("/api/aliases")
def api_aliases_get() -> Response:
    lang = (request.args.get("lang") or DEFAULT_EN).strip()
    data = load_labels(DATA_ROOT)
    aliases = (data.get("aliases", {}) or {}).get(lang, {})
    return jsonify(aliases if isinstance(aliases, dict) else {})

@app.post("/api/aliases")
def api_aliases_set() -> Response:
    """
    Body:
    { "lang": "zh-Hant", "alias": "光明中心", "canonical": "New_York_City/LightHouse/6_floor or .../destId", "delete": false }
    """
    payload = request.get_json(force=True)
    lang = (payload.get("lang") or "").strip()
    alias = (payload.get("alias") or "").strip()
    canonical = (payload.get("canonical") or "").strip()
    to_delete = bool(payload.get("delete", False))
    if not lang or not alias:
        return jsonify({"error": "lang and alias required"}), 400

    data = load_labels(DATA_ROOT)
    amap = data.setdefault("aliases", {}).setdefault(lang, {})
    if to_delete:
        if alias in amap:
            del amap[alias]
    else:
        if not canonical:
            return jsonify({"error": "canonical required when not deleting"}), 400
        amap[alias] = canonical
    save_labels(DATA_ROOT, data)
    return jsonify({"ok": True})

# --------------------------- UI (single HTML) ---------------------------

INDEX_HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>UNav Language Label Tool</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root{--border:#e5e7eb;--text:#111827;--muted:#6b7280;--primary:#2563eb;--bg:#f9fafb}
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,'Noto Sans',sans-serif,'Apple Color Emoji','Segoe UI Emoji';
       margin:16px;color:var(--text);background:#f9fafb}
  h2{margin:0 0 12px 0}
  .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
  .wrap{display:flex;gap:12px}
  .left{width:32%;min-width:280px;max-height:74vh;overflow:auto;border:1px solid var(--border);background:#fff;border-radius:12px;padding:10px}
  .right{flex:1;min-width:340px}
  .card{border:1px solid var(--border);border-radius:12px;padding:12px;background:#fff}
  .muted{color:var(--muted);font-size:12px}
  select,input,button,textarea{font:inherit;padding:8px}
  button{border:1px solid var(--border);background:#fff;border-radius:8px;cursor:pointer}
  button.primary{background:var(--primary);color:#fff;border-color:var(--primary)}
  .tabs{display:flex;gap:8px;margin-bottom:8px}
  .tab{padding:8px 12px;border:1px solid var(--border);border-bottom:none;border-radius:10px 10px 0 0;background:#fff;cursor:pointer}
  .tab.active{background:#fff;font-weight:600;border-bottom:1px solid #fff}
  .panel{border:1px solid var(--border);border-radius:0 12px 12px 12px;background:#fff;padding:12px}
  .kv{display:grid;grid-template-columns:180px 1fr;gap:8px;align-items:center}
  .grid2{display:grid;grid-template-columns:repeat(2,minmax(200px,1fr));gap:12px}
  .ok{color:#059669}.err{color:#b91c1c}
  details{margin:2px 0}
  summary{cursor:pointer;padding:4px 6px;border-radius:6px}
  .tree-leaf{padding:2px 6px;margin:1px 0;border-radius:6px;cursor:pointer}
  .tree-leaf.active{background:#e0e7ff}

  /* Indentation for hierarchy (no JS changes needed) */
  details > summary { margin-left: 0; font-weight:600; }
  details > details > summary { margin-left: 12px; }
  details > details > details > summary { margin-left: 24px; }
  .tree-leaf { margin-left: 36px; }
</style>
<body>
  <h2>UNav Language Label Tool</h2>

  <div class="wrap">
    <div class="left" id="tree"></div>

    <div class="right">
      <div class="row" style="margin-bottom:8px">
        <div><span class="muted">DATA_ROOT:</span> <span id="droot"></span></div>
        <div><span class="muted">Engine:</span> <span id="engine"></span></div>

        <div class="row">
          <label>Target Language</label>
          <select id="lang"></select>
          <input id="customLang" placeholder="Custom code (e.g., zh-Hant)" style="min-width:200px">
          <button id="useCustom">Use</button>
          <button id="rescan">Rescan</button>
        </div>
      </div>

      <div class="tabs">
        <div class="tab active" data-tab="labels">Labels</div>
        <div class="tab" data-tab="aliases">Aliases</div>
      </div>
      <div class="panel" id="panel"></div>
    </div>
  </div>

<script>
const $ = s => document.querySelector(s);
const treeDiv = $("#tree"), panel = $("#panel"), langSel = $("#lang");
const customLang = $("#customLang"), useCustom = $("#useCustom");
const droot = $("#droot"), engine = $("#engine");
const rescanBtn = $("#rescan");

let META=null, TREE=null, LABELS=null;
let currentNode=null; // {prefix:'P|B|F|D', key:'...', fallback:'...'}
let TARGET_LANG=null;

const TABS = document.querySelectorAll('.tab');
TABS.forEach(t=>t.onclick=()=>{TABS.forEach(x=>x.classList.remove('active'));t.classList.add('active');renderPanel();});

async function get(u){ const r=await fetch(u); return r.json(); }
async function post(u,data){ const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}); return r.json(); }

async function boot(){
  META = await get('/api/meta');
  await reloadTreeAndLabels();

  droot.textContent = META.data_root;
  engine.textContent = TREE.used_nav ? 'nav' : 'files';

  // Build target language select (excluding English)
  const opts = META.langs.filter(x=>x.code!=="en").map(x=>`<option value="${x.code}">${x.name} (${x.code})</option>`).join('');
  langSel.innerHTML = opts;
  TARGET_LANG = (META.langs.find(x=>x.code==="zh") || META.langs[0] || {code:"en"}).code;
  if (TARGET_LANG === "en" && META.langs.length>1) TARGET_LANG = META.langs[1].code;
  langSel.value = TARGET_LANG;

  langSel.onchange = ()=>{ TARGET_LANG = langSel.value; buildTree(); selectNode(currentNode?.prefix||null, currentNode?.key||null, currentNode?.fallback||null); renderPanel(); };
  useCustom.onclick = ()=>{ const code=(customLang.value||"").trim(); if(!code) return; TARGET_LANG=code; langSel.value=""; buildTree(); selectNode(currentNode?.prefix||null, currentNode?.key||null, currentNode?.fallback||null); renderPanel(); };
  rescanBtn.onclick = async ()=>{ await reloadTreeAndLabels(); engine.textContent = TREE.used_nav ? 'nav' : 'files'; buildTree(); selectFirstLeaf(); renderPanel(); };

  buildTree();
  selectFirstLeaf();
  renderPanel();
}

async function reloadTreeAndLabels(){
  TREE = await get('/api/tree');
  LABELS = await get('/api/labels');
}

function getLabel(section,key,lang,fallback){
  const entry = ((LABELS[section]||{})[key]||{});
  return entry[lang] || (lang!=="en" ? (entry['en']||'') : '') || (fallback||'');
}

function destDisplayName(place, building, floor, destId, fallback){
  const key = `${place}/${building}/${floor}/${destId}`;
  const localized = getLabel('destinations', key, TARGET_LANG||'zh', fallback);
  // Decorate with ID for clarity
  return `Dest: ${destId} — ${localized}`;
}

function buildTree(){
  treeDiv.innerHTML = '';
  const places = TREE.places || {};
  const dests = TREE.dests || {};
  Object.keys(places).sort().forEach(place=>{
    const d1 = document.createElement('details'); d1.open=true;
    const placeName = getLabel('places', place, TARGET_LANG||'zh', place);
    d1.innerHTML = `<summary>Place: ${placeName} <span class="muted">[${place}]</span></summary>`;
    const bmap = places[place]||{};
    Object.keys(bmap).sort().forEach(b=>{
      const d2 = document.createElement('details'); d2.open=true;
      const bKey = `${place}/${b}`;
      const bName = getLabel('buildings', bKey, TARGET_LANG||'zh', b);
      d2.innerHTML = `<summary>Building: ${bName} <span class="muted">[${b}]</span></summary>`;
      (bmap[b]||[]).forEach(f=>{
        const d3 = document.createElement('details'); d3.open=true;
        const fKey = `${place}/${b}/${f}`;
        const fName = getLabel('floors', fKey, TARGET_LANG||'zh', f);
        d3.innerHTML = `<summary>Floor: ${fName} <span class="muted">[${f}]</span></summary>`;
        const rows = (((dests[place]||{})[b]||{})[f]||[]);
        rows.forEach(([id,name])=>{
          const a = document.createElement('div');
          a.className='tree-leaf'; a.dataset.node=`D::${place}/${b}/${f}/${id}`;
          a.textContent = destDisplayName(place,b,f,id,name);
          a.onclick = ()=>selectNode('D', `${place}/${b}/${f}/${id}`, id);
          d3.appendChild(a);
        });
        d2.appendChild(d3);
        d3.querySelector('summary').onclick=ev=>{
          ev.stopPropagation();
          selectNode('F', `${place}/${b}/${f}`, f);
        };
      });
      d1.appendChild(d2);
      d2.querySelector('summary').onclick=ev=>{
        ev.stopPropagation();
        selectNode('B', `${place}/${b}`, b);
      };
    });
    treeDiv.appendChild(d1);
    d1.querySelector('summary').onclick=ev=>{
      ev.stopPropagation();
      selectNode('P', place, place);
    };
  });
  // Re-highlight current dest if any
  if(currentNode && currentNode.prefix==='D'){
    const el = treeDiv.querySelector(`[data-node="D::${currentNode.key}"]`);
    if(el) el.classList.add('active');
  }
}

function clearActiveLeaves(){ treeDiv.querySelectorAll('.tree-leaf').forEach(el=>el.classList.remove('active')); }
function selectNode(prefix,key,fallback){
  if(!prefix || !key){ currentNode=null; return; }
  currentNode = {prefix,key,fallback};
  clearActiveLeaves();
  if(prefix==='D'){
    const el = treeDiv.querySelector(`[data-node="D::${key}"]`);
    if(el) el.classList.add('active');
  }
  renderPanel();
}
function selectFirstLeaf(){
  const firstDest = treeDiv.querySelector('.tree-leaf');
  if(firstDest){
    const node = firstDest.dataset.node;
    const [prefix,key] = node.split('::');
    const fallback = key.split('/').slice(-1)[0];
    selectNode(prefix,key,fallback);
    return;
  }
  const firstSum = treeDiv.querySelector('summary');
  if(firstSum){
    const placeId = (firstSum.querySelector('.muted')?.textContent||'').replace('[','').replace(']','');
    selectNode('P',placeId,placeId);
  }
}
function activeTab(){ const t=document.querySelector('.tab.active'); return t?t.dataset.tab:'labels'; }
function renderPanel(){ const tab=activeTab(); if(tab==='labels') renderLabelsTab(); else renderAliasesTab(); }

/* -------------------- Labels Tab (English + Target) -------------------- */
function pathInfo(){
  if(!currentNode) return null;
  const p = currentNode;
  let section=null, key=p.key, fallback=p.fallback;
  if(p.prefix==='P') section='places';
  else if(p.prefix==='B') section='buildings';
  else if(p.prefix==='F') section='floors';
  else if(p.prefix==='D') section='destinations';
  return {section,key,fallback};
}
function resolveLangName(code){
  const item = (META.langs||[]).find(x=>x.code===code);
  return item ? `${item.name} (${code})` : code;
}
function renderLabelsTab(){
  const info = pathInfo();
  if(!info){ panel.innerHTML = '<div class="muted">Select a node on the left.</div>'; return; }
  const tgt = TARGET_LANG || 'zh';
  const vEN = getLabel(info.section, info.key, 'en', info.fallback);
  const vTG = getLabel(info.section, info.key, tgt, info.fallback);
  panel.innerHTML = `
    <div class="kv" style="margin-bottom:12px">
      <div class="muted">Section</div><div>${info.section}</div>
      <div class="muted">Key</div><div>${info.key}</div>
      <div class="muted">Fallback</div><div>${info.fallback}</div>
    </div>
    <div class="grid2">
      <div class="col"><label>English (en)</label><input id="labEN" value="${vEN.replaceAll('"','&quot;')}"></div>
      <div class="col"><label>${resolveLangName(tgt)}</label><input id="labTG" value="${vTG.replaceAll('"','&quot;')}"></div>
    </div>
    <div class="row" style="margin-top:12px">
      <button id="copyFromEN">Copy EN → Target</button>
      <button id="save" class="primary">Save</button>
      <span id="saveMsg" class="muted"></span>
    </div>`;
  $("#copyFromEN").onclick = ()=>{ $("#labTG").value = $("#labEN").value; };
  $("#save").onclick = async ()=>{
    const labels = {"en": $("#labEN").value}; labels[tgt] = $("#labTG").value;
    const resp = await post('/api/labels', {section:info.section, key:info.key, labels});
    $("#saveMsg").textContent = resp.ok ? 'Saved.' : ('Error: ' + (resp.error||'')); 
    $("#saveMsg").className = resp.ok ? 'ok' : 'err';
    LABELS = await get('/api/labels');         // refresh labels cache
    buildTree();                               // refresh tree to reflect localized destination names
    // Re-select the same node if possible
    selectNode(info.section==='destinations'?'D':(info.section==='floors'?'F':(info.section==='buildings'?'B':'P')), info.key, info.fallback);
  };
}

/* -------------------- Aliases Tab (bound to target lang) -------------------- */
async function renderAliasesTab(){
  const lang = TARGET_LANG || 'zh';
  const aliases = await get('/api/aliases?lang=' + encodeURIComponent(lang));
  const rows = Object.entries(aliases).sort((a,b)=>a[0].localeCompare(b[0]))
    .map(([a,c])=>`<tr><td>${a}</td><td>${c}</td><td><button data-del="${a}">Delete</button></td></tr>`).join('');
  panel.innerHTML = `
    <div class="row" style="margin-bottom:8px">
      <div class="muted">Language:</div><div>${resolveLangName(lang)}</div>
    </div>
    <div class="card" style="margin-bottom:12px">
      <div class="kv">
        <label>Alias text</label><input id="aliasText">
        <label>Canonical ID</label><input id="aliasCanon" placeholder="e.g. New_York_City/LightHouse/6_floor or .../destId">
      </div>
      <div class="row" style="margin-top:8px">
        <button id="aliasSave" class="primary">Add / Update</button>
        <span id="aliasMsg" class="muted"></span>
      </div>
    </div>
    <div class="card">
      <table style="width:100%;border-collapse:collapse">
        <thead><tr><th style="text-align:left">Alias</th><th style="text-align:left">Canonical</th><th></th></tr></thead>
        <tbody>${rows || '<tr><td colspan="3" class="muted">No aliases yet.</td></tr>'}</tbody>
      </table>
    </div>`;
  $("#aliasSave").onclick = async ()=>{
    const alias = ($("#aliasText").value||'').trim();
    const canonical = ($("#aliasCanon").value||'').trim();
    if(!alias || !canonical){ $("#aliasMsg").textContent='Both fields are required.'; $("#aliasMsg").className='err'; return; }
    const resp = await post('/api/aliases', {lang, alias, canonical});
    $("#aliasMsg").textContent = resp.ok ? 'Saved.' : ('Error: '+(resp.error||'')); 
    $("#aliasMsg").className = resp.ok ? 'ok' : 'err';
    renderAliasesTab();
  };
  panel.querySelectorAll('button[data-del]').forEach(btn=>{
    btn.onclick = async ()=>{
      const alias = btn.getAttribute('data-del');
      await post('/api/aliases', {lang, alias, delete:true});
      renderAliasesTab();
    };
  });
}

boot();
</script>
</body>
</html>
"""


@app.get("/")
def index() -> Response:
    return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")

# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UNav i18n web label editor (auto-derived)")
    p.add_argument("--data-final-root", required=True, help="Path to DATA_FINAL_ROOT")
    p.add_argument("--use-nav", action="store_true", help="Prefer FacilityNavigator for destinations")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5009)
    return p.parse_args()

def main() -> int:
    global DATA_ROOT, USE_NAV
    ns = parse_args()
    DATA_ROOT = Path(ns.data_final_root).resolve()
    USE_NAV = bool(ns.use_nav)
    # Ensure labels.json exists
    load_labels(DATA_ROOT)
    app.run(host=ns.host, port=ns.port, debug=False)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
