"""
BoxScan — code.py
Portal → Medición → Optimización
Versión actualizada:
- optimización multi-vehículo
- todas las entregas obligatorias
- mapa por calles reales (Directions)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import cv2
import numpy as np
import threading
import time
import csv
import json
import os
import requests
from math import dist
from datetime import datetime, timezone

from optimizer import run_full_optimization

STREAM_URL = 0
ARUCO_REAL_SIZE_CM = 5.0
MIN_BOX_AREA = 5000
CSV_FILE = "boxes.csv"
JSONL_FILE = "boxes.jsonl"
GOOGLE_API_KEY = "XXXXXXXX" #Enter your Google Maps API. Charges will be applied.

latest_result = {
    "status": "starting",
    "length_cm": None,
    "width_cm": None,
    "timestamp": None,
    "marker_id": None,
}
latest_frame = None
lock = threading.Lock()
pending_box = {}


def geocode_address(address: str) -> dict:
    try:
        if not GOOGLE_API_KEY:
            return {"ok": False, "error": "GOOGLE_API_KEY vacía o no configurada"}

        r = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={
                "address": f"{address}, Medellín, Colombia",
                "key": GOOGLE_API_KEY,
                "language": "es",
                "region": "co",
            },
            timeout=10
        )

        data = r.json()
        status = data.get("status", "UNKNOWN")

        if status != "OK":
            return {
                "ok": False,
                "error": f"Geocoding falló: {status}",
                "google_response": data
            }

        res = data["results"][0]
        loc = res["geometry"]["location"]
        lat, lng = round(loc["lat"], 7), round(loc["lng"], 7)

        neighborhood = None
        locality = None
        for comp in res.get("address_components", []):
            if "neighborhood" in comp["types"] or "sublocality_level_1" in comp["types"]:
                neighborhood = comp["long_name"]
            if "locality" in comp["types"]:
                locality = comp["long_name"]

        er = requests.get(
            "https://maps.googleapis.com/maps/api/elevation/json",
            params={"locations": f"{lat},{lng}", "key": GOOGLE_API_KEY},
            timeout=10
        )
        edata = er.json()
        elev = round(edata["results"][0]["elevation"], 1) if edata.get("status") == "OK" else None

        return {
            "ok": True,
            "formatted_address": res["formatted_address"],
            "neighborhood": neighborhood or "Desconocido",
            "locality": locality or "Medellín",
            "lat": lat,
            "lng": lng,
            "elevation_m": elev
        }

    except Exception as e:
        return {"ok": False, "error": f"Excepción en geocode: {e}"}


HEADERS = [
    "box_id", "timestamp", "input_address", "formatted_address", "neighborhood",
    "locality", "lat", "lng", "elevation_m", "cargo_type", "length_cm", "width_cm",
    "height_cm", "weight_kg", "volume_cm3", "volume_m3", "density_kg_m3", "source", "notes"
]

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADERS)

def next_box_id() -> str:
    init_csv()
    try:
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            if len(rows) <= 1:
                return "BX-000001"
            n = int(rows[-1][0].replace("BX-", "")) + 1
            return f"BX-{n:06d}"
    except Exception:
        return "BX-000001"

def save_record(record: dict):
    init_csv()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([record.get(h, "") for h in HEADERS])

    with open(JSONL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def get_recent_boxes_json(limit=500) -> list:
    try:
        with open(JSONL_FILE, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
            return rows[-limit:][::-1]
    except Exception as e:
        print("JSONL read error:", e)
        return []


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype="float32")

def detect_aruco_scale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters(),
    )
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(corners) == 0:
        return None, None, None

    mc = order_points(corners[0][0])
    px = (dist(mc[0], mc[1]) + dist(mc[1], mc[2])) / 2.0
    return px / ARUCO_REAL_SIZE_CM, mc, int(ids[0][0])

def find_box_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 50, 150)
    edges = cv2.erode(cv2.dilate(edges, None, iterations=2), None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_BOX_AREA:
            continue
        poly = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(poly) == 4:
            candidates.append((area, poly))

    if not candidates:
        return None

    return order_points(sorted(candidates, reverse=True, key=lambda x: x[0])[0][1].reshape(4, 2))

def draw_annotated(image, marker_pts, box_pts, length_cm, width_cm):
    out = image.copy()

    if marker_pts is not None:
        cv2.polylines(out, [marker_pts.astype(int)], True, (255, 180, 0), 2)
        cv2.putText(out, "ArUco", tuple(marker_pts.astype(int)[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)

    if box_pts is not None:
        cv2.polylines(out, [box_pts.astype(int)], True, (0, 220, 120), 3)
        tl, tr, br, bl = box_pts
        mt = ((tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5)
        mr = ((tr[0] + br[0]) * 0.5, (tr[1] + br[1]) * 0.5)

        cv2.putText(out, f"L:{length_cm:.1f}cm", (int(mt[0]) - 60, int(mt[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 120), 2)
        cv2.putText(out, f"W:{width_cm:.1f}cm", (int(mr[0]) + 10, int(mr[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 120), 2)

    return out


def process_stream():
    global latest_result, latest_frame

    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_DSHOW)
    if not cap.isOpened():
        with lock:
            latest_result["status"] = "error_opening_stream"
        return

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            with lock:
                latest_result["status"] = "stream_read_error"
            time.sleep(0.2)
            continue

        ppc, marker_pts, marker_id = detect_aruco_scale(frame)
        l = w = None
        status = "no_marker"
        box_pts = None

        if ppc:
            box_pts = find_box_contour(frame)
            if box_pts is not None:
                tl, tr, br, bl = box_pts
                l = dist(tl, tr) / ppc
                w = dist(tr, br) / ppc
                status = "ok"
            else:
                status = "marker_no_box"

        ann = draw_annotated(frame, marker_pts, box_pts, l or 0, w or 0)

        with lock:
            latest_result = {
                "status": status,
                "length_cm": round(l, 2) if l else None,
                "width_cm": round(w, 2) if w else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "marker_id": marker_id,
            }
            latest_frame = ann

        time.sleep(0.03)

def mjpeg_gen():
    while True:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.1)
            continue

        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        time.sleep(0.03)


BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0c0e14;--surface:#13151f;--surface2:#1a1d2a;--border:#252838;
      --accent:#00e5a0;--accent2:#ff6b35;--text:#e8eaf0;--muted:#5a6075;
      --danger:#ff3860;--warn:#ffd60a;}
html,body{min-height:100%;background:var(--bg);color:var(--text);font-family:'Syne',sans-serif}
nav{background:var(--surface);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100}
.nav-inner{max-width:1280px;margin:0 auto;padding:0 24px;height:60px;display:flex;align-items:center;justify-content:space-between}
.nav-brand{font-weight:800;font-size:18px;letter-spacing:-.5px}
.nav-steps{display:flex;align-items:center}
.step{display:flex;align-items:center;gap:8px;padding:8px 16px;border-radius:8px;text-decoration:none;color:var(--muted);font-size:13px;font-weight:600;transition:all .2s}
.step:hover{color:var(--text)}
.step.active{color:var(--accent)}
.step-num{width:22px;height:22px;border-radius:50%;border:1.5px solid currentColor;display:flex;align-items:center;justify-content:center;font-family:'DM Mono',monospace;font-size:11px}
.step.active .step-num{background:var(--accent);border-color:var(--accent);color:#000}
.step-line{width:28px;height:1px;background:var(--border)}
main{max-width:1280px;margin:0 auto;padding:32px 24px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:28px}
h1{font-size:26px;font-weight:800;letter-spacing:-.5px;margin-bottom:4px}
h2{font-size:18px;font-weight:700;margin-bottom:18px}
.sub{color:var(--muted);font-size:13px;margin-bottom:24px}
label{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;display:block;margin-bottom:6px}
input,select,textarea{width:100%;padding:12px 15px;background:var(--surface2);border:1px solid var(--border);border-radius:10px;color:var(--text);font-family:'Syne',sans-serif;font-size:14px;outline:none;transition:border .2s}
input:focus,select:focus,textarea:focus{border-color:var(--accent)}
select option{background:var(--surface2)}
.btn{padding:12px 24px;border:none;border-radius:10px;font-family:'Syne',sans-serif;font-size:14px;font-weight:700;cursor:pointer;transition:all .2s}
.btn-primary{background:var(--accent);color:#000}
.btn-primary:hover{background:#00ffb3;transform:translateY(-1px)}
.btn-secondary{background:var(--surface2);color:var(--text);border:1px solid var(--border)}
.btn-secondary:hover{border-color:var(--accent);color:var(--accent)}
.btn-full{width:100%}
.field{margin-bottom:16px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
.metric{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:14px}
.metric.lit{border-color:var(--accent);background:rgba(0,229,160,.05)}
.m-lbl{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:6px}
.m-val{font-size:22px;font-weight:800;color:var(--accent)}
.m-val.empty{color:var(--muted);font-size:18px}
.m-val.ok{color:var(--accent)}
.m-val.err{color:var(--danger)}
.m-val.warn{color:var(--warn)}
.tag{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;font-family:'DM Mono',monospace}
.tag-ok{background:rgba(0,229,160,.15);color:var(--accent)}
.tag-err{background:rgba(255,56,96,.15);color:var(--danger)}
.tag-warn{background:rgba(255,214,10,.15);color:var(--warn)}
.alert{padding:13px 16px;border-radius:10px;font-size:13px;margin-top:10px}
.alert-ok{background:rgba(0,229,160,.08);border:1px solid rgba(0,229,160,.3);color:var(--accent)}
.alert-err{background:rgba(255,56,96,.08);border:1px solid rgba(255,56,96,.3);color:var(--danger)}
.alert-info{background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.3);color:#a5b4fc}
.box-row{display:flex;align-items:flex-start;gap:12px;padding:12px;border:1px solid var(--border);border-radius:10px;margin-bottom:8px}
.order-num{width:28px;height:28px;border-radius:50%;background:var(--accent);color:#000;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.vehicle-block{margin-bottom:18px;padding:14px;border:1px solid var(--border);border-radius:12px;background:var(--surface2)}
.vehicle-title{font-weight:800;margin-bottom:8px}
.small{font-size:12px;color:var(--muted)}
"""

def nav(active=1):
    def cls(n):
        return "active" if n == active else ""
    return f"""<nav><div class="nav-inner">
  <div class="nav-brand">📦 BoxScan</div>
  <div class="nav-steps">
    <a href="/" class="step {cls(1)}"><span class="step-num">1</span> Portal</a>
    <div class="step-line"></div>
    <a href="/scan" class="step {cls(2)}"><span class="step-num">2</span> Medición</a>
    <div class="step-line"></div>
    <a href="/optimize" class="step {cls(3)}"><span class="step-num">3</span> Optimización</a>
  </div>
</div></nav>"""

PAGE_PORTAL = """<!DOCTYPE html><html lang="es"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BoxScan — Portal</title><style>CSS_HERE</style></head><body>
NAV_HERE
<main>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px">
<div class="card">
  <h1>Registro de caja</h1>
  <p class="sub">Ingresa los datos de entrega. El Box ID se genera automáticamente como primary key.</p>
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:24px;padding:16px;background:var(--surface2);border-radius:12px;border:1px solid var(--border)">
    <div><div class="m-lbl">Box ID — Primary Key</div>
    <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:800;color:var(--accent)" id="boxIdDisplay">...</div></div>
    <div style="margin-left:auto"><span class="tag tag-ok">AUTO</span></div>
  </div>
  <form id="portalForm">
    <div class="field">
      <label>Dirección de entrega en Medellín</label>
      <div style="display:flex;gap:8px">
        <input type="text" id="address" placeholder="Calle 10 #43-25" required/>
        <button type="button" class="btn btn-secondary" onclick="validateAddress()" style="white-space:nowrap">🗺 Validar</button>
      </div>
    </div>
    <div id="geoPanel" style="display:none;margin-bottom:16px;padding:16px;background:rgba(0,229,160,.06);border:1px solid rgba(0,229,160,.25);border-radius:12px">
      <div style="font-weight:700;font-size:14px;margin-bottom:3px" id="geoFormatted"></div>
      <div style="font-family:'DM Mono',monospace;font-size:11px;color:var(--muted);margin-bottom:12px" id="geoNeighborhood"></div>
      <div class="grid3">
        <div class="metric"><div class="m-lbl">Latitud</div><div style="font-family:'DM Mono',monospace;font-size:13px;font-weight:700" id="geoLat">-</div></div>
        <div class="metric"><div class="m-lbl">Longitud</div><div style="font-family:'DM Mono',monospace;font-size:13px;font-weight:700" id="geoLng">-</div></div>
        <div class="metric"><div class="m-lbl">Elevación</div><div style="font-family:'DM Mono',monospace;font-size:13px;font-weight:700" id="geoElev">-</div></div>
      </div>
      <a id="mapsLink" href="#" target="_blank" style="display:inline-block;margin-top:10px;font-size:12px;color:var(--accent);text-decoration:none">Ver en Google Maps →</a>
    </div>
    <div class="grid2" style="margin-bottom:16px">
      <div class="field" style="margin:0"><label>Tipo de carga</label>
        <select id="cargo_type" required><option value="">Seleccionar...</option>
          <option>Electronics</option><option>Documents</option><option>Clothing and textiles</option>
          <option>Footwear</option><option>Food dry goods</option><option>Beverages</option>
          <option>Pharmaceuticals</option><option>Cosmetics and personal care</option>
          <option>Home goods</option><option>Auto parts</option><option>Hardware and tools</option>
          <option>Industrial supplies</option><option>Fragile items</option><option>Other</option>
        </select></div>
      <div class="field" style="margin:0"><label>Peso (kg)</label>
        <input type="number" id="weight_kg" step="0.01" min="0.01" placeholder="0.00" required/></div>
    </div>
    <div class="field"><label>Notas</label><textarea id="notes" rows="3" placeholder="Observaciones..."></textarea></div>
    <div id="portalAlert"></div>
    <button type="submit" class="btn btn-primary btn-full" style="margin-top:8px">Continuar a medición →</button>
  </form>
</div>
<div class="card">
  <h2>Últimas cajas registradas</h2>
  <div id="recentBoxes"><p style="color:var(--muted);font-size:14px">Cargando...</p></div>
</div>
</div></main>
<script>
let geoData=null,currentBoxId=null;
async function loadBoxId(){const r=await fetch('/next_box_id');const d=await r.json();currentBoxId=d.box_id;document.getElementById('boxIdDisplay').textContent=d.box_id;}
async function loadRecent(){const r=await fetch('/recent_boxes');const boxes=await r.json();const el=document.getElementById('recentBoxes');
if(!boxes.length){el.innerHTML='<p style="color:var(--muted);font-size:14px">No hay cajas registradas aún.</p>';return;}
el.innerHTML=boxes.map(b=>`<div style="padding:12px;border:1px solid var(--border);border-radius:10px;margin-bottom:8px">
<div style="display:flex;justify-content:space-between;margin-bottom:5px">
<span style="font-family:'DM Mono',monospace;font-size:13px;font-weight:700;color:var(--accent)">${b.box_id}</span>
<span style="font-size:11px;color:var(--muted)">${(b.timestamp||'').slice(0,16).replace('T',' ')}</span></div>
<div style="font-size:13px;margin-bottom:3px">${b.formatted_address||b.input_address||'-'}</div>
<div style="font-size:11px;color:var(--muted)">${b.cargo_type||'-'} · ${b.weight_kg?b.weight_kg+' kg':'-'} · ${b.length_cm?b.length_cm+'×'+b.width_cm+'×'+b.height_cm+' cm':'Sin medir'}</div>
</div>`).join('');}
async function validateAddress(){const addr=document.getElementById('address').value.trim();
if(!addr){alert('Escribe una dirección primero');return;}
document.getElementById('portalAlert').innerHTML='<div class="alert alert-info">🗺 Validando...</div>';
const r=await fetch('/geocode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({address:addr})});
const d=await r.json();
if(!r.ok){document.getElementById('portalAlert').innerHTML='<div class="alert alert-err">⚠️ '+(d.detail||'No se pudo validar')+'</div>';return;}
geoData=d;
document.getElementById('geoFormatted').textContent=d.formatted_address;
document.getElementById('geoNeighborhood').textContent=d.neighborhood+' · '+d.locality;
document.getElementById('geoLat').textContent=d.lat;
document.getElementById('geoLng').textContent=d.lng;
document.getElementById('geoElev').textContent=(d.elevation_m??'-')+' m.s.n.m.';
document.getElementById('geoPanel').style.display='block';
document.getElementById('mapsLink').href=`https://www.google.com/maps?q=${d.lat},${d.lng}`;
document.getElementById('portalAlert').innerHTML='<div class="alert alert-ok">✅ Dirección validada</div>';}
document.getElementById('portalForm').addEventListener('submit',async e=>{e.preventDefault();
if(!geoData){document.getElementById('portalAlert').innerHTML='<div class="alert alert-err">⚠️ Valida la dirección primero</div>';return;}
const payload={box_id:currentBoxId,address:document.getElementById('address').value,cargo_type:document.getElementById('cargo_type').value,weight_kg:parseFloat(document.getElementById('weight_kg').value),notes:document.getElementById('notes').value,geo:geoData};
const r=await fetch('/start_box',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
const d=await r.json();if(!r.ok){document.getElementById('portalAlert').innerHTML='<div class="alert alert-err">'+(d.detail||'Error')+'</div>';return;}
window.location.href='/scan?box_id='+currentBoxId;});
loadBoxId();loadRecent();
</script></body></html>"""

PAGE_SCAN = """<!DOCTYPE html><html lang="es"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BoxScan — Medición</title><style>CSS_HERE</style></head><body>
NAV_HERE
<main>
<div style="display:grid;grid-template-columns:1.5fr 1fr;gap:24px">
<div class="card">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
    <div><h2 style="margin:0">Medición automática</h2>
    <p style="color:var(--muted);font-size:13px">Muestra la caja con el marcador ArUco a la cámara</p></div>
    <div style="margin-left:auto;text-align:right">
      <div class="m-lbl">Box ID</div>
      <div style="font-family:'DM Mono',monospace;font-size:16px;font-weight:800;color:var(--accent)" id="currentBoxId">-</div>
    </div>
  </div>
  <img src="/video" style="width:100%;border-radius:12px;background:#0a0a14;min-height:360px;object-fit:cover"/>
  <div class="grid2" style="margin-top:14px">
    <div class="metric"><div class="m-lbl">Status</div><div class="m-val empty" id="status">starting</div></div>
    <div class="metric"><div class="m-lbl">Marker ID</div><div class="m-val empty" id="marker_id">-</div></div>
  </div>
  <div style="margin-top:12px;padding:12px;background:var(--surface2);border-radius:10px;font-size:12px;color:var(--muted);line-height:1.8">
    <b style="color:var(--text)">Instrucciones:</b> Imprime el marcador ArUco a <b>5×5 cm</b> · Colócalo junto a la caja · Cuando status sea <span class="tag tag-ok">ok</span> durante 5 frames, el botón se activa
  </div>
</div>
<div class="card">
  <h2>Dimensiones detectadas</h2>
  <div class="grid3" style="margin-bottom:18px">
    <div class="metric" id="cardL"><div class="m-lbl">Largo</div><div class="m-val empty" id="length_cm">—</div><div style="font-size:10px;color:var(--muted);margin-top:3px">cm</div></div>
    <div class="metric" id="cardW"><div class="m-lbl">Ancho</div><div class="m-val empty" id="width_cm">—</div><div style="font-size:10px;color:var(--muted);margin-top:3px">cm</div></div>
    <div class="metric" id="cardH"><div class="m-lbl">Alto est.</div><div class="m-val empty" id="height_cm">—</div><div style="font-size:10px;color:var(--muted);margin-top:3px">cm</div></div>
  </div>
  <div class="grid2" style="margin-bottom:18px">
    <div class="metric"><div class="m-lbl">Volumen</div><div class="m-val empty" id="volume_m3">—</div><div style="font-size:10px;color:var(--muted);margin-top:3px">m³</div></div>
    <div class="metric"><div class="m-lbl">Densidad</div><div class="m-val empty" id="density">—</div><div style="font-size:10px;color:var(--muted);margin-top:3px">kg/m³</div></div>
  </div>
  <div style="border:1px solid var(--border);border-radius:12px;padding:14px;margin-bottom:18px;font-size:13px" id="boxSummary">
    <p style="color:var(--muted)">Cargando datos del registro...</p>
  </div>
  <div id="scanAlert"></div>
  <button class="btn btn-primary btn-full" id="confirmBtn" onclick="confirmScan()" disabled style="margin-top:8px">⏳ Esperando medición estable...</button>
  <button class="btn btn-secondary btn-full" onclick="window.location.href='/'" style="margin-top:10px">← Volver al portal</button>
</div>
</div></main>
<script>
const params=new URLSearchParams(window.location.search);
const boxId=params.get('box_id');
let latest=null,pending=null,stableCount=0;
document.getElementById('currentBoxId').textContent=boxId||'-';
async function loadPending(){if(!boxId)return;const r=await fetch('/pending_box/'+boxId);if(!r.ok)return;pending=await r.json();
document.getElementById('boxSummary').innerHTML=`<div style="display:grid;gap:5px">
<div><b>Destino:</b> ${pending.formatted_address||pending.address}</div>
<div><b>Barrio:</b> ${pending.neighborhood||'-'} ${pending.elevation_m?'· '+pending.elevation_m+' m.s.n.m.':''}</div>
<div><b>Cargo:</b> ${pending.cargo_type} · <b>Peso:</b> ${pending.weight_kg} kg</div></div>`;}
async function fetchLatest(){try{const d=await(await fetch('/latest')).json();latest=d;
const st=d.status;
document.getElementById('status').textContent=st;
document.getElementById('status').className='m-val '+(st==='ok'?'ok':st==='starting'?'warn':'err');
document.getElementById('marker_id').textContent=d.marker_id??'-';
document.getElementById('marker_id').className='m-val '+(d.marker_id!==null?'ok':'empty');
const l=d.length_cm,w=d.width_cm,h=(l&&w)?Math.min(l,w).toFixed(1):null;
const setD=(id,cid,val)=>{const e=document.getElementById(id);e.textContent=val??'—';e.className='m-val '+(val?'ok':'empty');document.getElementById(cid).className='metric'+(val?' lit':'');};
setD('length_cm','cardL',l?l.toFixed(1):null);setD('width_cm','cardW',w?w.toFixed(1):null);setD('height_cm','cardH',h);
if(l&&w&&h&&pending){const vol=l*w*parseFloat(h)/1e6;
document.getElementById('volume_m3').textContent=vol.toFixed(6);document.getElementById('volume_m3').className='m-val ok';
document.getElementById('density').textContent=(pending.weight_kg/vol).toFixed(1);document.getElementById('density').className='m-val ok';}
if(st==='ok'){stableCount++;if(stableCount>=5){const btn=document.getElementById('confirmBtn');btn.disabled=false;btn.textContent='✅ Confirmar y guardar';}}
else stableCount=0;}catch(e){}}
async function confirmScan(){if(!latest||latest.status!=='ok'){document.getElementById('scanAlert').innerHTML='<div class="alert alert-err">⚠️ No hay medición válida</div>';return;}
const r=await fetch('/confirm_scan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({box_id:boxId})});
const d=await r.json();
if(!r.ok){document.getElementById('scanAlert').innerHTML='<div class="alert alert-err">'+(d.detail||'Error')+'</div>';return;}
document.getElementById('scanAlert').innerHTML='<div class="alert alert-ok">✅ Caja guardada correctamente</div>';
setTimeout(()=>window.location.href='/optimize',900);}
loadPending();setInterval(fetchLatest,700);
</script></body></html>"""

PAGE_OPTIMIZE = """<!DOCTYPE html><html lang="es"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BoxScan — Optimización</title><style>CSS_HERE</style></head><body>
NAV_HERE
<main>
<div style="display:grid;grid-template-columns:1.25fr .75fr;gap:24px">
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px">
    <div>
      <h1 style="margin-bottom:6px">Optimización de entregas</h1>
      <p class="sub" style="margin:0">Todas las entregas asignadas en múltiples vehículos, con trazado por calles y evaluación física de subida</p>
    </div>
    <button class="btn btn-primary" onclick="runOptimization()">▶ Ejecutar optimización</button>
  </div>

  <div class="grid3" style="margin-bottom:18px">
    <div class="metric"><div class="m-lbl">Entregas</div><div class="m-val" id="totalDel">0</div></div>
    <div class="metric"><div class="m-lbl">Peso total</div><div class="m-val" id="totalW">0</div></div>
    <div class="metric"><div class="m-lbl">Vehículos usados</div><div class="m-val" id="totalVeh">0</div></div>
  </div>

  <div id="map" style="width:100%;height:520px;border-radius:14px;border:1px solid var(--border);overflow:hidden;background:#111"></div>
</div>

<div class="card">
  <h2>Plan recomendado</h2>
  <div id="summaryBox" style="font-size:13px;color:var(--muted);margin-bottom:16px">
    Ejecuta la optimización para ver el resumen.
  </div>
  <div id="routeList"></div>
</div>
</div></main>

<script>
let map, markers = [], deliveries = [];
let directionsService;
let directionsRenderers = [];

function initMap(){
  map = new google.maps.Map(document.getElementById('map'), {
    center:{lat:6.2442,lng:-75.5812},
    zoom:12,
    styles:[
      {elementType:'geometry',stylers:[{color:'#13151f'}]},
      {elementType:'labels.text.fill',stylers:[{color:'#d5d9e5'}]},
      {elementType:'labels.text.stroke',stylers:[{color:'#1a1d2a'}]},
      {featureType:'road',elementType:'geometry',stylers:[{color:'#252838'}]},
      {featureType:'water',elementType:'geometry',stylers:[{color:'#0c0e14'}]}
    ]
  });
  directionsService = new google.maps.DirectionsService();
  loadDeliveries();
}

async function loadDeliveries(){
  const r = await fetch('/recent_boxes_json?limit=200');
  deliveries = (await r.json()).filter(b => b.lat && b.lng);
  document.getElementById('totalDel').textContent = deliveries.length;
  document.getElementById('totalW').textContent = deliveries.reduce((s,b)=>s+(parseFloat(b.weight_kg)||0),0).toFixed(1);
}

function clearMap(){
  markers.forEach(m => m.setMap(null));
  markers = [];
  directionsRenderers.forEach(r => r.setMap(null));
  directionsRenderers = [];
}

function rendererColor(i){
  const colors = ['#00e5a0','#ffd60a','#4cc9f0','#ff6b35','#c77dff','#f72585'];
  return colors[i % colors.length];
}

async function drawRouteChunks(points, color){
  if (!points || points.length < 2) return;

  const maxPointsPerRequest = 25;
  let start = 0;

  while (start < points.length - 1) {
    const end = Math.min(start + maxPointsPerRequest - 1, points.length - 1);
    const segment = points.slice(start, end + 1);

    const origin = segment[0];
    const destination = segment[segment.length - 1];
    const waypoints = segment.slice(1, -1).map(p => ({ location: p, stopover: true }));

    const renderer = new google.maps.DirectionsRenderer({
      suppressMarkers: true,
      preserveViewport: true,
      polylineOptions: {
        strokeColor: color,
        strokeOpacity: 0.82,
        strokeWeight: 5
      }
    });
    renderer.setMap(map);
    directionsRenderers.push(renderer);

    await new Promise((resolve) => {
      directionsService.route({
        origin,
        destination,
        waypoints,
        optimizeWaypoints: false,
        travelMode: google.maps.TravelMode.DRIVING
      }, (result, status) => {
        if (status === 'OK') {
          renderer.setDirections(result);
        } else {
          console.error('Directions error:', status);
        }
        resolve();
      });
    });

    start = end;
  }
}

function slopeLabelClass(v){
  if(v === 'requiere_punto_alterno') return 'tag-err';
  if(v === 'muy_alta' || v === 'alta') return 'tag-warn';
  return 'tag-ok';
}

async function runOptimization(){
  const r = await fetch('/optimize_run',{method:'POST'});
  const d = await r.json();
  if(!r.ok){alert(d.detail||'Error');return;}

  clearMap();
  document.getElementById('totalVeh').textContent = d.totals?.total_vehicles_used ?? 0;

  let firstPoint = null;
  let markerCount = 1;

  for(let i=0; i<(d.routes||[]).length; i++){
    const route = d.routes[i];
    const color = rendererColor(i);
    const points = [];

    (route.stops||[]).forEach((stop)=>{
      if(!stop.lat || !stop.lng) return;
      const pos = {lat: parseFloat(stop.lat), lng: parseFloat(stop.lng)};
      if(!firstPoint) firstPoint = pos;
      points.push(pos);

     let markerFill = color;

if (stop.box_id === 'HUB') {
  markerFill = '#ffd60a';
} else if (stop.box_id === 'HUB_RETURN') {
  markerFill = '#ff6b35';
}

const mk = new google.maps.Marker({
  position: pos,
  map,
  label: {text: String(markerCount++), color: '#000', fontWeight: 'bold', fontSize: '11px'},
  icon: {
    path: google.maps.SymbolPath.CIRCLE,
    scale: 14,
    fillColor: markerFill,
    fillOpacity: 1,
    strokeColor: '#0c0e14',
    strokeWeight: 2
  }
});
      markers.push(mk);
    });

    if(points.length >= 2){
      await drawRouteChunks(points, color);
    } else if (points.length === 1) {
      map.panTo(points[0]);
    }
  }

  if(firstPoint) map.panTo(firstPoint);

  document.getElementById('summaryBox').innerHTML = `
    <div style="display:grid;gap:8px">
      <div><b>Vehículos usados:</b> ${(d.totals&&d.totals.total_vehicles_used!=null)?d.totals.total_vehicles_used:'-'}</div>
      <div><b>Entregas asignadas:</b> ${(d.totals&&d.totals.total_deliveries!=null)?d.totals.total_deliveries:'-'}</div>
      <div><b>Salida válida:</b> ${(d.metadata&&d.metadata.valid_departure)||'-'}</div>
      <div><b>Distancia total:</b> ${(d.totals&&d.totals.total_distance_km!=null)?d.totals.total_distance_km+' km':'-'}</div>
      <div><b>Tiempo total:</b> ${(d.totals&&d.totals.total_time_min!=null)?d.totals.total_time_min+' min':'-'}</div>
      <div><b>Costo estimado:</b> ${(d.totals&&d.totals.total_cost_cop!=null)?d.totals.total_cost_cop+' COP':'-'}</div>
      <div><b>Empaque 3D global:</b> ${(d.packing_3d&&d.packing_3d.overall_efficiency_pct!=null)?d.packing_3d.overall_efficiency_pct+'%':'-'}</div>
    </div>`;

  document.getElementById('routeList').innerHTML = (d.routes||[]).map((route, idx)=>`
  <div class="vehicle-block">
    <div class="vehicle-title">Vehículo ${idx+1}: ${(route.vehicle&&route.vehicle.name)||'-'} · ${(route.stats&&route.stats.total_deliveries)||0} entregas</div>
    <div class="small" style="margin-bottom:10px">
      ${(route.stats&&route.stats.total_distance_km)||0} km · ${(route.stats&&route.stats.total_time_min)||0} min · ${(route.stats&&route.stats.total_cost_cop)||0} COP · Empaque ${(route.packing_3d&&route.packing_3d.efficiency_pct)||0}%
    </div>
    ${(route.stops||[]).map((b,i)=>{
      if (b.box_id === "HUB") {
        return `
          <div class="box-row" style="border-left:3px solid #00e5a0">
            <div class="order-num">🏁</div>
            <div style="flex:1">
              <div style="font-weight:700">Salida desde HUB</div>
              <div style="color:var(--muted);font-size:12px">${b.address || '-'}</div>
              <div style="font-size:11px;color:var(--muted);margin-top:2px">
                Hora de salida: ${b.estimated_arrival || '-'}
              </div>
            </div>
            <span class="tag tag-ok">inicio</span>
          </div>`;
      }

      if (b.box_id === "HUB_RETURN") {
        return `
          <div class="box-row" style="border-left:3px solid #ff6b35">
            <div class="order-num">🏁</div>
            <div style="flex:1">
              <div style="font-weight:700">Regreso al HUB</div>
              <div style="color:var(--muted);font-size:12px">${b.address || '-'}</div>
              <div style="font-size:11px;color:var(--muted);margin-top:2px">
                Hora estimada: ${b.estimated_arrival || '-'}
              </div>
            </div>
            <span class="tag tag-warn">retorno</span>
          </div>`;
      }

      return `
        <div class="box-row">
          <div class="order-num">${i+1}</div>
          <div style="flex:1">
            <div style="font-weight:700">${b.box_id}</div>
            <div style="color:var(--muted);font-size:12px">${b.address||'-'}</div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px">
              ${b.elevation_m ? '⛰ ' + b.elevation_m + ' m' : ''} · ${b.weight_kg || 0} kg · ${b.cargo_type || '-'} · ETA ${b.estimated_arrival || '-'}
            </div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px">
              Pendiente ${b.slope_pct ?? '-'}% · Potencia req. ${(b.physics && b.physics.power_required_kw != null) ? b.physics.power_required_kw + ' kW' : '-'} · Disponible ${(b.physics && b.physics.power_available_kw != null) ? b.physics.power_available_kw + ' kW' : '-'}
            </div>
            ${b.alternate_drop ? `<div style="font-size:11px;color:#ffd60a;margin-top:4px">${b.alternate_drop}</div>` : ``}
          </div>
          <span class="tag ${slopeLabelClass(b.slope_viability)}">${b.slope_viability || 'normal'}</span>
        </div>`;
    }).join('')}
  </div>
`).join('');
}
</script>
<script async defer src="https://maps.googleapis.com/maps/api/js?key=API_KEY_HERE&callback=initMap"></script>
</body></html>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_csv()
    threading.Thread(target=process_stream, daemon=True).start()
    yield

app = FastAPI(title="BoxScan", lifespan=lifespan)

def render(page_html, active=1):
    return (
        page_html
        .replace("CSS_HERE", BASE_CSS)
        .replace("NAV_HERE", nav(active))
        .replace("API_KEY_HERE", GOOGLE_API_KEY)
    )

@app.get("/", response_class=HTMLResponse)
def portal():
    return render(PAGE_PORTAL, 1)

@app.get("/scan", response_class=HTMLResponse)
def scan_page():
    return render(PAGE_SCAN, 2)

@app.get("/optimize", response_class=HTMLResponse)
def optimize_page():
    return render(PAGE_OPTIMIZE, 3)

@app.get("/video")
def video():
    return StreamingResponse(
        mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/latest")
def get_latest():
    with lock:
        return JSONResponse(content=latest_result)

@app.get("/next_box_id")
def get_next_id():
    return {"box_id": next_box_id()}

@app.get("/recent_boxes")
def recent_boxes():
    return get_recent_boxes_json(20)

@app.get("/recent_boxes_json")
def recent_boxes_json(limit: int = 500):
    return get_recent_boxes_json(limit)

@app.post("/geocode")
def geocode(data: dict):
    addr = data.get("address", "").strip()
    if not addr:
        raise HTTPException(400, "Dirección vacía")

    result = geocode_address(addr)
    if not result.get("ok"):
        raise HTTPException(422, result.get("error", "No se pudo geocodificar"))
    return result

@app.post("/start_box")
def start_box(data: dict):
    global pending_box
    box_id = data.get("box_id") or next_box_id()

    pending_box[box_id] = {
        "box_id": box_id,
        "address": data.get("address", ""),
        "formatted_address": data.get("geo", {}).get("formatted_address", ""),
        "neighborhood": data.get("geo", {}).get("neighborhood", ""),
        "locality": data.get("geo", {}).get("locality", ""),
        "lat": data.get("geo", {}).get("lat"),
        "lng": data.get("geo", {}).get("lng"),
        "elevation_m": data.get("geo", {}).get("elevation_m"),
        "cargo_type": data.get("cargo_type", ""),
        "weight_kg": data.get("weight_kg", 0),
        "notes": data.get("notes", ""),
    }
    return {"box_id": box_id}

@app.get("/pending_box/{box_id}")
def get_pending(box_id: str):
    if box_id not in pending_box:
        raise HTTPException(404, "No hay datos pendientes")
    return pending_box[box_id]

@app.post("/confirm_scan")
def confirm_scan(data: dict):
    box_id = data.get("box_id")
    if box_id not in pending_box:
        raise HTTPException(404, "No hay datos pendientes para este Box ID")

    with lock:
        current = latest_result.copy()

    if current.get("status") != "ok":
        raise HTTPException(400, "No hay medición válida")

    p = pending_box.pop(box_id)
    l = float(current["length_cm"])
    w = float(current["width_cm"])
    h = min(l, w)

    vol_cm3 = l * w * h
    vol_m3 = vol_cm3 / 1_000_000

    record = {
        "box_id": box_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_address": p["address"],
        "formatted_address": p["formatted_address"],
        "neighborhood": p["neighborhood"],
        "locality": p["locality"],
        "lat": p["lat"],
        "lng": p["lng"],
        "elevation_m": p["elevation_m"],
        "cargo_type": p["cargo_type"],
        "length_cm": round(l, 2),
        "width_cm": round(w, 2),
        "height_cm": round(h, 2),
        "weight_kg": round(float(p["weight_kg"]), 2),
        "volume_cm3": round(vol_cm3, 2),
        "volume_m3": round(vol_m3, 6),
        "density_kg_m3": round(float(p["weight_kg"]) / vol_m3, 2) if vol_m3 > 0 else None,
        "source": "opencv_stream",
        "notes": p.get("notes", ""),
    }
    save_record(record)
    return {"message": "Guardado", "record": record}

@app.post("/optimize_run")
def optimize_run():
    boxes = get_recent_boxes_json(200)

    valid = []
    for b in boxes:
        try:
            lat = float(b.get("lat", 0) or 0)
            lng = float(b.get("lng", 0) or 0)
            elev = float(b.get("elevation_m", 0) or 0)
            weight = float(b.get("weight_kg", 0) or 0)
            volume = float(b.get("volume_m3", 0) or 0)

            if lat == 0 or lng == 0:
                continue

            item = dict(b)
            item["lat"] = lat
            item["lng"] = lng
            item["elevation_m"] = elev
            item["weight_kg"] = weight
            item["volume_m3"] = volume
            valid.append(item)

        except Exception as e:
            print("Skipping box:", b, "error:", e)

    if not valid:
        raise HTTPException(400, "No hay entregas con coordenadas para optimizar")

    try:
        result = run_full_optimization(valid, datetime.now())
        if result.get("error"):
            raise HTTPException(400, result["error"])

        return {
            "routes": result.get("routes", []),
            "totals": result.get("totals", {}),
            "metadata": result.get("metadata", {}),
            "fleet_plan": result.get("fleet_plan", []),
            "packing_3d": result.get("packing_3d", {}),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Error ejecutando optimización: {str(e)}")
