# 📦 BoxScan

AI-powered **last-mile delivery optimizer** for small courier operations in Medellín.

Combines **route optimization**, **cargo loading**, and **computer vision** into a single system.

---

## 🚀 Features

* 🚚 Vehicle Routing Optimization (OR-Tools)
* 📦 3D Bin Packing (cargo loading)
* 📷 Automatic package measurement (OpenCV + ArUco)
* ⛰️ Slope-aware routing (Medellín terrain)
* ⏰ Traffic-aware routes (Google Maps API)
* 🚫 Pico y placa constraints
* 🌐 FastAPI web app

---

## 🛠️ Tech Stack

* Python, FastAPI
* Google OR-Tools
* OpenCV
* pandas / NumPy
* Google Maps APIs

---

## ⚙️ Setup

```bash
git clone https://github.com/tiagog20/thesis.git
cd thesis
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
uvicorn code:app --reload
```

Open:

```
http://127.0.0.1:8000
```

---

## 🔑 Requirements

* Python 3.10+
* Google Maps API Key

---

## 📊 What it does

Given a set of packages:

* computes the **optimal route**
* selects the **best vehicle(s)**
* generates the **loading plan**

All before the delivery starts.

---

## ⚠️ Notes

* Uses Google Maps API (cost applies)
* Designed for small/medium delivery operations
* Tested on real Medellín-like data

---

## 📚 Context

Built as an MSc thesis project (EAFIT, 2026).
Focus: **practical optimization, not prediction**.

---

## 🤝 Contribute

PRs welcome. Fork it, improve it, adapt it.

---
