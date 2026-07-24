# 📦 BoxScam

AI-powered **last-mile delivery optimization system** for small courier operations in Medellín.
Combines **route optimization**, **3D cargo loading**, and **computer vision-based package measurement** into a single deployable web application.

---

## 🎯 The Problem

Small last-mile couriers in Medellín face a unique combination of operational constraints:

- Steep topographic gradients that affect vehicle climbing capacity
- Pico y placa traffic restrictions that limit daily circulation windows
- Narrow margins that preclude adoption of enterprise logistics software
- Manual package measurement that propagates errors downstream into routing and loading

BoxScan was built specifically for this reality — running on a laptop, requiring no cloud infrastructure, and accessible to operators without a technical background.

---

## 🚀 Features

- 🚚 **Multi-vehicle routing** with Google OR-Tools (VRP solver)
- 📦 **3D bin packing** with reverse delivery-order sequencing
- 📷 **Automated package measurement** using OpenCV and ArUco markers
- ⛰️ **Slope-aware routing** with physics-based vehicle climb evaluation
- ⏰ **Traffic-aware routes** via Google Maps Distance Matrix API
- 🚫 **Pico y placa compliance** with automatic departure-time adjustment
- 🧮 **Heterogeneous fleet selection** across four vehicle types
- 🌐 **FastAPI web interface** with three-page operational workflow

---

## 🏗️ System Architecture

BoxScan integrates three modules orchestrated by a FastAPI backend:

| Module | File | Responsibility |
|---|---|---|
| Application layer | `code.py` | Web interface, camera stream, geocoding, persistence |
| Route optimizer | `optimizer.py` | VRP, clustering, slope penalties, fleet selection |
| Cargo packer | `box_optimizer.py` | 3D bin packing with delivery-order sequencing |

The complete workflow takes a courier from package intake to optimized route in three screens: registration, measurement, and optimization.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn
- **Optimization:** Google OR-Tools
- **Computer vision:** OpenCV (with ArUco contrib modules)
- **Data:** pandas, NumPy
- **External APIs:** Google Maps Geocoding, Elevation, and Distance Matrix
- **Frontend:** HTML, CSS, JavaScript (vanilla)

---

## ⚙️ Setup

```bash
git clone https://github.com/tiagog20/thesis.git
cd thesis
pip install -r requirements.txt
```

Create a `.env` file with your Google Maps API key:

```
GOOGLE_MAPS_API_KEY=your_key_here
```

---

## ▶️ Run

```bash
uvicorn code:app --reload
```

Then open:

```
http://127.0.0.1:8000
```

The interface guides you through three steps: **Portal** (register packages), **Measurement** (capture dimensions with the camera), and **Optimization** (generate routes and loading plans).

---

## 🔑 Requirements

- Python 3.10 or higher
- A Google Maps API key with Geocoding, Elevation, and Distance Matrix enabled
- A standard webcam (for the measurement module)
- A printed 5 × 5 cm ArUco marker (DICT_4X4_50)

---

## 📊 What It Does

Given a set of registered packages, BoxScan:

1. **Validates** every delivery address via the Google Maps Geocoding API
2. **Measures** each package automatically using ArUco-based computer vision
3. **Clusters** deliveries geographically with a custom K-means algorithm
4. **Selects** the most cost-efficient vehicle from a fleet catalog
5. **Optimizes** the route with real traffic, slope penalties, and time windows
6. **Packs** the cargo in reverse delivery order for accessible unloading
7. **Exports** the complete plan as a dated JSON file

All before the delivery vehicle leaves the warehouse.

---

## 📁 Repository Structure

```
thesis/
├── code.py                      # FastAPI app and computer vision
├── optimizer.py                 # VRP optimizer with OR-Tools
├── box_optimizer.py             # 3D bin packing algorithm
├── boxes.csv / boxes.jsonl      # Registered package records
├── dataset.csv                  # Historical delivery dataset
├── optimization_YYYYMMDD.json   # Sample optimization output
├── EDA.ipynb                    # Exploratory data analysis
├── requirements.txt             # Python dependencies
└── Test/                        # Camera calibration scripts
```

---

## ⚠️ Notes and Limitations

- The Google Maps Distance Matrix API consumes credits — large daily volumes can incur high costs
- Designed for small-to-medium courier operations (up to ~500 simultaneous deliveries per run)
- The pico y placa schedule is hardcoded for Medellín's 2024–2026 rules
- Tested on real Medellín delivery data; performance in other cities will depend on local geographic and regulatory conditions

---

## 📚 Academic Context

This system was developed as part of an MSc thesis at EAFIT University (2026), in the Analytics and Data Science program. The research focuses on **practical optimization for resource-constrained operators**, not predictive modeling.

The methodological foundation builds on the seminal work of Bruni et al. (2023) on machine learning heuristics for last-mile delivery and third-party logistics, adapted to the operational reality of small courier companies in Medellín.

**Author:** Santiago González Granada
**Advisor:** Juan Carlos Monroy Osorio

---

## 🤝 Contributing

Pull requests are welcome. If you adapt this system to another city, another fleet, or another set of constraints, please open an issue first to discuss the changes. The system is designed to be forked and modified — that's the point.

---

## 📄 License

Released under the MIT License. Use it, modify it, deploy it — just keep the attribution.
