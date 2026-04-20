"""
BoxScan — optimizer.py
Versión con clustering automático para 200-500 entregas:
- clustering geográfico automático
- optimización por cluster
- múltiples vehículos
- todas las entregas obligatorias
- pendiente como evaluación física suave, no bloqueo duro
- máximo 5 vehículos por cluster
"""

import json
import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

import requests
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from box_optimizer import load_boxes_catalog_from_jsonl, optimize_vehicle_loading_from_catalog


GOOGLE_API_KEY = "XXXXXXXXXX" # Enter your Google Maps API here. Charges will be applied 

HUB = {
    "name": "Hub BoxScan",
    "address": "Carrera 50 #50-10, La Candelaria, Medellín",
    "lat": 6.2518,
    "lng": -75.5636,
    "elevation_m": 1495.0,
    "open_time": "08:00",
    "depart_time": "09:00",
}


@dataclass
class DeliveryBox:
    box_id: str
    address: str
    lat: float
    lng: float
    elevation_m: float
    cargo_type: str
    weight_kg: float
    length_cm: float
    width_cm: float
    height_cm: float
    volume_m3: float

    def volume_cm3(self) -> float:
        return self.length_cm * self.width_cm * self.height_cm


@dataclass
class VehicleTemplate:
    vehicle_id: str
    name: str
    type: str
    tare_weight_kg: float
    max_weight_kg: float
    cargo_length_cm: float
    cargo_width_cm: float
    cargo_height_cm: float
    volume_m3: float
    horsepower_hp: float
    drivetrain_efficiency: float
    cost_per_km: float
    speed_kmh: float
    assigned_plate_digit: int
    target_stop_capacity: int


@dataclass
class VehicleInstance:
    vehicle_id: str
    template_id: str
    name: str
    type: str
    tare_weight_kg: float
    max_weight_kg: float
    cargo_length_cm: float
    cargo_width_cm: float
    cargo_height_cm: float
    volume_m3: float
    horsepower_hp: float
    drivetrain_efficiency: float
    cost_per_km: float
    speed_kmh: float
    assigned_plate_digit: int
    target_stop_capacity: int


class VehicleFleet:
    def __init__(self):
        self.templates: List[VehicleTemplate] = [
            VehicleTemplate(
                "kangoo", "Renault Kangoo", "van_pequena",
                1450, 650, 170, 114, 120, 2.33, 110, 0.85, 450, 35, 0, 25
            ),
            VehicleTemplate(
                "trafic", "Renault Trafic", "van_mediana",
                1900, 1200, 267, 164, 139, 6.10, 145, 0.85, 650, 30, 5, 40
            ),
            VehicleTemplate(
                "nv350", "Nissan NV350", "van_grande",
                2150, 1500, 310, 170, 155, 8.20, 129, 0.84, 850, 28, 2, 55
            ),
            VehicleTemplate(
                "npr", "Chevrolet NPR", "camion_pequeno",
                3200, 3500, 440, 210, 210, 19.40, 150, 0.82, 1200, 25, 1, 80
            ),
        ]

    def build_vehicle_plan(
        self,
        total_weight_kg: float,
        total_volume_m3: float,
        total_deliveries: int,
        restricted_digits: List[int],
    ) -> List[VehicleInstance]:
        remaining_weight = max(total_weight_kg, 0.0)
        remaining_volume = max(total_volume_m3, 0.0)
        remaining_stops = max(total_deliveries, 0)
        vehicles: List[VehicleInstance] = []
        counter = 1

        candidates = sorted(
            self.templates,
            key=lambda t: (
                (t.cost_per_km + (100000 if t.assigned_plate_digit in restricted_digits else 0))
                / max(t.max_weight_kg + (t.volume_m3 * 250), 1)
            )
        )

        while remaining_weight > 0.01 or remaining_volume > 0.001 or remaining_stops > 0:
            best_tpl = None
            best_score = None

            for tpl in candidates:
                cover_weight = min(remaining_weight / max(tpl.max_weight_kg, 1), 1.0)
                cover_volume = min(remaining_volume / max(tpl.volume_m3, 0.001), 1.0)
                cover_stops = min(remaining_stops / max(tpl.target_stop_capacity, 1), 1.0)
                cover = cover_weight + cover_volume + cover_stops
                if cover <= 0:
                    cover = 0.01

                restriction_penalty = 500 if tpl.assigned_plate_digit in restricted_digits else 0
                score = (tpl.cost_per_km + restriction_penalty) / cover

                if best_score is None or score < best_score:
                    best_score = score
                    best_tpl = tpl

            assert best_tpl is not None

            vehicles.append(
                VehicleInstance(
                    vehicle_id=f"{best_tpl.vehicle_id}-{counter:02d}",
                    template_id=best_tpl.vehicle_id,
                    name=best_tpl.name,
                    type=best_tpl.type,
                    tare_weight_kg=best_tpl.tare_weight_kg,
                    max_weight_kg=best_tpl.max_weight_kg,
                    cargo_length_cm=best_tpl.cargo_length_cm,
                    cargo_width_cm=best_tpl.cargo_width_cm,
                    cargo_height_cm=best_tpl.cargo_height_cm,
                    volume_m3=best_tpl.volume_m3,
                    horsepower_hp=best_tpl.horsepower_hp,
                    drivetrain_efficiency=best_tpl.drivetrain_efficiency,
                    cost_per_km=best_tpl.cost_per_km,
                    speed_kmh=best_tpl.speed_kmh,
                    assigned_plate_digit=best_tpl.assigned_plate_digit,
                    target_stop_capacity=best_tpl.target_stop_capacity,
                )
            )
            counter += 1

            remaining_weight -= best_tpl.max_weight_kg
            remaining_volume -= best_tpl.volume_m3
            remaining_stops -= best_tpl.target_stop_capacity

        return vehicles


class PicoYPlaca:
    SCHEDULE = {
        0: [1, 2],
        1: [3, 4],
        2: [5, 6],
        3: [7, 8],
        4: [9, 0],
        5: [],
        6: [],
    }

    HOURS = [
        ("06:00", "08:30"),
        ("17:30", "20:00"),
    ]

    @classmethod
    def restricted_digits(cls, dt: datetime) -> List[int]:
        return cls.SCHEDULE.get(dt.weekday(), [])

    @classmethod
    def is_restricted(cls, plate_digit: int, dt: datetime) -> bool:
        if plate_digit not in cls.restricted_digits(dt):
            return False
        hhmm = dt.strftime("%H:%M")
        return any(start <= hhmm <= end for start, end in cls.HOURS)

    @classmethod
    def next_valid_departure(cls, plate_digit: int, dt: datetime) -> datetime:
        if not cls.is_restricted(plate_digit, dt):
            return dt
        for _, end in cls.HOURS:
            end_dt = dt.replace(
                hour=int(end.split(":")[0]),
                minute=int(end.split(":")[1]),
                second=0,
                microsecond=0,
            )
            if dt < end_dt:
                return end_dt
        return dt + timedelta(hours=1)


class TimeWindows:
    @classmethod
    def get_window(cls, cargo_type: str):
        return 0, 24 * 60


class GoogleMapsClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _check_key(self):
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY no configurada")

    def distance_matrix_batched(self, coords: List[str], departure_time: int, chunk_size: int = 10) -> Dict:
        self._check_key()
        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        n = len(coords)
        rows_out = [{"elements": [None] * n} for _ in range(n)]

        now_ts = int(datetime.now().timestamp())
        safe_departure = max(int(departure_time), now_ts + 60)

        for i0 in range(0, n, chunk_size):
            origins = coords[i0:i0 + chunk_size]
            for j0 in range(0, n, chunk_size):
                destinations = coords[j0:j0 + chunk_size]
                params = {
                    "origins": "|".join(origins),
                    "destinations": "|".join(destinations),
                    "key": self.api_key,
                    "mode": "driving",
                    "language": "es",
                    "units": "metric",
                    "departure_time": safe_departure,
                    "traffic_model": "best_guess",
                }
                r = requests.get(url, params=params, timeout=30)
                data = r.json()

                if data.get("status") != "OK":
                    raise RuntimeError(
                        f"Distance Matrix API falló: {data.get('status')} - "
                        f"{data.get('error_message', 'sin error_message')}"
                    )

                sub_rows = data.get("rows", [])
                for i_sub, row in enumerate(sub_rows):
                    gi = i0 + i_sub
                    elements = row.get("elements", [])
                    for j_sub, elem in enumerate(elements):
                        gj = j0 + j_sub
                        rows_out[gi]["elements"][gj] = elem

        return {"status": "OK", "rows": rows_out}


def approx_fallback_time_and_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> Tuple[int, int]:
    R = 6371000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1 - a, 1e-12)))
    dist_m = max(int(R * c), 1)
    time_s = max(int((dist_m / 1000.0) / 22.0 * 3600.0), 60)
    return dist_m, time_s


def slope_pct(elev_a: float, elev_b: float, dist_m: float) -> float:
    if dist_m <= 0:
        return 0.0
    return round(abs(elev_b - elev_a) / dist_m * 100, 2)


def estimate_climb_margin(vehicle: VehicleInstance, payload_kg: float, slope_pct_value: float, speed_kmh: float) -> Dict:
    mass = vehicle.tare_weight_kg + max(payload_kg, 0.0)
    slope_fraction = max(slope_pct_value, 0.0) / 100.0
    theta = math.atan(slope_fraction)
    speed_mps = max(speed_kmh, 1.0) / 3.6
    g = 9.81
    crr = 0.015

    force_grade = mass * g * math.sin(theta)
    force_roll = mass * g * crr * math.cos(theta)
    force_total = force_grade + force_roll

    power_required_w = force_total * speed_mps
    power_available_w = vehicle.horsepower_hp * 745.7 * vehicle.drivetrain_efficiency

    margin_w = power_available_w - power_required_w
    feasible = margin_w >= 0

    return {
        "mass_kg": round(mass, 1),
        "speed_kmh": round(speed_kmh, 1),
        "power_required_kw": round(power_required_w / 1000.0, 2),
        "power_available_kw": round(power_available_w / 1000.0, 2),
        "power_margin_kw": round(margin_w / 1000.0, 2),
        "feasible": feasible,
    }


def slope_viability_label(slope: float, climb_report: Dict) -> str:
    if not climb_report.get("feasible", True):
        return "requiere_punto_alterno"
    if slope > 18:
        return "muy_alta"
    if slope > 12:
        return "alta"
    if slope > 7:
        return "moderada"
    return "normal"


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(max(1 - a, 1e-12)))


def choose_cluster_count(deliveries: List[DeliveryBox]) -> int:
    n = len(deliveries)
    MAX_VEHICLES_PER_CLUSTER = 5
    AVG_STOPS_PER_VEHICLE = 25
    max_stops_per_cluster = MAX_VEHICLES_PER_CLUSTER * AVG_STOPS_PER_VEHICLE
    k = math.ceil(n / max_stops_per_cluster)
    return max(1, k)


def kmeans_deliveries(deliveries: List[DeliveryBox], k: int, iterations: int = 20) -> List[List[DeliveryBox]]:
    if k <= 1 or len(deliveries) <= 1:
        return [deliveries[:]]

    rng = random.Random(42)
    points = [(d.lat, d.lng) for d in deliveries]
    seed_indices = list(range(len(points)))
    rng.shuffle(seed_indices)
    centers = [points[idx] for idx in seed_indices[:k]]

    while len(centers) < k:
        centers.append(points[len(centers) % len(points)])

    assignments = [0] * len(deliveries)

    for _ in range(iterations):
        for idx, p in enumerate(points):
            best_cluster = 0
            best_dist = None
            for c_idx, c in enumerate(centers):
                dkm = haversine_km(p[0], p[1], c[0], c[1])
                if best_dist is None or dkm < best_dist:
                    best_dist = dkm
                    best_cluster = c_idx
            assignments[idx] = best_cluster

        buckets = [[] for _ in range(k)]
        for idx, a in enumerate(assignments):
            buckets[a].append(deliveries[idx])

        new_centers = []
        for b in buckets:
            if b:
                new_centers.append((
                    sum(d.lat for d in b) / len(b),
                    sum(d.lng for d in b) / len(b),
                ))
            else:
                new_centers.append(random.choice(points))

        centers = new_centers

    clusters = [[] for _ in range(k)]
    for idx, a in enumerate(assignments):
        clusters[a].append(deliveries[idx])

    clusters = [c for c in clusters if c]

     
    AVG_STOPS_PER_VEHICLE = 25
    max_cluster_size =  AVG_STOPS_PER_VEHICLE

    final_clusters: List[List[DeliveryBox]] = []

    for cluster in clusters:
        if len(cluster) <= max_cluster_size:
            final_clusters.append(cluster)
        else:
            parts = math.ceil(len(cluster) / max_cluster_size)
            cluster_sorted = sorted(cluster, key=lambda d: (d.lat, d.lng))
            for p in range(parts):
                final_clusters.append(cluster_sorted[p::parts])

    return final_clusters


class RouteOptimizer:
    def __init__(
        self,
        hub: Dict,
        deliveries: List[DeliveryBox],
        vehicles: List[VehicleInstance],
        maps_client: GoogleMapsClient,
    ):
        self.hub = hub
        self.deliveries = deliveries
        self.vehicles = vehicles
        self.maps = maps_client
        self.nodes: List[Dict] = []
        self.distance_matrix: List[List[int]] = []
        self.time_matrix: List[List[int]] = []
        self.slope_matrix: List[List[float]] = []

    def build_nodes(self):
        self.nodes = [{
            "box_id": "HUB",
            "address": self.hub["address"],
            "lat": self.hub["lat"],
            "lng": self.hub["lng"],
            "elevation_m": self.hub["elevation_m"],
            "weight_kg": 0.0,
            "volume_m3": 0.0,
            "cargo_type": "",
            "time_window": (0, 24 * 60),
            "is_hub": True,
        }]
        for d in self.deliveries:
            self.nodes.append({
                "box_id": d.box_id,
                "address": d.address,
                "lat": d.lat,
                "lng": d.lng,
                "elevation_m": d.elevation_m,
                "weight_kg": d.weight_kg,
                "volume_m3": d.volume_m3,
                "cargo_type": d.cargo_type,
                "time_window": TimeWindows.get_window(d.cargo_type),
                "is_hub": False,
            })

    def fetch_matrices(self, departure_dt: datetime):
        coords = [f"{n['lat']},{n['lng']}" for n in self.nodes]
        departure_ts = int(departure_dt.timestamp())
        data = self.maps.distance_matrix_batched(coords, departure_ts, chunk_size=10)

        if data.get("status") != "OK":
            raise RuntimeError("Distance Matrix no devolvió estado OK")

        n = len(self.nodes)
        self.distance_matrix = [[0] * n for _ in range(n)]
        self.time_matrix = [[0] * n for _ in range(n)]
        self.slope_matrix = [[0.0] * n for _ in range(n)]

        rows = data.get("rows", [])
        if len(rows) != n:
            raise RuntimeError(f"Distance Matrix devolvió {len(rows)} filas, pero se esperaban {n}")

        for i, row in enumerate(rows):
            elements = row.get("elements", [])
            if len(elements) != n:
                raise RuntimeError(
                    f"Distance Matrix devolvió {len(elements)} elementos en la fila {i}, pero se esperaban {n}"
                )
            for j, elem in enumerate(elements):
                if i == j:
                    dist_m, time_s = 0, 0
                elif elem and elem.get("status") == "OK":
                    dist_m = int(elem["distance"]["value"])
                    duration = elem.get("duration_in_traffic") or elem.get("duration") or {}
                    time_s = int(duration.get("value", elem["duration"]["value"]))
                else:
                    dist_m, time_s = approx_fallback_time_and_distance(
                        float(self.nodes[i]["lat"]),
                        float(self.nodes[i]["lng"]),
                        float(self.nodes[j]["lat"]),
                        float(self.nodes[j]["lng"]),
                    )

                self.distance_matrix[i][j] = dist_m
                self.time_matrix[i][j] = time_s
                self.slope_matrix[i][j] = slope_pct(
                    float(self.nodes[i].get("elevation_m", 0) or 0),
                    float(self.nodes[j].get("elevation_m", 0) or 0),
                    dist_m,
                )

    def _build_cost_matrix(self) -> List[List[int]]:
        n = len(self.nodes)
        cost_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                base_time = int(self.time_matrix[i][j])
                sl = float(self.slope_matrix[i][j])

                slope_penalty_s = 0
                if sl > 20:
                    slope_penalty_s = 60
                elif sl > 12:
                    slope_penalty_s = 30
                elif sl > 6:
                    slope_penalty_s = 10

                cost_matrix[i][j] = base_time + slope_penalty_s
        return cost_matrix

    def solve(self, departure_dt: datetime) -> Dict:
        n = len(self.nodes)
        num_vehicles = len(self.vehicles)
        if n <= 1:
            return {"routes": [], "totals": {}}

        cost_matrix = self._build_cost_matrix()
        starts = [0] * num_vehicles
        ends = [0] * num_vehicles

        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, starts, ends)
        routing = pywrapcp.RoutingModel(manager)

        penalty = 10_000_000
        for node in range(1, len(self.nodes)):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        def transit_cb(from_index, to_index):
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            return int(cost_matrix[i][j])

        transit_idx = routing.RegisterTransitCallback(transit_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        def time_cb(from_index, to_index):
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            service_s = 12 * 60 if j != 0 else 0
            return int(self.time_matrix[i][j] + service_s)

        time_idx = routing.RegisterTransitCallback(time_cb)
        routing.AddDimension(
            time_idx,
            2 * 60 * 60,
            48 * 60 * 60,
            False,
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        depart_s = departure_dt.hour * 3600 + departure_dt.minute * 60 + departure_dt.second
        for v in range(num_vehicles):
            start_index = routing.Start(v)
            time_dim.CumulVar(start_index).SetRange(depart_s, depart_s)

        for node_idx, node in enumerate(self.nodes):
            if node_idx == 0:
                continue
            idx = manager.NodeToIndex(node_idx)
            time_dim.CumulVar(idx).SetRange(0, 48 * 60 * 60)

        def demand_weight_cb(from_index):
            node = manager.IndexToNode(from_index)
            return int(round(float(self.nodes[node]["weight_kg"] or 0)))

        demand_weight_idx = routing.RegisterUnaryTransitCallback(demand_weight_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_weight_idx,
            0,
            [int(v.max_weight_kg) for v in self.vehicles],
            True,
            "CapacityWeight",
        )

        # La dimensión de volumen se desactiva temporalmente porque los volúmenes
        # unitarios son muy pequeños (~0.00009 m3) y el redondeo entero puede volver
        # el modelo infactible. Se conserva el control de volumen en el packing 3D.

        search = pywrapcp.DefaultRoutingSearchParameters()
        search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search.time_limit.seconds = 45
        search.log_search = False


        solution = routing.SolveWithParameters(search)
        if not solution:
            raise RuntimeError(
                "OR-Tools no encontró una solución factible en un cluster. "
                "Revisa capacidad del cluster o estados de Distance Matrix."
            )

        return self._extract_solution(manager, routing, solution, departure_dt)

    def _extract_solution(self, manager, routing, solution, departure_dt: datetime) -> Dict:
        all_routes = []
        all_assigned_ids: List[str] = []
        total_dist_m = 0
        total_time_s = 0
        total_cost = 0.0

        for vehicle_idx, vehicle in enumerate(self.vehicles):
            index = routing.Start(vehicle_idx)
            if solution.Value(routing.NextVar(index)) == routing.End(vehicle_idx):
                continue

            current_time = departure_dt
            route_stops = []
            route_total_dist_m = 0
            route_total_time_s = 0
            route_box_ids: List[str] = []

            payload_remaining = 0.0
            tmp_index = index
            while not routing.IsEnd(tmp_index):
                next_index = solution.Value(routing.NextVar(tmp_index))
                next_node = manager.IndexToNode(next_index)
                if next_node != 0:
                    payload_remaining += float(self.nodes[next_node]["weight_kg"] or 0)
                tmp_index = next_index

            while not routing.IsEnd(index):
                from_node = manager.IndexToNode(index)
                next_index = solution.Value(routing.NextVar(index))
                to_node = manager.IndexToNode(next_index)

                dist_m = int(self.distance_matrix[from_node][to_node])
                travel_s = int(self.time_matrix[from_node][to_node])

                if from_node == 0 and to_node != 0:
                    route_stops.append({
                        "stop_number": 0,
                        "box_id": "HUB",
                        "address": self.hub["address"],
                        "lat": self.hub["lat"],
                        "lng": self.hub["lng"],
                        "elevation_m": self.hub["elevation_m"],
                        "estimated_arrival": departure_dt.strftime("%H:%M"),
                        "distance_from_prev_m": 0,
                        "travel_time_min": 0,
                        "weight_kg": 0,
                        "volume_m3": 0,
                        "cargo_type": "",
                        "slope_pct": 0,
                        "slope_viability": "normal",
                        "physics": {
                            "power_required_kw": 0,
                            "power_available_kw": round(vehicle.horsepower_hp * 745.7 * vehicle.drivetrain_efficiency / 1000.0, 2),
                        },
                        "alternate_drop": None,
                        "is_hub": True,
                    })

                if to_node != 0:
                    node = self.nodes[to_node]
                    arrival = current_time + timedelta(seconds=travel_s)
                    slope_value = float(self.slope_matrix[from_node][to_node])
                    climb = estimate_climb_margin(
                        vehicle=vehicle,
                        payload_kg=payload_remaining,
                        slope_pct_value=slope_value,
                        speed_kmh=max(vehicle.speed_kmh * 0.55, 10),
                    )
                    viability = slope_viability_label(slope_value, climb)
                    alternate_drop = None
                    if not climb["feasible"]:
                        alternate_drop = "Punto alterno sugerido a 50-150 m del destino"

                    route_stops.append({
                        "stop_number": len(route_stops) + 1,
                        "box_id": node["box_id"],
                        "address": node["address"],
                        "lat": node["lat"],
                        "lng": node["lng"],
                        "elevation_m": node["elevation_m"],
                        "estimated_arrival": arrival.strftime("%H:%M"),
                        "distance_from_prev_m": dist_m,
                        "travel_time_min": round(travel_s / 60, 1),
                        "weight_kg": node["weight_kg"],
                        "volume_m3": node["volume_m3"],
                        "cargo_type": node["cargo_type"],
                        "slope_pct": round(slope_value, 2),
                        "slope_viability": viability,
                        "physics": climb,
                        "alternate_drop": alternate_drop,
                        "is_hub": False,
                    })
                    route_box_ids.append(node["box_id"])
                    all_assigned_ids.append(node["box_id"])
                    current_time = arrival + timedelta(minutes=12)
                    payload_remaining -= float(node["weight_kg"] or 0)
                else:
                    if from_node != 0:
                        current_time = current_time + timedelta(seconds=travel_s)
                        route_stops.append({
                            "stop_number": len(route_stops) + 1,
                            "box_id": "HUB_RETURN",
                            "address": self.hub["address"],
                            "lat": self.hub["lat"],
                            "lng": self.hub["lng"],
                            "elevation_m": self.hub["elevation_m"],
                            "estimated_arrival": current_time.strftime("%H:%M"),
                            "distance_from_prev_m": dist_m,
                            "travel_time_min": round(travel_s / 60, 1),
                            "weight_kg": 0,
                            "volume_m3": 0,
                            "cargo_type": "",
                            "slope_pct": 0,
                            "slope_viability": "normal",
                            "physics": {
                                "power_required_kw": 0,
                                "power_available_kw": round(vehicle.horsepower_hp * 745.7 * vehicle.drivetrain_efficiency / 1000.0, 2),
                            },
                            "alternate_drop": None,
                            "is_hub": True,
                        })

                route_total_dist_m += dist_m
                route_total_time_s += travel_s
                index = next_index

            route_cost = (route_total_dist_m / 1000.0) * vehicle.cost_per_km
            total_dist_m += route_total_dist_m
            total_time_s += route_total_time_s
            total_cost += route_cost

            all_routes.append({
                "vehicle": asdict(vehicle),
                "stops": route_stops,
                "ordered_box_ids": route_box_ids,
                "stats": {
                    "total_distance_km": round(route_total_dist_m / 1000.0, 2),
                    "total_time_min": round(route_total_time_s / 60.0, 1),
                    "total_cost_cop": round(route_cost),
                    "estimated_return": current_time.strftime("%H:%M"),
                    "total_deliveries": len([s for s in route_stops if s["box_id"] not in ("HUB", "HUB_RETURN")]),
                },
            })

        expected_ids = sorted([d.box_id for d in self.deliveries])
        assigned_set = set(all_assigned_ids)
        missing = [bid for bid in expected_ids if bid not in assigned_set]
        if missing:
            raise RuntimeError(
                "La optimización no asignó todas las entregas obligatorias en el cluster: " + ", ".join(missing)
            )

        return {
            "routes": all_routes,
            "totals": {
                "total_distance_km": round(total_dist_m / 1000.0, 2),
                "total_time_min": round(total_time_s / 60.0, 1),
                "total_cost_cop": round(total_cost),
                "total_deliveries": len(all_assigned_ids),
                "total_vehicles_used": len(all_routes),
            },
        }


class BinPacker3D:
    def __init__(self, vehicle: VehicleInstance):
        self.vehicle = vehicle

    def pack(self, boxes: List[DeliveryBox]) -> Dict:
        total_weight = sum(b.weight_kg for b in boxes)
        total_volume = sum(b.volume_m3 for b in boxes)

        fits_by_weight = total_weight <= self.vehicle.max_weight_kg
        fits_by_volume = total_volume <= self.vehicle.volume_m3

        sorted_boxes = sorted(
            boxes,
            key=lambda b: b.length_cm * b.width_cm * b.height_cm,
            reverse=True,
        )

        packed = []
        unpacked = []
        used_volume = 0.0
        used_weight = 0.0

        for b in sorted_boxes:
            if used_weight + b.weight_kg <= self.vehicle.max_weight_kg and used_volume + b.volume_m3 <= self.vehicle.volume_m3:
                packed.append({
                    "box_id": b.box_id,
                    "dims": {"l": b.length_cm, "w": b.width_cm, "h": b.height_cm},
                    "weight_kg": b.weight_kg,
                })
                used_weight += b.weight_kg
                used_volume += b.volume_m3
            else:
                unpacked.append({
                    "box_id": b.box_id,
                    "dims": {"l": b.length_cm, "w": b.width_cm, "h": b.height_cm},
                    "weight_kg": b.weight_kg,
                })

        efficiency = (used_volume / self.vehicle.volume_m3 * 100) if self.vehicle.volume_m3 else 0

        return {
            "packed": packed,
            "unpacked": unpacked,
            "total_packed": len(packed),
            "total_unpacked": len(unpacked),
            "fits": fits_by_weight and fits_by_volume and len(unpacked) == 0,
            "fits_by_weight": fits_by_weight,
            "fits_by_volume": fits_by_volume,
            "efficiency_pct": round(efficiency, 1),
            "vehicle_volume_m3": self.vehicle.volume_m3,
            "used_volume_m3": round(used_volume, 4),
            "used_weight_kg": round(used_weight, 2),
            "van_dims": {
                "l": self.vehicle.cargo_length_cm,
                "w": self.vehicle.cargo_width_cm,
                "h": self.vehicle.cargo_height_cm,
            },
        }


def parse_boxes(raw_boxes: List[Dict]) -> List[DeliveryBox]:
    parsed: List[DeliveryBox] = []
    for b in raw_boxes:
        lat = float(b.get("lat", 0) or 0)
        lng = float(b.get("lng", 0) or 0)
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            continue

        parsed.append(
            DeliveryBox(
                box_id=str(b.get("box_id", "")),
                address=b.get("formatted_address") or b.get("input_address") or "",
                lat=lat,
                lng=lng,
                elevation_m=float(b.get("elevation_m", 0) or 0),
                cargo_type=b.get("cargo_type", "Other"),
                weight_kg=float(b.get("weight_kg", 0) or 0),
                length_cm=float(b.get("length_cm", 40) or 40),
                width_cm=float(b.get("width_cm", 30) or 30),
                height_cm=float(b.get("height_cm", 25) or 25),
                volume_m3=float(b.get("volume_m3", 0) or 0),
            )
        )
    return parsed


def run_full_optimization(raw_boxes: List[Dict], date: Optional[datetime] = None) -> Dict:
    if date is None:
        date = datetime.now()

    deliveries = parse_boxes(raw_boxes)
    if not deliveries:
        return {"error": "No hay cajas válidas para optimizar"}

    restricted_digits = PicoYPlaca.restricted_digits(date)
    fleet = VehicleFleet()
    maps_client = GoogleMapsClient(GOOGLE_API_KEY)
    box_catalog = load_boxes_catalog_from_jsonl("boxes.jsonl")

    planned_departure = date.replace(
        hour=int(HUB["depart_time"].split(":")[0]),
        minute=int(HUB["depart_time"].split(":")[1]),
        second=0,
        microsecond=0,
    )
    if planned_departure <= date:
        planned_departure = date + timedelta(minutes=5)

    cluster_count = choose_cluster_count(deliveries)
    clusters = kmeans_deliveries(deliveries, cluster_count)

    all_routes: List[Dict] = []
    all_fleet_plan: List[Dict] = []
    all_packings: List[Dict] = []
    total_distance_km = 0.0
    total_time_min = 0.0
    total_cost_cop = 0
    total_vehicles_used = 0
    total_vehicle_volume = 0.0
    total_used_volume = 0.0

    delivery_by_id = {d.box_id: d for d in deliveries}
    assigned_ids = set()

    for cluster_idx, cluster_deliveries in enumerate(clusters, start=1):
        cluster_weight = sum(d.weight_kg for d in cluster_deliveries)
        cluster_volume = sum(d.volume_m3 for d in cluster_deliveries)
        cluster_count_del = len(cluster_deliveries)

        cluster_vehicles = fleet.build_vehicle_plan(
            cluster_weight,
            cluster_volume,
            cluster_count_del,
            restricted_digits,
        )
        # 🔥 clave: permitir crecimiento dinámico
        max_extra = 10  # seguridad para no explotar infinito

        min_vehicles_by_stops = max(1, math.ceil(cluster_count_del / 25))
        min_vehicles_by_stops = min(min_vehicles_by_stops, 5)

        while len(cluster_vehicles) < min_vehicles_by_stops:
            extra = fleet.templates[0]
            cluster_vehicles.append(
                VehicleInstance(
                    vehicle_id=f"{extra.vehicle_id}-extra-{len(cluster_vehicles)+1:02d}",
                    template_id=extra.vehicle_id,
                    name=extra.name,
                    type=extra.type,
                    tare_weight_kg=extra.tare_weight_kg,
                    max_weight_kg=extra.max_weight_kg,
                    cargo_length_cm=extra.cargo_length_cm,
                    cargo_width_cm=extra.cargo_width_cm,
                    cargo_height_cm=extra.cargo_height_cm,
                    volume_m3=extra.volume_m3,
                    horsepower_hp=extra.horsepower_hp,
                    drivetrain_efficiency=extra.drivetrain_efficiency,
                    cost_per_km=extra.cost_per_km,
                    speed_kmh=extra.speed_kmh,
                    assigned_plate_digit=extra.assigned_plate_digit,
                    target_stop_capacity=extra.target_stop_capacity,
                )
            )

        def compute_cluster_departure(vehicles_for_cluster: List[VehicleInstance]) -> datetime:
            cluster_departures: List[datetime] = []
            for v in vehicles_for_cluster:
                dep = PicoYPlaca.next_valid_departure(v.assigned_plate_digit, planned_departure)
                now_safe = datetime.now() + timedelta(minutes=1)
                if dep < now_safe:
                    dep = now_safe
                cluster_departures.append(dep)
            return max(cluster_departures) if cluster_departures else planned_departure

        success = False
        attempt = 0
        max_attempts = 8
        cluster_result = None

        while not success and attempt < max_attempts:
            cluster_departure = compute_cluster_departure(cluster_vehicles)
            try:
                optimizer = RouteOptimizer(
                    hub=HUB,
                    deliveries=cluster_deliveries,
                    vehicles=cluster_vehicles,
                    maps_client=maps_client,
                )
                optimizer.build_nodes()
                optimizer.fetch_matrices(cluster_departure)
                cluster_result = optimizer.solve(cluster_departure)
                success = True
            except RuntimeError as e:
                attempt += 1
                print(f"⚠️ Reintentando cluster {cluster_idx}, intento {attempt}: {e}")

                extra = max(
                    fleet.templates,
                    key=lambda v: v.max_weight_kg + (v.volume_m3 * 300)
                )

                cluster_vehicles.append(
                    VehicleInstance(
                        vehicle_id=f"{extra.vehicle_id}-retry-{attempt}",
                        template_id=extra.vehicle_id,
                        name=extra.name,
                        type=extra.type,
                        tare_weight_kg=extra.tare_weight_kg,
                        max_weight_kg=extra.max_weight_kg,
                        cargo_length_cm=extra.cargo_length_cm,
                        cargo_width_cm=extra.cargo_width_cm,
                        cargo_height_cm=extra.cargo_height_cm,
                        volume_m3=extra.volume_m3,
                        horsepower_hp=extra.horsepower_hp,
                        drivetrain_efficiency=extra.drivetrain_efficiency,
                        cost_per_km=extra.cost_per_km,
                        speed_kmh=extra.speed_kmh,
                        assigned_plate_digit=extra.assigned_plate_digit,
                        target_stop_capacity=extra.target_stop_capacity,
                    )
                )

        if not success:
            if len(cluster_deliveries) <= 3:
                raise RuntimeError(f"Cluster {cluster_idx} imposible incluso tras reintentos")

            cluster_sorted = sorted(cluster_deliveries, key=lambda d: (d.lat, d.lng))
            mid = len(cluster_sorted) // 2
            subclusters = [cluster_sorted[:mid], cluster_sorted[mid:]]

            for sub_idx, sub_cluster in enumerate(subclusters, start=1):
                if not sub_cluster:
                    continue

                sub_weight = sum(d.weight_kg for d in sub_cluster)
                sub_volume = sum(d.volume_m3 for d in sub_cluster)
                sub_vehicles = fleet.build_vehicle_plan(
                    sub_weight,
                    sub_volume,
                    len(sub_cluster),
                    restricted_digits,
                )

                sub_success = False
                sub_result = None
                sub_attempt = 0

                while not sub_success and sub_attempt < max_attempts:
                    sub_departure = compute_cluster_departure(sub_vehicles)
                    try:
                        sub_optimizer = RouteOptimizer(
                            hub=HUB,
                            deliveries=sub_cluster,
                            vehicles=sub_vehicles,
                            maps_client=maps_client,
                        )
                        sub_optimizer.build_nodes()
                        sub_optimizer.fetch_matrices(sub_departure)
                        sub_result = sub_optimizer.solve(sub_departure)
                        sub_success = True
                    except RuntimeError:
                        sub_attempt += 1
                        extra = max(
                            fleet.templates,
                            key=lambda v: v.max_weight_kg + (v.volume_m3 * 300)
                        )
                        sub_vehicles.append(
                            VehicleInstance(
                                vehicle_id=f"{extra.vehicle_id}-sub-{cluster_idx}-{sub_idx}-{sub_attempt}",
                                template_id=extra.vehicle_id,
                                name=extra.name,
                                type=extra.type,
                                tare_weight_kg=extra.tare_weight_kg,
                                max_weight_kg=extra.max_weight_kg,
                                cargo_length_cm=extra.cargo_length_cm,
                                cargo_width_cm=extra.cargo_width_cm,
                                cargo_height_cm=extra.cargo_height_cm,
                                volume_m3=extra.volume_m3,
                                horsepower_hp=extra.horsepower_hp,
                                drivetrain_efficiency=extra.drivetrain_efficiency,
                                cost_per_km=extra.cost_per_km,
                                speed_kmh=extra.speed_kmh,
                                assigned_plate_digit=extra.assigned_plate_digit,
                                target_stop_capacity=extra.target_stop_capacity,
                            )
                        )

                if not sub_success or sub_result is None:
                    raise RuntimeError(f"Cluster {cluster_idx}.{sub_idx} imposible incluso tras split")

                for route in sub_result["routes"]:
                    route["cluster_id"] = cluster_idx
                    route["vehicle"]["cluster_id"] = cluster_idx

                    vehicle = VehicleInstance(**{k: v for k, v in route["vehicle"].items() if k != "cluster_id"})
                    ordered_boxes = [delivery_by_id[bid] for bid in route["ordered_box_ids"] if bid in delivery_by_id]
                    packer = BinPacker3D(vehicle)
                    pack = packer.pack(ordered_boxes)
                    route["packing_3d"] = pack
                    all_packings.append(pack)
                    total_used_volume += pack["used_volume_m3"]
                    total_vehicle_volume += vehicle.volume_m3

                    for bid in route["ordered_box_ids"]:
                        assigned_ids.add(bid)

                    all_routes.append(route)

                all_fleet_plan.extend([{**asdict(v), "cluster_id": cluster_idx} for v in sub_vehicles])

                totals = sub_result["totals"]
                total_distance_km += float(totals.get("total_distance_km", 0) or 0)
                total_time_min += float(totals.get("total_time_min", 0) or 0)
                total_cost_cop += int(totals.get("total_cost_cop", 0) or 0)
                total_vehicles_used += int(totals.get("total_vehicles_used", 0) or 0)

            continue

        assert cluster_result is not None



        for route in cluster_result["routes"]:
            route["cluster_id"] = cluster_idx
            route["vehicle"]["cluster_id"] = cluster_idx

            vehicle = VehicleInstance(**{k: v for k, v in route["vehicle"].items() if k != "cluster_id"})

            pack = optimize_vehicle_loading_from_catalog(
                vehicle_dict=route["vehicle"],
                route_stops=route["stops"],
                box_catalog=box_catalog,
            )

            route["packing_3d"] = pack
            all_packings.append(pack)
            total_used_volume += pack["stats"]["used_volume_cm3"] / 1_000_000.0
            total_vehicle_volume += vehicle.volume_m3

            for bid in route["ordered_box_ids"]:
                assigned_ids.add(bid)

            all_routes.append(route)

        all_fleet_plan.extend([{**asdict(v), "cluster_id": cluster_idx} for v in cluster_vehicles])

        totals = cluster_result["totals"]
        total_distance_km += float(totals.get("total_distance_km", 0) or 0)
        total_time_min += float(totals.get("total_time_min", 0) or 0)
        total_cost_cop += int(totals.get("total_cost_cop", 0) or 0)
        total_vehicles_used += int(totals.get("total_vehicles_used", 0) or 0)

    missing_ids = sorted([d.box_id for d in deliveries if d.box_id not in assigned_ids])
    if missing_ids:
        raise RuntimeError(
            "La optimización por clusters dejó entregas sin asignar: " + ", ".join(missing_ids)
        )

    total_weight = sum(d.weight_kg for d in deliveries)
    total_volume = sum(d.volume_m3 for d in deliveries)

    all_routes = sorted(
        all_routes,
        key=lambda r: (
            r.get("cluster_id", 0),
            r.get("vehicle", {}).get("vehicle_id", "")
        )
    )

    result = {
        "metadata": {
            "date": date.strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
            "total_boxes": len(deliveries),
            "total_weight_kg": round(total_weight, 2),
            "total_volume_m3": round(total_volume, 4),
            "hub": HUB["name"],
            "hub_address": HUB["address"],
            "planned_depart": planned_departure.strftime("%H:%M"),
            "valid_departure": planned_departure.strftime("%H:%M"),
            "vehicles_planned": len(all_fleet_plan),
            "clusters_used": len(clusters),
        },
        "fleet_plan": all_fleet_plan,
        "routes": all_routes,
        "totals": {
            "total_distance_km": round(total_distance_km, 2),
            "total_time_min": round(total_time_min, 1),
            "total_cost_cop": round(total_cost_cop),
            "total_deliveries": len(assigned_ids),
            "total_vehicles_used": total_vehicles_used,
            "total_weight_kg": round(total_weight, 2),
            "total_volume_m3": round(total_volume, 4),
        },
        "packing_3d": {
            "overall_efficiency_pct": round((total_used_volume / total_vehicle_volume * 100), 1) if total_vehicle_volume else 0.0,
            "vehicles": all_packings,
            "fits_all": all(p["stats"]["boxes_unplaced"] == 0 for p in all_packings) if all_packings else True,
        },
    }

    output_file = f"optimization_{date.strftime('%Y%m%d')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
