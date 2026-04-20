"""
box_optimizer.py

Módulo independiente para optimizar la disposición de cajas dentro del vehículo
sin tocar optimizer.py.

Diseñado para evitar discrepancias:
- La asignación y el orden de entrega salen de optimizer.py
- Las dimensiones y peso salen del dataset original (boxes.jsonl o deliveries parseadas)
- La unión se hace por box_id

Incluye:
- packing 3D heurístico
- soporte para orden de descarga
- helpers para construir la carga desde rutas del optimizer
- loader opcional desde boxes.jsonl
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import json


# ============================================================
# MODELOS
# ============================================================

@dataclass
class Box3D:
    box_id: str
    length_cm: float
    width_cm: float
    height_cm: float
    weight_kg: float
    delivery_order: int
    fragile: bool = False
    can_rotate: bool = True
    cargo_type: str = "Other"

    @property
    def volume_cm3(self) -> float:
        return self.length_cm * self.width_cm * self.height_cm


@dataclass
class VehicleBin:
    vehicle_id: str
    name: str
    cargo_length_cm: float
    cargo_width_cm: float
    cargo_height_cm: float
    max_weight_kg: float

    @property
    def volume_cm3(self) -> float:
        return self.cargo_length_cm * self.cargo_width_cm * self.cargo_height_cm


@dataclass
class Placement:
    box_id: str
    x_cm: float
    y_cm: float
    z_cm: float
    length_cm: float
    width_cm: float
    height_cm: float
    weight_kg: float
    delivery_order: int
    fragile: bool
    cargo_type: str

    @property
    def x2(self) -> float:
        return self.x_cm + self.length_cm

    @property
    def y2(self) -> float:
        return self.y_cm + self.width_cm

    @property
    def z2(self) -> float:
        return self.z_cm + self.height_cm


@dataclass
class FreeSpace:
    x_cm: float
    y_cm: float
    z_cm: float
    length_cm: float
    width_cm: float
    height_cm: float

    @property
    def volume_cm3(self) -> float:
        return self.length_cm * self.width_cm * self.height_cm


# ============================================================
# UTILIDADES
# ============================================================

def overlap_1d(a1: float, a2: float, b1: float, b2: float) -> bool:
    return max(a1, b1) < min(a2, b2)


def placements_intersect(a: Placement, b: Placement) -> bool:
    return (
        overlap_1d(a.x_cm, a.x2, b.x_cm, b.x2)
        and overlap_1d(a.y_cm, a.y2, b.y_cm, b.y2)
        and overlap_1d(a.z_cm, a.z2, b.z_cm, b.z2)
    )


def box_variants(box: Box3D) -> List[Tuple[float, float, float]]:
    dims = [box.length_cm, box.width_cm, box.height_cm]
    if not box.can_rotate:
        return [(box.length_cm, box.width_cm, box.height_cm)]

    variants = set()
    for a in dims:
        for b in dims:
            for c in dims:
                if sorted([a, b, c]) == sorted(dims):
                    variants.add((a, b, c))
    return sorted(list(variants))


def fits_in_space(space: FreeSpace, dims: Tuple[float, float, float]) -> bool:
    l, w, h = dims
    return (
        l <= space.length_cm + 1e-9
        and w <= space.width_cm + 1e-9
        and h <= space.height_cm + 1e-9
    )


def support_ratio(candidate: Placement, placed: List[Placement]) -> float:
    if candidate.z_cm <= 1e-9:
        return 1.0

    base_area = candidate.length_cm * candidate.width_cm
    if base_area <= 0:
        return 0.0

    supported_area = 0.0
    for p in placed:
        if abs(p.z2 - candidate.z_cm) > 1e-6:
            continue

        x_overlap = max(0.0, min(candidate.x2, p.x2) - max(candidate.x_cm, p.x_cm))
        y_overlap = max(0.0, min(candidate.y2, p.y2) - max(candidate.y_cm, p.y_cm))
        supported_area += x_overlap * y_overlap

    return min(1.0, supported_area / base_area)


# ============================================================
# HEURÍSTICA DE PACKING 3D
# ============================================================

class PackingOptimizer3D:
    """
    Heurística estable para MVP:
    - usa espacios libres
    - prioriza estabilidad
    - respeta parcialmente el orden de descarga:
      * entregas tardías más al fondo
      * entregas tempranas más cerca de la puerta
    Convención:
    - x = profundidad del vehículo (0 cerca de la puerta, mayor = más al fondo)
    - y = ancho
    - z = altura
    """

    def __init__(self, vehicle: VehicleBin):
        self.vehicle = vehicle

    def optimize(self, boxes: List[Box3D]) -> Dict:
        # Cargar primero las entregas tardías para dejarlas al fondo.
        ordered = sorted(
            boxes,
            key=lambda b: (-b.delivery_order, -b.weight_kg, -b.volume_cm3)
        )

        placed: List[Placement] = []
        unplaced: List[Box3D] = []
        free_spaces: List[FreeSpace] = [
            FreeSpace(
                x_cm=0.0,
                y_cm=0.0,
                z_cm=0.0,
                length_cm=self.vehicle.cargo_length_cm,
                width_cm=self.vehicle.cargo_width_cm,
                height_cm=self.vehicle.cargo_height_cm,
            )
        ]

        current_weight = 0.0

        for box in ordered:
            if current_weight + box.weight_kg > self.vehicle.max_weight_kg + 1e-9:
                unplaced.append(box)
                continue

            placement = self._place_box(box, free_spaces, placed)
            if placement is None:
                unplaced.append(box)
                continue

            placed.append(placement)
            current_weight += box.weight_kg
            free_spaces = self._split_spaces(free_spaces, placement)
            free_spaces = self._prune_spaces(free_spaces)

        used_volume = sum(p.length_cm * p.width_cm * p.height_cm for p in placed)
        efficiency = (used_volume / self.vehicle.volume_cm3 * 100.0) if self.vehicle.volume_cm3 else 0.0

        unload_score = 0.0
        for p in placed:
            # Penaliza primeras entregas ubicadas demasiado al fondo.
            unload_score += (100 - p.delivery_order) * p.x_cm

        center_of_mass = self._center_of_mass(placed)

        return {
            "vehicle": asdict(self.vehicle),
            "placed": [asdict(p) for p in placed],
            "unplaced": [asdict(b) for b in unplaced],
            "stats": {
                "boxes_total": len(boxes),
                "boxes_placed": len(placed),
                "boxes_unplaced": len(unplaced),
                "weight_used_kg": round(current_weight, 2),
                "weight_capacity_kg": round(self.vehicle.max_weight_kg, 2),
                "used_volume_cm3": round(used_volume, 2),
                "vehicle_volume_cm3": round(self.vehicle.volume_cm3, 2),
                "efficiency_pct": round(efficiency, 2),
                "unload_score": round(unload_score, 2),
                "center_of_mass_cm": center_of_mass,
            },
        }

    def _place_box(
        self,
        box: Box3D,
        free_spaces: List[FreeSpace],
        placed: List[Placement],
    ) -> Optional[Placement]:
        best_candidate: Optional[Placement] = None
        best_score: Optional[Tuple[float, float, float, float, float]] = None

        for space in sorted(free_spaces, key=lambda s: (s.z_cm, -s.x_cm, s.y_cm, s.volume_cm3)):
            for dims in box_variants(box):
                if not fits_in_space(space, dims):
                    continue

                l, w, h = dims
                candidate = Placement(
                    box_id=box.box_id,
                    x_cm=space.x_cm,
                    y_cm=space.y_cm,
                    z_cm=space.z_cm,
                    length_cm=l,
                    width_cm=w,
                    height_cm=h,
                    weight_kg=box.weight_kg,
                    delivery_order=box.delivery_order,
                    fragile=box.fragile,
                    cargo_type=box.cargo_type,
                )

                if any(placements_intersect(candidate, p) for p in placed):
                    continue

                support = support_ratio(candidate, placed)
                if candidate.z_cm > 0 and support < 0.7:
                    continue

                # Fragile: preferir arriba y no debajo de cajas pesadas
                fragile_bonus = -candidate.z_cm if box.fragile else candidate.z_cm

                leftover = (
                    (space.length_cm - l)
                    + (space.width_cm - w)
                    + (space.height_cm - h)
                )

                # delivery_order alto = entregar tarde = más al fondo (x alto)
                order_penalty = abs(candidate.x_cm - self._target_x_for_order(box.delivery_order))

                score = (
                    0.0 if candidate.z_cm == 0 else 1.0 - support,
                    order_penalty,
                    fragile_bonus,
                    candidate.z_cm,
                    leftover,
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best_candidate = candidate

        return best_candidate

    def _target_x_for_order(self, delivery_order: int) -> float:
        # Aproximación: primeras entregas cerca de la puerta, últimas al fondo.
        if delivery_order <= 1:
            return 0.0
        return min(self.vehicle.cargo_length_cm * 0.75, delivery_order * 5.0)

    def _split_spaces(self, spaces: List[FreeSpace], placement: Placement) -> List[FreeSpace]:
        new_spaces: List[FreeSpace] = []

        for s in spaces:
            dummy = Placement(
                box_id="_space_test",
                x_cm=s.x_cm,
                y_cm=s.y_cm,
                z_cm=s.z_cm,
                length_cm=s.length_cm,
                width_cm=s.width_cm,
                height_cm=s.height_cm,
                weight_kg=0.0,
                delivery_order=0,
                fragile=False,
                cargo_type="",
            )

            if not placements_intersect(dummy, placement):
                new_spaces.append(s)
                continue

            # fondo restante en x
            back_len = s.x_cm + s.length_cm - placement.x2
            if back_len > 1e-6:
                new_spaces.append(
                    FreeSpace(
                        x_cm=placement.x2,
                        y_cm=s.y_cm,
                        z_cm=s.z_cm,
                        length_cm=back_len,
                        width_cm=s.width_cm,
                        height_cm=s.height_cm,
                    )
                )

            # lateral restante en y
            side_w = s.y_cm + s.width_cm - placement.y2
            if side_w > 1e-6:
                new_spaces.append(
                    FreeSpace(
                        x_cm=s.x_cm,
                        y_cm=placement.y2,
                        z_cm=s.z_cm,
                        length_cm=s.length_cm,
                        width_cm=side_w,
                        height_cm=s.height_cm,
                    )
                )

            # arriba en z
            top_h = s.z_cm + s.height_cm - placement.z2
            if top_h > 1e-6:
                new_spaces.append(
                    FreeSpace(
                        x_cm=s.x_cm,
                        y_cm=s.y_cm,
                        z_cm=placement.z2,
                        length_cm=s.length_cm,
                        width_cm=s.width_cm,
                        height_cm=top_h,
                    )
                )

        return [s for s in new_spaces if s.length_cm > 1e-6 and s.width_cm > 1e-6 and s.height_cm > 1e-6]

    def _prune_spaces(self, spaces: List[FreeSpace]) -> List[FreeSpace]:
        pruned: List[FreeSpace] = []
        for i, a in enumerate(spaces):
            contained = False
            for j, b in enumerate(spaces):
                if i == j:
                    continue
                if (
                    a.x_cm >= b.x_cm - 1e-9
                    and a.y_cm >= b.y_cm - 1e-9
                    and a.z_cm >= b.z_cm - 1e-9
                    and a.x_cm + a.length_cm <= b.x_cm + b.length_cm + 1e-9
                    and a.y_cm + a.width_cm <= b.y_cm + b.width_cm + 1e-9
                    and a.z_cm + a.height_cm <= b.z_cm + b.height_cm + 1e-9
                ):
                    contained = True
                    break
            if not contained:
                pruned.append(a)
        return pruned

    def _center_of_mass(self, placed: List[Placement]) -> Dict[str, float]:
        if not placed:
            return {"x_cm": 0.0, "y_cm": 0.0, "z_cm": 0.0}

        total_weight = sum(p.weight_kg for p in placed)
        if total_weight <= 0:
            return {"x_cm": 0.0, "y_cm": 0.0, "z_cm": 0.0}

        x = sum((p.x_cm + p.length_cm / 2.0) * p.weight_kg for p in placed) / total_weight
        y = sum((p.y_cm + p.width_cm / 2.0) * p.weight_kg for p in placed) / total_weight
        z = sum((p.z_cm + p.height_cm / 2.0) * p.weight_kg for p in placed) / total_weight
        return {"x_cm": round(x, 2), "y_cm": round(y, 2), "z_cm": round(z, 2)}


# ============================================================
# ADAPTADORES
# ============================================================

def build_vehicle_bin_from_optimizer_vehicle(vehicle_dict: Dict) -> VehicleBin:
    return VehicleBin(
        vehicle_id=str(vehicle_dict.get("vehicle_id", "")),
        name=str(vehicle_dict.get("name", "Vehículo")),
        cargo_length_cm=float(vehicle_dict.get("cargo_length_cm", 0) or 0),
        cargo_width_cm=float(vehicle_dict.get("cargo_width_cm", 0) or 0),
        cargo_height_cm=float(vehicle_dict.get("cargo_height_cm", 0) or 0),
        max_weight_kg=float(vehicle_dict.get("max_weight_kg", 0) or 0),
    )


def load_boxes_catalog_from_jsonl(jsonl_path: str) -> Dict[str, Dict]:
    catalog: Dict[str, Dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            box_id = str(item.get("box_id", ""))
            if box_id:
                catalog[box_id] = item
    return catalog


def build_boxes_from_route_and_catalog(route_stops: List[Dict], box_catalog: Dict[str, Dict]) -> List[Box3D]:
    out: List[Box3D] = []

    for i, stop in enumerate(route_stops, start=1):
        box_id = str(stop.get("box_id", ""))
        if box_id in ("", "HUB", "HUB_RETURN"):
            continue

        raw = box_catalog.get(box_id)
        if not raw:
            continue

        cargo_type = str(raw.get("cargo_type", "Other") or "Other")
        fragile = cargo_type.lower() in {
            "fragile items",
            "electronics",
            "cosmetics and personal care",
            "pharmaceuticals",
        }

        out.append(
            Box3D(
                box_id=box_id,
                length_cm=float(raw.get("length_cm", 0) or 0),
                width_cm=float(raw.get("width_cm", 0) or 0),
                height_cm=float(raw.get("height_cm", 0) or 0),
                weight_kg=float(raw.get("weight_kg", 0) or 0),
                delivery_order=int(stop.get("stop_number", i)),
                fragile=fragile,
                can_rotate=not fragile,
                cargo_type=cargo_type,
            )
        )

    return out


def optimize_vehicle_loading_from_catalog(
    vehicle_dict: Dict,
    route_stops: List[Dict],
    box_catalog: Dict[str, Dict],
) -> Dict:
    vehicle = build_vehicle_bin_from_optimizer_vehicle(vehicle_dict)
    boxes = build_boxes_from_route_and_catalog(route_stops, box_catalog)
    optimizer = PackingOptimizer3D(vehicle)
    return optimizer.optimize(boxes)


def optimize_vehicle_loading_from_jsonl(
    vehicle_dict: Dict,
    route_stops: List[Dict],
    jsonl_path: str,
) -> Dict:
    catalog = load_boxes_catalog_from_jsonl(jsonl_path)
    return optimize_vehicle_loading_from_catalog(vehicle_dict, route_stops, catalog)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    vehicle = {
        "vehicle_id": "kangoo-01",
        "name": "Renault Kangoo",
        "cargo_length_cm": 170,
        "cargo_width_cm": 114,
        "cargo_height_cm": 120,
        "max_weight_kg": 650,
    }

    box_catalog = {
        "BX-001": {
            "box_id": "BX-001",
            "length_cm": 50,
            "width_cm": 35,
            "height_cm": 25,
            "weight_kg": 8,
            "cargo_type": "Industrial supplies",
        },
        "BX-002": {
            "box_id": "BX-002",
            "length_cm": 40,
            "width_cm": 30,
            "height_cm": 20,
            "weight_kg": 5,
            "cargo_type": "Pharmaceuticals",
        },
        "BX-003": {
            "box_id": "BX-003",
            "length_cm": 60,
            "width_cm": 40,
            "height_cm": 30,
            "weight_kg": 10,
            "cargo_type": "Hardware and tools",
        },
    }

    route_stops = [
        {"box_id": "HUB", "stop_number": 0},
        {"box_id": "BX-002", "stop_number": 1},
        {"box_id": "BX-003", "stop_number": 2},
        {"box_id": "BX-001", "stop_number": 3},
        {"box_id": "HUB_RETURN", "stop_number": 4},
    ]

    result = optimize_vehicle_loading_from_catalog(vehicle, route_stops, box_catalog)
    print(json.dumps(result, indent=2, ensure_ascii=False))
