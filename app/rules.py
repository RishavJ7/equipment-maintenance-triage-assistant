"""Deterministic threshold checks. No AI involved: same input -> same output."""
import re

LEVELS = ["Low", "Medium", "High", "Critical"]


def S(label, unit, warn, alarm, direction="high", rng=(0, 1000)):
    return {"label": label, "unit": unit, "warn": warn, "alarm": alarm,
            "direction": direction, "range": rng}


# direction "high": warn/alarm when value >= limit. "low": when value <= limit.
THRESHOLDS = {
    "compressor": {
        "discharge_temp": S("Discharge temperature", "°C", 110, 130, "high", (-20, 300)),
        "vibration": S("Vibration velocity", "mm/s", 4.5, 7.1, "high", (0, 50)),
        "discharge_pressure": S("Discharge pressure", "bar", 9.5, 11.0, "high", (0, 40)),
        "oil_pressure": S("Oil pressure", "bar", 2.0, 1.5, "low", (0, 15)),
        "motor_current": S("Motor current", "A", 95, 110, "high", (0, 500)),
    },
    "pump": {
        "bearing_temp": S("Bearing temperature", "°C", 75, 90, "high", (-20, 250)),
        "vibration": S("Vibration velocity", "mm/s", 4.5, 7.1, "high", (0, 50)),
        "discharge_pressure": S("Discharge pressure", "bar", 3.0, 2.0, "low", (0, 40)),
        "flow_rate": S("Flow rate", "% of rated", 80, 60, "low", (0, 150)),
        "motor_current": S("Motor current", "A", 40, 48, "high", (0, 500)),
    },
    "motor": {
        "winding_temp": S("Winding temperature", "°C", 120, 140, "high", (-20, 300)),
        "bearing_temp": S("Bearing temperature", "°C", 80, 95, "high", (-20, 250)),
        "vibration": S("Vibration velocity", "mm/s", 4.5, 7.1, "high", (0, 50)),
        "current_imbalance": S("Phase current imbalance", "%", 5, 10, "high", (0, 100)),
    },
    "conveyor": {
        "gearbox_oil_temp": S("Gearbox oil temperature", "°C", 80, 95, "high", (-20, 250)),
        "vibration": S("Vibration velocity", "mm/s", 4.5, 7.1, "high", (0, 50)),
        "motor_current": S("Drive motor current", "A", 30, 38, "high", (0, 500)),
        "belt_speed": S("Belt speed", "% of nominal", 90, 80, "low", (0, 150)),
    },
}

SAFETY_KEYWORDS = ["smoke", "fire", "flames", "sparks", "sparking", "burning smell",
                   "burnt smell", "exposed wire", "electric shock", "gas leak"]

FAMILIES = {
    "temperature": (["temp", "temperature", "heat"], ["temp"]),
    "vibration": (["vibration", "vibrating", "shaking"], ["vib"]),
    "pressure": (["pressure"], ["pressure"]),
    "current": (["current", "amps"], ["current"]),
}


def _status(spec, v):
    if spec["direction"] == "high":
        return "alarm" if v >= spec["alarm"] else "warning" if v >= spec["warn"] else "normal"
    return "alarm" if v <= spec["alarm"] else "warning" if v <= spec["warn"] else "normal"


def _fmt(x):
    return f"{x:g}"


def evaluate(equipment_type, readings):
    """readings: list of {"sensor": str, "value": float|None}. Duplicates allowed (conflict detection)."""
    specs = THRESHOLDS[equipment_type]
    grouped = {}
    for r in readings or []:
        name = (r.get("sensor") or "").strip()
        if name:
            grouped.setdefault(name, []).append(r.get("value"))

    results, notes = [], []
    for name, spec in specs.items():
        vals = grouped.pop(name, None)
        item = {"sensor": name, "label": spec["label"], "unit": spec["unit"], "values": [],
                "used_value": None, "status": "not_provided", "conflict": False,
                "limits": {"warn": spec["warn"], "alarm": spec["alarm"], "direction": spec["direction"]},
                "note": "No reading provided. This sensor was not checked."}
        if vals is not None:
            valid, invalid = [], []
            for v in vals:
                if v is None:
                    continue
                (valid if spec["range"][0] <= v <= spec["range"][1] else invalid).append(v)
            item["values"] = valid + invalid
            if invalid:
                notes.append(f"{spec['label']}: reading(s) {', '.join(map(_fmt, invalid))} are outside the plausible "
                             f"range {spec['range'][0]}-{spec['range'][1]} {spec['unit']} and were ignored (possible sensor fault or typo).")
            if not valid and invalid:
                item["status"], item["note"] = "invalid", "All readings implausible; not evaluated."
            elif not valid:
                item["note"] = "Sensor listed without a value. Not checked."
            else:
                hi, lo = max(valid), min(valid)
                worst = hi if spec["direction"] == "high" else lo
                item["used_value"] = worst
                item["status"] = _status(spec, worst)
                op = ">=" if spec["direction"] == "high" else "<="
                if len(valid) > 1 and (hi - lo) > max(0.10 * abs(hi), 1e-9):
                    item["conflict"] = True
                    notes.append(f"{spec['label']}: conflicting readings ({', '.join(map(_fmt, valid))} {spec['unit']}). "
                                 f"Evaluated conservatively using {_fmt(worst)}; verify with a calibrated instrument.")
                    item["note"] = f"Conflicting readings; worst case {_fmt(worst)} used."
                else:
                    item["note"] = (f"{_fmt(worst)} {spec['unit']} (warn {op}{_fmt(spec['warn'])}, "
                                    f"alarm {op}{_fmt(spec['alarm'])})")
        results.append(item)

    for name, vs in grouped.items():  # sensors with no rule defined
        results.append({"sensor": name, "label": name, "unit": "", "values": [v for v in vs if v is not None],
                        "used_value": None, "status": "unchecked", "conflict": False, "limits": None,
                        "note": "No threshold rule defined for this sensor; shown as an observation only."})
    for i, r in enumerate(results, 1):
        r["id"] = f"R{i}"

    checked = [r for r in results if r["status"] in ("normal", "warning", "alarm")]
    alarms = [r for r in checked if r["status"] == "alarm"]
    warns = [r for r in checked if r["status"] == "warning"]
    if not checked:
        overall = "insufficient_data"
        notes.append("No usable sensor readings. Threshold checks could not be performed.")
    else:
        overall = "alarm" if alarms else "warning" if warns else "normal"
    missing = [r["label"] for r in results if r["status"] == "not_provided"]
    if missing and checked:
        notes.append("Not provided (not checked): " + ", ".join(missing) + ".")
    return {"items": results, "overall": overall, "alarm_count": len(alarms),
            "warning_count": len(warns), "data_quality_notes": notes}


def text_conflicts(text, items):
    """Flag when free text says a quantity is normal but a sensor of that family is in warning/alarm."""
    out, low = [], text.lower()
    for fam, (words, keys) in FAMILIES.items():
        w = "|".join(words)
        pat = (rf"\b(?:normal|fine|stable|ok|okay|no unusual|no abnormal)\W+(?:\w+\W+){{0,2}}?(?:{w})\b"
               rf"|\b(?:{w})\w*\W+(?:\w+\W+){{0,2}}?(?:normal|fine|stable|ok|okay)\b")
        if not re.search(pat, low):
            continue
        for it in items:
            if any(k in it["sensor"].lower() for k in keys) and it["status"] in ("warning", "alarm"):
                out.append(f"The report text says {fam} is normal, but {it['label']} is in {it['status'].upper()} "
                           f"({_fmt(it['used_value'])} {it['unit']}). Verify which is correct.")
    return out


def safety_flags(*texts):
    low = " ".join(texts).lower()
    return [k for k in SAFETY_KEYWORDS if k in low]


def priority_floor(rule_result, flags):
    """Minimum priority that code enforces regardless of what the AI says."""
    reasons, floor = [], -1
    if flags:
        floor = 3
        reasons.append("Safety keyword(s) in report: " + ", ".join(flags) + " -> Critical.")
    if rule_result["alarm_count"] >= 2:
        floor = max(floor, 3)
        reasons.append(f"{rule_result['alarm_count']} sensors in ALARM -> Critical.")
    elif rule_result["alarm_count"] == 1:
        floor = max(floor, 2)
        reasons.append("1 sensor in ALARM -> at least High.")
    elif rule_result["warning_count"] >= 1:
        floor = max(floor, 1)
        reasons.append("Sensor(s) in WARNING -> at least Medium.")
    return (LEVELS[floor] if floor >= 0 else None), reasons
