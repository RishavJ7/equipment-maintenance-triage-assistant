from app import rules


def ev(t, r):
    return rules.evaluate(t, r)


def by(res, s):
    return next(i for i in res["items"] if i["sensor"] == s)


def test_thresholds_high_and_low():
    r = ev("compressor", [{"sensor": "discharge_temp", "value": 111}, {"sensor": "oil_pressure", "value": 1.4}])
    assert by(r, "discharge_temp")["status"] == "warning"
    assert by(r, "oil_pressure")["status"] == "alarm"
    assert r["overall"] == "alarm"


def test_normal():
    r = ev("pump", [{"sensor": "bearing_temp", "value": 60}])
    assert r["overall"] == "normal"


def test_boundaries_inclusive():
    assert by(ev("pump", [{"sensor": "bearing_temp", "value": 90}]), "bearing_temp")["status"] == "alarm"


def test_missing_sensors_reported_not_guessed():
    r = ev("motor", [])
    assert r["overall"] == "insufficient_data"
    assert all(i["status"] == "not_provided" for i in r["items"])


def test_conflicting_readings_use_worst_case():
    r = ev("compressor", [{"sensor": "discharge_temp", "value": 135}, {"sensor": "discharge_temp", "value": 100}])
    i = by(r, "discharge_temp")
    assert i["conflict"] and i["status"] == "alarm" and i["used_value"] == 135
    assert any("conflicting" in n for n in r["data_quality_notes"])


def test_implausible_value_ignored():
    r = ev("compressor", [{"sensor": "vibration", "value": 999}])
    assert by(r, "vibration")["status"] == "invalid"


def test_unknown_sensor_is_observation_only():
    r = ev("compressor", [{"sensor": "humidity", "value": 50}])
    assert by(r, "humidity")["status"] == "unchecked"


def test_text_vs_sensor_conflict():
    r = ev("compressor", [{"sensor": "discharge_temp", "value": 125}])
    assert rules.text_conflicts("Temperature is normal but it keeps tripping", r["items"])
    assert not rules.text_conflicts("Keeps tripping", r["items"])


def test_priority_floor():
    r = ev("compressor", [{"sensor": "discharge_temp", "value": 125}, {"sensor": "oil_pressure", "value": 1.0}])
    assert rules.priority_floor(r, [])[0] == "High" or rules.priority_floor(r, [])[0] == "Critical"
    r1 = ev("compressor", [{"sensor": "oil_pressure", "value": 1.0}])
    assert rules.priority_floor(r1, [])[0] == "High"
    assert rules.priority_floor(ev("pump", []), ["smoke"])[0] == "Critical"
    assert rules.priority_floor(ev("pump", [{"sensor": "bearing_temp", "value": 50}]), [])[0] is None
