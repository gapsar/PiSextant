# Checking a run against a known track

Put the truth in the run folder as `gps.jsonl`, one JSON object per line:

```json
{"type": "fix", "t_gps": 1792152000, "lat": 22.40240, "lon": 125.27118}
```

`t_gps` is the UTC time in Unix seconds. The tools interpolate linearly between
the points, so a few points on a straight course are enough; add points at the
start and the end to cover the whole run. `nmea_logger.py` writes this file
from a GPS on a serial port.

```bash
python sea_eval.py runs/<run>                  # offset, 60 s fix error, table per 10 min
python sea_eval.py runs/<run> --bin-min 30
python drift_check.py runs/<run> --position=48.8566,2.3522   # drift with temperature / time
```
