# Credits and licenses

- Star solving: [cedar-solve](https://github.com/smroid/cedar-solve) (Apache
  2.0), Steven Rosenthal's fork of [tetra3](https://github.com/esa/tetra3)
  (ESA, Apache 2.0); star detection:
  [cedar-detect](https://github.com/smroid/cedar-detect) (Steven Rosenthal).
- Star catalog: Hipparcos (ESA), in `data/database_1.npz`.
- Time and Earth orientation: [astropy](https://www.astropy.org) and
  [ERFA](https://github.com/liberfa/erfa).
- Map: [Leaflet](https://leafletjs.com) 1.9.4 (BSD 2-Clause, see
  `webui/static/vendor/LEAFLET-LICENSE`); map data © OpenStreetMap
  contributors.
- IMU driver: Adafruit CircuitPython LSM6DS.

PiSextant is released under the MIT License, see [LICENSE](../LICENSE).
Copyright (c) 2026 Gaspard de Francony. The third-party parts keep their own
licenses: Leaflet (BSD 2-Clause, `webui/static/vendor/LEAFLET-LICENSE`) and
the two files in `patches/`, which modify cedar-solve and stay under its
Apache 2.0 license.
