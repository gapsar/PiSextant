# Running

## From the web page

1. Wait about 10 minutes after power-on: the checklist shows when the IMU
   temperature is stable.
2. Check the chips at the top: the clock shows `NTP` (best) or `RTC` (yellow,
   good to about 1 s per week).
3. Start the **solver**, wait about 30 s, then start the **capture** with a
   duration and an image rate ("save 1 image in N"; images are only for
   offline work).
4. The map shows each fix (dots) and the 60-second average (line). The 60 s
   average is the position to use.

## From the command line (for example over ssh)

```bash
cd ~/PiSextant
~/pisextant-solver/bin/python solve_worker.py configs/solver.json &
sleep 30
~/pisextant-hw/bin/python capture_run_wave.py --duration-min 240 --png-every 30
```

Or through the web server, which keeps the map live:

```bash
curl -s -X POST localhost:8080/api/proc/start -H 'Content-Type: application/json' -d '{"name":"solver"}'
curl -s -X POST localhost:8080/api/proc/start -H 'Content-Type: application/json' \
     -d '{"name":"capture","duration_min":240,"png_every":30}'
curl -s localhost:8080/api/status
curl -s -X POST localhost:8080/api/proc/stop -H 'Content-Type: application/json' -d '{"name":"capture"}'
```

Ctrl-C or the stop button finishes the run cleanly.

Main capture options: `--duration-min`, `--png-every N` (default 10; all
images cost about 125 MB per minute), `--smooth-s` (averaging window, default
60 s). `--wave-s` (gyro-based wave compensation) is off by default: the gyro
offset of the ISM330DHCX is not corrected yet and makes it worse.
