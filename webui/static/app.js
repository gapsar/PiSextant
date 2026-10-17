/* PiSextant web UI -- plain JS, no build step. */
"use strict";

// ---- map -------------------------------------------------------------------
var map = L.map("map", {worldCopyJump: true}).setView([0, 0], 3);
L.tileLayer("/tiles/{z}/{x}/{y}.png", {
  maxZoom: 16,
  attribution: "&copy; OpenStreetMap contributors (cached locally)"
}).addTo(map);

var gpsLine = L.polyline([], {color: "#e34948", weight: 3}).addTo(map);
var smoothLine = L.polyline([], {color: "#1baf7a", weight: 3}).addTo(map);
var fixDots = L.layerGroup().addTo(map);
var smoothMarker = null;
var truthMarker = null;
var fixCount = 0;

L.control.layers(null, {
  "GPS truth": gpsLine, "smoothed fixes": smoothLine, "raw fixes": fixDots
}).addTo(map);

function follow(latlng) {
  if (document.getElementById("follow").checked) map.panTo(latlng);
}
function addFix(lat, lon) {
  fixCount++;
  if (fixCount % 10 === 0) {                       // decimate raw dots
    L.circleMarker([lat, lon], {radius: 2, color: "#2a78d6",
                                opacity: 0.5}).addTo(fixDots);
  }
}
function addSmoothed(lat, lon) {
  smoothLine.addLatLng([lat, lon]);
  if (!smoothMarker) {
    smoothMarker = L.marker([lat, lon], {title: "smoothed fix"}).addTo(map);
  } else smoothMarker.setLatLng([lat, lon]);
  follow([lat, lon]);
}
function clearTruthMarker() {
  if (truthMarker) { map.removeLayer(truthMarker); truthMarker = null; }
}
function setTruthMarker(lat, lon) {
  if (!truthMarker) {
    truthMarker = L.circleMarker([lat, lon], {radius: 6, color: "#e34948",
                                              fillOpacity: 0.9}).addTo(map);
  } else truthMarker.setLatLng([lat, lon]);
}

// ---- helpers ---------------------------------------------------------------
function $(id) { return document.getElementById(id); }
function post(url, body) {
  return fetch(url, {method: "POST",
                     headers: {"Content-Type": "application/json"},
                     body: JSON.stringify(body || {})})
         .then(function (r) { return r.json(); });
}
function feed(id, line, max) {
  var el = $(id);
  el.textContent = (line + "\n" + el.textContent)
                   .split("\n").slice(0, max || 12).join("\n");
}
function dlKml(which) { window.location = "/api/kml/" + which; }

// ---- gravity chart (canvas, last 60 s) ------------------------------------
var gravHist = [];
function drawGravity() {
  var c = $("grav-chart"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (gravHist.length < 2) return;
  var t1 = gravHist[gravHist.length - 1].t, t0 = t1 - 60;
  var colors = ["#e34948", "#1baf7a", "#2a78d6"];
  var all = [];
  gravHist.forEach(function (s) { all.push(s.g[0], s.g[1], s.g[2]); });
  var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
  if (hi - lo < 1) { var m = (hi + lo) / 2; lo = m - 0.5; hi = m + 0.5; }
  for (var axis = 0; axis < 3; axis++) {
    ctx.beginPath();
    ctx.strokeStyle = colors[axis];
    gravHist.forEach(function (s, i) {
      var x = (s.t - t0) / 60 * c.width;
      var y = c.height - (s.g[axis] - lo) / (hi - lo) * c.height;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

// ---- temperature chart (canvas, whole run) --------------------------------
var tempHist = [];
function drawTemp() {
  var c = $("temp-chart"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (tempHist.length < 2) return;
  var t0 = tempHist[0][0], t1 = tempHist[tempHist.length - 1][0];
  if (t1 - t0 < 1) return;
  var vals = tempHist.map(function (p) { return p[1]; });
  var lo = Math.min.apply(null, vals) - 1, hi = Math.max.apply(null, vals) + 1;
  ctx.beginPath();
  ctx.strokeStyle = "#eda100";
  tempHist.forEach(function (p, i) {
    var x = (p[0] - t0) / (t1 - t0) * c.width;
    var y = c.height - (p[1] - lo) / (hi - lo) * c.height;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#777"; ctx.font = "10px sans-serif";
  ctx.fillText((hi - 1).toFixed(1) + " °C", 2, 10);
  ctx.fillText((lo + 1).toFixed(1) + " °C", 2, c.height - 2);
  ctx.fillText(((t1 - t0) / 60).toFixed(0) + " min", c.width - 44, c.height - 2);
}

// ---- error vs truth (rolling window) --------------------------------------
var errHist = [];   // [t_ms, err_smoothed]
function noteError(err, errSm) {
  var parts = [];
  if (errSm != null) {
    errHist.push([Date.now(), errSm]);
    while (errHist.length && Date.now() - errHist[0][0] > 300000) errHist.shift();
    var sorted = errHist.map(function (p) { return p[1]; }).sort(function (a, b) { return a - b; });
    var med = sorted[Math.floor(sorted.length / 2)];
    parts.push("error vs truth: smoothed " + errSm.toFixed(1) +
               " nmi (5-min median " + med.toFixed(1) + ")");
  }
  if (err != null) parts.push("instant " + err.toFixed(1) + " nmi");
  $("err-info").textContent = parts.join("   ");
}

// ---- SSE stream ------------------------------------------------------------
var es = new EventSource("/api/stream");
es.onmessage = function (ev) {
  var d = JSON.parse(ev.data);
  if (d.type === "solve") {
    var s = "#" + d.frame_id + " " + d.status +
            " cent=" + (d.n_centroids || 0) +
            (d.fix ? " fix " + d.fix[0].toFixed(3) + "," + d.fix[1].toFixed(3)
                   : "");
    feed("solve-feed", s, 10);
    if (d.fix) addFix(d.fix[0], d.fix[1]);
    if (d.smoothed) {
      addSmoothed(d.smoothed[0], d.smoothed[1]);
      $("fix-info").textContent =
        "smoothed: " + d.smoothed[0].toFixed(4) + ", " +
        d.smoothed[1].toFixed(4) + "  (n=" + d.smoothed_n + ")" +
        (d.fix ? "   instant: " + d.fix[0].toFixed(3) + ", " +
                 d.fix[1].toFixed(3) : "");
    }
    if (d.err_nmi != null || d.err_smoothed_nmi != null) {
      noteError(d.err_nmi, d.err_smoothed_nmi);
    }
  } else if (d.type === "gravity") {
    gravHist.push(d);
    while (gravHist.length && d.t - gravHist[0].t > 60) gravHist.shift();
    var g = d.g, n = Math.sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
    $("grav-info").textContent =
      "g = [" + g[0].toFixed(2) + ", " + g[1].toFixed(2) + ", " +
      g[2].toFixed(2) + "]  |g| = " + n.toFixed(4);
    drawGravity();
  } else if (d.type === "temp") {
    $("chip-temp").textContent = "temp: " + d.temp_raw + " °C";
    if (d.t != null && d.temp_raw != null) {
      tempHist.push([d.t, d.temp_raw]);
      drawTemp();
    }
  } else if (d.type === "gps") {
    gpsLine.addLatLng([d.lat, d.lon]);
    setTruthMarker(d.lat, d.lon);
    $("gps-info").textContent = "GPS: " + d.lat.toFixed(5) + ", " +
      d.lon.toFixed(5) + (d.sog_kn != null ? "  " + d.sog_kn + " kn" : "");
  } else if (d.type === "fixed_position") {
    showFixedPosition(d.position);
  } else if (d.type === "proc") {
    setProcChip(d.name, d.running);
  } else if (d.type === "proc_log") {
    feed("proc-log", "[" + d.name + "] " + d.line, 8);
  } else if (d.type === "run") {
    $("chip-run").textContent = "run: " + (d.run || "—");
    resetSeries();
    refreshRuns();
  } else if (d.type === "history_ready") {
    loadRunData();
  }
};

function resetSeries() {
  smoothLine.setLatLngs([]);
  fixDots.clearLayers();
  fixCount = 0;
}

function loadRunData() {
  fetch("/api/run_data").then(function (r) { return r.json(); })
  .then(function (d) {
    resetSeries();
    tempHist = (d.temps || []).filter(function (p) { return p[1] != null; });
    drawTemp();
    if (tempHist.length) $("chip-temp").textContent =
      "temp: " + tempHist[tempHist.length - 1][1] + " °C";
    errHist = [];
    (d.fixes || []).forEach(function (p) { addFix(p[1], p[2]); });
    smoothLine.setLatLngs((d.smoothed || []).map(function (p) {
      return [p[1], p[2]]; }));
    gpsLine.setLatLngs((d.gps || []).map(function (p) {
      return [p[1], p[2]]; }));
    var smoothed = d.smoothed || [];
    var last = smoothed[smoothed.length - 1];
    if (last) { addSmoothed(last[1], last[2]); map.setView([last[1], last[2]], 9); }
    showFixedPosition(d.fixed_position);
  });
}

// ---- processes -------------------------------------------------------------
var procRunning = {solver: false, capture: false};
function setProcChip(name, running) {
  procRunning[name] = running;
  if (name === "solver") $("chip-solver").textContent =
    "solver: " + (running ? "ON" : "off");
  if (name === "capture") $("chip-capture").textContent =
    "capture: " + (running ? "ON" : "off");
  var btn = $("btn-" + name);
  if (btn) btn.textContent = (running ? "stop " : "start ") + name;
}
function toggleProc(name) {
  var btn = $("btn-" + name);
  var req;
  if (procRunning[name]) {
    req = post("/api/proc/stop", {name: name});   // capture: up to 90 s to finalize
    if (btn) btn.textContent = "stopping " + name + "…";
  } else {
    var body = {name: name};
    if (name === "solver") body.config = $("solver-config").value;
    if (name === "capture") {
      body.duration_min = parseFloat($("cap-min").value) || 30;
      body.png_every = parseInt($("cap-png").value, 10) || 30;
    }
    req = post("/api/proc/start", body);
  }
  if (btn) btn.disabled = true;
  req.then(function (r) {
    if (!r.ok || /killed/.test(r.msg || "")) alert(name + ": " + r.msg);
  }).catch(function (e) { alert(name + ": " + e); })
  .then(function () { if (btn) btn.disabled = false; refreshStatus(); });
}

// ---- truth -----------------------------------------------------------------
function showFixedPosition(p) {
  if (p) {
    setTruthMarker(p[0], p[1]);
    $("pos-info").textContent = "fixed position: " + p[0].toFixed(6) + ", " + p[1].toFixed(6);
  } else {
    if (!gpsLine.getLatLngs().length) clearTruthMarker();
    $("pos-info").textContent = "no fixed position";
  }
}
function setPosition() {
  var lat = parseFloat($("pos-lat").value), lon = parseFloat($("pos-lon").value);
  if (isNaN(lat) || isNaN(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
    $("pos-info").textContent = "invalid: enter decimal degrees, e.g. 48.8566 and 2.3522";
    return;
  }
  post("/api/position", {lat: lat, lon: lon});
}
function clearPosition() { post("/api/position", {clear: true}); }

// ---- tiles -----------------------------------------------------------------
function tileBody(confirm) {
  var b = map.getBounds();
  return {zmin: parseInt($("tile-zmin").value),
          zmax: parseInt($("tile-zmax").value),
          lat1: b.getNorth(), lon1: b.getWest(),
          lat2: b.getSouth(), lon2: b.getEast(), confirm: confirm};
}
function tileEstimate() {
  post("/api/tiles/download", tileBody(false)).then(function (r) {
    $("tile-status").textContent = r.estimate + " tiles for this view/zooms";
  });
}
function tileDownload() {
  post("/api/tiles/download", tileBody(true)).then(function (r) {
    $("tile-status").textContent = r.ok ? ("downloading " + r.started +
      " tiles…") : r.msg;
    if (r.ok) pollTiles();
  });
}
function tilePurge() {
  post("/api/tiles/purge", {}).then(function (r) {
    $("tile-status").textContent = "purged " + r.removed +
      " blocked/invalid tiles (" + r.kept + " real tiles kept). Reloading map…";
    // force Leaflet to re-request tiles (busts the browser tile cache)
    map.eachLayer(function (l) { if (l.redraw) l.redraw(); });
    setTimeout(function () { map.setZoom(map.getZoom()); }, 200);
  });
}
function pollTiles() {
  fetch("/api/tiles/status").then(function (r) { return r.json(); })
  .then(function (s) {
    $("tile-status").textContent = s.done + "/" + s.total +
      (s.errors ? " (" + s.errors + " errors)" : "") +
      (s.message ? " — " + s.message : "");
    if (s.active) setTimeout(pollTiles, 2000);
    else if (!s.message) $("tile-status").textContent += " — done";
  });
}

// ---- runs + status ---------------------------------------------------------
function watchRun() { post("/api/watch", {run: $("run-select").value}); }
function watchLatest() { post("/api/watch", {run: "latest"}); }
function refreshRuns() {
  fetch("/api/runs").then(function (r) { return r.json(); })
  .then(function (d) {
    var sel = $("run-select");
    sel.innerHTML = "";
    (d.runs || []).forEach(function (r) {
      var o = document.createElement("option");
      o.value = o.textContent = r.run;
      sel.appendChild(o);
    });
  });
}
function refreshStatus() {
  fetch("/api/status").then(function (r) { return r.json(); })
  .then(function (s) {
    $("chip-run").textContent = "run: " + (s.run || "—");
    // NTP = best; RTC = fine without network (DS3231, ~1 s/week); none = time may be wrong
    var clk = s.clock_source;
    $("chip-ntp").textContent = "clock: " +
      (clk === "ntp" ? "NTP" : clk === "rtc" ? "RTC" : clk === "none" ? "NOT SET!" : "?");
    $("chip-ntp").className = "chip" +
      (clk === "rtc" ? " warn" : clk === "none" ? " bad" : "");
    $("chip-disk").textContent = "disk: " +
      (s.disk_free_gb === null ? "?" : s.disk_free_gb + " GB");
    setProcChip("solver", s.procs.solver);
    setProcChip("capture", s.procs.capture);
    renderChecklist(s.checklist);
  });
}
function renderChecklist(cl) {
  var el = $("checklist");
  if (!cl || !cl.items || !cl.items.length) {
    el.textContent = "no active run";
    el.className = "hint";
    return;
  }
  el.className = "";
  var html = "";
  cl.items.forEach(function (it) {
    var mark = it.ok === true ? "✅" : it.ok === false ? "❌" : it.ok === "warn" ? "⚠️" : "…";
    html += "<div class='check-item'>" + mark + " " + it.name +
            " <span class='hint'>— " + (it.detail || "") + "</span></div>";
  });
  el.innerHTML = html;
}
refreshRuns();
refreshStatus();
loadRunData();
setInterval(refreshStatus, 10000);
