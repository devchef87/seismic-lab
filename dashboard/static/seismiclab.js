'use strict';

const CONFIG = {
    STREAM_URL: '/api/stream',
    FAULT_RISK_INTERVAL: 60000,
    HEATMAP_INTERVAL: 300000,
    RECONNECT_DELAY: 3000,
    MAP_CENTER: [2, 220],
    MAP_ZOOM: 1.5,
};

const state = {
    earthquakes: [],
    signals: {},
    signalHistory: {},
    eqMarkers: {},
    streamConnected: false,
    topFaultZone: null,
    seenQuakeIds: new Set(),
    lastSeedlinkAlert: 0,
    audioCtx: null,
    seismoStation: null,
    seismoBuffer: [],
    seismoEnvelope: [],
    seismoMode: 'raw',
    seismoScale: 'live',
    seismoSampleRate: 5,
    seismoInterval: null,
    seismoWs: null,
    seismoRaf: null,
    stationMarkers: {},
    seedlinkStations: [],
    embeddedSeismos: {},
    dartMarkers: {},
    volcanoMarkers: {},
    tidalMarkers: {},
    tidalData: null,
    filterMag: 3.5,
    filterHours: 24,
};

function getAudioCtx() {
    if (!state.audioCtx) state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    return state.audioCtx;
}

function playQuakeAlert(mag) {
    try {
        const ctx = getAudioCtx();
        if (ctx.state === 'suspended') ctx.resume();
        const now = ctx.currentTime;

        const beeps = mag >= 6 ? 5 : mag >= 5 ? 4 : 3;
        const freq = mag >= 6 ? 880 : mag >= 5 ? 740 : 620;
        const tempo = mag >= 6 ? 0.14 : 0.18;

        for (let i = 0; i < beeps; i++) {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, now);
            osc.frequency.setValueAtTime(freq * 1.02, now + tempo * 0.5);

            const t = now + i * (tempo + 0.06);
            gain.gain.setValueAtTime(0, t);
            gain.gain.linearRampToValueAtTime(0.15, t + 0.015);
            gain.gain.setValueAtTime(0.15, t + tempo * 0.6);
            gain.gain.exponentialRampToValueAtTime(0.001, t + tempo);

            osc.start(t);
            osc.stop(t + tempo + 0.01);
        }
    } catch (e) {
        // audio not available
    }
}

function checkNewQuakes(quakes) {
    if (state.seenQuakeIds.size === 0) {
        for (const eq of quakes) state.seenQuakeIds.add(eq.id);
        return;
    }
    let alerted = false;
    for (const eq of quakes) {
        if (!state.seenQuakeIds.has(eq.id)) {
            state.seenQuakeIds.add(eq.id);
            if (eq.magnitude >= 4.0 && !alerted) {
                playQuakeAlert(eq.magnitude);
                alerted = true;
            }
        }
    }
}

const SIGNAL_INFO = {
    'noaa_swpc__kp_index': {
        name: 'Kp Index',
        what: 'The planetary K-index (Kp) is a 3-hour global measure of geomagnetic disturbance derived from ground-based magnetometer stations worldwide. It quantifies how much Earth\'s magnetic field is deviating from its quiet baseline.',
        scale: '0 to 9 (quasi-logarithmic). Each integer step represents roughly a 2x increase in disturbance amplitude.',
        baseline: 'Kp 0–2 is geomagnetically quiet. This is the normal background state for most days.',
        watch: 'Kp 3–4 is unsettled — minor geomagnetic activity, usually not significant. Kp 5 is a minor geomagnetic storm (G1). Kp 6–7 is moderate to strong (G2–G3). Kp 8–9 is severe to extreme storm (G4–G5) — rare, typically from major CME impacts.',
        why: 'Geomagnetic storms induce telluric currents in Earth\'s crust. Some peer-reviewed studies (Marchetti et al., Urata et al.) have found statistical correlation between elevated Kp periods and increased seismicity in already-stressed fault zones within 24–72 hours. We use Kp as supporting context, not a primary predictor.',
        source: 'NOAA Space Weather Prediction Center (SWPC)',
    },
    'noaa_swpc__dst_index': {
        name: 'Dst Index',
        what: 'The Disturbance Storm Time (Dst) index measures the intensity of Earth\'s ring current — a toroidal electric current flowing westward around Earth at 3–8 Earth radii. It\'s the primary indicator of geomagnetic storm intensity.',
        scale: 'Measured in nanotesla (nT). Ranges from roughly +20 nT (compressed field) to -500+ nT (extreme storm).',
        baseline: 'Dst between +20 and -20 nT is quiet. Normal fluctuations stay within this range.',
        watch: 'Dst -30 to -50 nT: weak storm developing. Dst -50 to -100 nT: moderate storm (watch closely). Dst -100 to -200 nT: strong storm. Dst below -200 nT: severe/extreme storm. The rate of change (dDst/dt) is critical — a sudden drop of >8 nT/hr signals a storm commencement (SSC), which our onset detector flags.',
        why: 'A rapidly declining Dst means the ring current is intensifying quickly, typically from CME shock arrival. This sudden compression generates ground-level induced electric fields. Our Tier 1 detector uses dDst/dt as a leading indicator — the rate of decline matters more than the absolute value.',
        source: 'NOAA SWPC / Kyoto WDC',
    },
    'goes_magnetometer__goes_mag_total': {
        name: 'GOES Mag |B| (Total Field)',
        what: 'The total magnetic field magnitude measured by GOES satellites at geosynchronous orbit (~35,786 km altitude). This is a direct measurement of the magnetospheric field strength at Earth\'s boundary with the solar wind.',
        scale: 'Measured in nanotesla (nT). Typical quiet values are 80–110 nT at geosynchronous orbit.',
        baseline: '80–110 nT during quiet conditions. The field strength at geosynchronous orbit depends on local time (day/night asymmetry).',
        watch: 'Sudden jump of >10–15 nT: magnetospheric compression from solar wind pressure pulse or CME shock — this is what you\'re seeing now. Sudden drop of >15 nT: magnetospheric expansion, possible substorm. Values below 50 nT: extreme compression has pushed the magnetopause inside geosynchronous orbit (rare, severe). This is our fastest-responding geomag indicator — it reacts before Kp or Dst update.',
        why: 'GOES |B| responds to solar wind changes within minutes, while Kp and Dst lag by hours. A sudden compression means increased solar wind pressure is hitting Earth\'s magnetosphere right now. Our onset detector flags jumps >10 nT as "magnetospheric compression" — a leading indicator that Kp and Dst will follow.',
        source: 'NOAA GOES-16/18 magnetometer',
    },
    'goes_magnetometer__goes_mag_hp': {
        name: 'GOES Mag Hp (Parallel Component)',
        what: 'The Hp component of the magnetic field at GOES — the component parallel to Earth\'s dipole axis. This is the most sensitive component to magnetospheric compression and expansion.',
        scale: 'Measured in nanotesla (nT). Quiet values typically 40–70 nT.',
        baseline: '40–70 nT during quiet periods. Tracks closely with |B| but more sensitive to north-south field changes.',
        watch: 'Hp tracks compression events similarly to |B|. A sharp increase means the field is being compressed. A sharp decrease (especially going negative) indicates the north-south component has flipped — a signature of substorm onset or southward IMF turning, which drives stronger geomagnetic coupling.',
        why: 'Hp is a complementary measurement to |B|. When both jump simultaneously, it confirms a true compression event rather than instrument noise or local effects.',
        source: 'NOAA GOES-16/18 magnetometer',
    },
    'usgs_magnetometer__mag_z': {
        name: 'Ground Mag Z (Vertical Component)',
        what: 'The vertical component of Earth\'s magnetic field measured by ground-based magnetometer observatories. Unlike GOES (which measures the field in space), this measures the field at Earth\'s surface, directly where telluric currents are induced.',
        scale: 'Measured in nanotesla (nT). Absolute values vary by station latitude (typically 20,000–60,000 nT). We track the variation from baseline, not absolute values.',
        baseline: 'Varies by location. What matters is the rate of change (dB/dt) — quiet conditions show smooth, slow variations of <1 nT/min.',
        watch: 'dB/dt > 5 nT/min: notable. dB/dt > 10 nT/min: significant (power grid operators start watching). dB/dt > 50 nT/min: extreme (associated with GIC events). Our detector flags when dB/dt is >2x above its own recent baseline — meaning the rate-of-change has at least doubled.',
        why: 'Ground dB/dt is directly proportional to the electric field induced in the crust (Faraday\'s law). Rapid magnetic field changes generate telluric currents that flow through conductive structures including fault zones. This is the most direct proxy we have for the actual stress-adding mechanism that connects geomagnetic events to seismicity.',
        source: 'USGS Geomagnetism Program',
    },
    'intermagnet__imag_f': {
        name: 'INTERMAGNET F (Total Field)',
        what: 'The total magnetic field intensity from the INTERMAGNET global network of magnetic observatories. This is a high-quality, calibrated measurement of the complete geomagnetic field at ground level.',
        scale: 'Measured in nanotesla (nT). Absolute values range from ~25,000 nT (equator) to ~65,000 nT (poles).',
        baseline: 'Extremely stable during quiet conditions. Daily variations (Sq) are typically 20–50 nT with a smooth sinusoidal shape driven by solar heating of the ionosphere.',
        watch: 'Deviations >100 nT from expected Sq pattern: storm effects reaching the ground. Rapid fluctuations (Pi2 pulsations, 40–150 second period): substorm onset signature. Irregular high-frequency variations: storm-time disturbance field. Compare with GOES — if GOES shows compression but INTERMAGNET F is quiet, the disturbance hasn\'t reached ground level yet.',
        why: 'INTERMAGNET provides ground truth for what\'s actually happening at Earth\'s surface. GOES tells us what\'s coming, ground magnetometers tell us what\'s arrived. The combination gives us a complete picture of the geomagnetic disturbance chain from space to crust.',
        source: 'INTERMAGNET international network',
    },
};

function openSignalInfo(key) {
    const info = SIGNAL_INFO[key];
    if (!info) return;

    let modal = document.getElementById('signal-info-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'signal-info-modal';
        modal.className = 'signal-info-modal';
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeSignalInfo();
        });
        document.body.appendChild(modal);
    }

    modal.innerHTML = `
        <div class="signal-info-panel">
            <button class="signal-info-close" onclick="closeSignalInfo()">&times;</button>
            <div class="signal-info-title">${info.name}</div>
            <div class="signal-info-section">
                <div class="signal-info-heading">What is this?</div>
                <div class="signal-info-body">${info.what}</div>
            </div>
            <div class="signal-info-section">
                <div class="signal-info-heading">Scale</div>
                <div class="signal-info-body">${info.scale}</div>
            </div>
            <div class="signal-info-section">
                <div class="signal-info-heading">Normal baseline</div>
                <div class="signal-info-body">${info.baseline}</div>
            </div>
            <div class="signal-info-section">
                <div class="signal-info-heading">What to watch for</div>
                <div class="signal-info-body">${info.watch}</div>
            </div>
            <div class="signal-info-section">
                <div class="signal-info-heading">Why we track it</div>
                <div class="signal-info-body">${info.why}</div>
            </div>
            <div class="signal-info-source">Source: ${info.source}</div>
        </div>
    `;
    requestAnimationFrame(() => modal.classList.add('visible'));
}

function closeSignalInfo() {
    const modal = document.getElementById('signal-info-modal');
    if (modal) modal.classList.remove('visible');
}

// ── Map (inside grid center cell) ────────────────────────
const map = new maplibregl.Map({
    container: 'map',
    style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    center: [CONFIG.MAP_CENTER[1], CONFIG.MAP_CENTER[0]],
    zoom: CONFIG.MAP_ZOOM,
    pitch: 40,
    maxPitch: 60,
    attributionControl: false,
    renderWorldCopies: true,
});

let mapReady = false;
const hoverPopup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 10, maxWidth: '280px' });

function setTooltip(marker, html) {
    marker._tipHtml = html;
    const el = marker.getElement();
    if (el._hasTip) return;
    el._hasTip = true;
    el.addEventListener('mouseenter', () => {
        hoverPopup.setHTML(marker._tipHtml).setLngLat(marker.getLngLat()).addTo(map);
    });
    el.addEventListener('mouseleave', () => hoverPopup.remove());
}

// ── Plate boundaries ─────────────────────────────────────
const PLATE_BOUNDARIES = [
    [[-55,-70],[-46,-75],[-38,-73],[-33,-71],[-24,-70],[-18,-70],[-12,-76],[-5,-80],[2,-78]],
    [[2,-78],[8,-83],[14,-93],[17,-100],[20,-106],[23,-110]],
    [[23,-110],[28,-113],[32,-117],[34,-120],[37,-122],[40,-125],[44,-127],[48,-128],[51,-130]],
    [[51,-130],[55,-136],[57,-150],[56,-158],[53,-167],[52,-172],[51,178],[50,172],[50,165]],
    [[50,165],[52,160],[50,157],[46,152],[42,146],[38,142],[35,140],[32,135],[30,132]],
    [[30,132],[27,128],[24,124],[20,122],[16,121],[12,124],[8,126],[4,127],[0,124],[-3,120],[-7,115],[-9,110],[-7,105],[-5,100],[-3,95]],
    [[-46,167],[-42,174],[-38,178],[-34,-178],[-28,-176],[-22,-174],[-16,-173]],
    [[36,-10],[37,-5],[37,0],[38,5],[39,10],[38,15],[37,20],[38,28],[39,35],[38,42],[36,48],[34,52],[32,57]],
    [[32,57],[31,62],[30,67],[30,72],[29,77],[28,82],[27,86],[28,92],[26,95],[22,96]],
    [[65,-18],[62,-22],[55,-30],[45,-28],[35,-35],[25,-45],[15,-46],[5,-32],[0,-14],[-10,-13],[-20,-12],[-30,-14],[-40,-15],[-50,-8],[-55,-5]],
    [[12,42],[8,38],[4,36],[0,33],[-4,30],[-8,32],[-12,34],[-16,35],[-20,35]],
];

function buildPlateGeoJSON() {
    const features = PLATE_BOUNDARIES.map(coords => ({
        type: 'Feature',
        geometry: {
            type: 'LineString',
            coordinates: coords.map(([lat, lon]) => [lon, lat])
        },
        properties: {}
    }));
    return { type: 'FeatureCollection', features };
}

// ── map.on('load') ──────────────────────────────────────
map.on('load', () => {
    // 3D terrain
    map.addSource('terrain-dem', {
        type: 'raster-dem',
        tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
        encoding: 'terrarium',
        tileSize: 256,
        maxzoom: 15,
    });
    map.addLayer({
        id: 'hillshade',
        type: 'hillshade',
        source: 'terrain-dem',
        paint: {
            'hillshade-illumination-direction': 315,
            'hillshade-exaggeration': 0.4,
            'hillshade-shadow-color': 'rgba(0,0,0,0.5)',
            'hillshade-highlight-color': 'rgba(255,255,255,0.12)',
            'hillshade-accent-color': 'rgba(80,120,180,0.15)',
        }
    });
    map.setTerrain({ source: 'terrain-dem', exaggeration: 1.5 });
    map.setSky({
        'sky-color': '#060810',
        'sky-horizon-blend': 0.3,
        'horizon-color': '#0a0e1a',
        'horizon-fog-blend': 0.8,
        'fog-color': '#060810',
        'fog-ground-blend': 0.9,
    });

    // Plate boundaries
    map.addSource('plate-boundaries', { type: 'geojson', data: buildPlateGeoJSON() });
    map.addLayer({
        id: 'plate-boundaries',
        type: 'line',
        source: 'plate-boundaries',
        paint: { 'line-color': 'rgba(255,255,255,0.18)', 'line-width': 1, 'line-dasharray': [5, 5] }
    });

    // Seismicity heatmap
    map.addSource('eq-heat', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({
        id: 'eq-heatmap',
        type: 'heatmap',
        source: 'eq-heat',
        paint: {
            'heatmap-radius': ['interpolate', ['linear'], ['zoom'],
                0, 40, 2, 35, 4, 35, 6, 40, 8, 50, 10, 60],
            'heatmap-weight': ['get', 'weight'],
            'heatmap-intensity': ['interpolate', ['linear'], ['zoom'],
                0, 0.7, 2, 0.8, 4, 0.9, 6, 1, 10, 1.2],
            'heatmap-opacity': ['interpolate', ['linear'], ['zoom'],
                0, 0.45, 3, 0.5, 6, 0.55, 10, 0.55],
            'heatmap-color': [
                'interpolate', ['linear'], ['heatmap-density'],
                0, 'transparent', 0.15, 'rgba(30,60,140,0.3)',
                0.35, 'rgba(60,100,180,0.4)', 0.55, 'rgba(140,120,60,0.5)',
                0.75, 'rgba(200,120,40,0.6)', 1.0, 'rgba(200,60,30,0.7)'
            ]
        }
    });

    // Fault risk heatmap
    map.addSource('fault-heat', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({
        id: 'fault-heatmap',
        type: 'heatmap',
        source: 'fault-heat',
        paint: {
            'heatmap-radius': ['interpolate', ['linear'], ['zoom'],
                0, 22, 2, 22, 4, 25, 6, 30, 8, 40, 10, 50],
            'heatmap-weight': ['get', 'weight'],
            'heatmap-intensity': ['interpolate', ['linear'], ['zoom'],
                0, 0.45, 3, 0.55, 6, 0.6, 10, 0.7],
            'heatmap-opacity': ['interpolate', ['linear'], ['zoom'],
                0, 0.3, 3, 0.33, 6, 0.35, 10, 0.35],
            'heatmap-color': [
                'interpolate', ['linear'], ['heatmap-density'],
                0, 'transparent', 0.4, 'transparent',
                0.5, 'rgba(220,90,20,0.06)', 0.65, 'rgba(240,70,15,0.12)',
                0.8, 'rgba(255,50,10,0.25)', 0.9, 'rgba(255,30,0,0.4)',
                1.0, 'rgba(255,20,0,0.55)'
            ]
        }
    });

    // Fault risk line overlays
    map.addSource('fault-lines', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({
        id: 'fault-lines-glow',
        type: 'line',
        source: 'fault-lines',
        paint: { 'line-color': ['get', 'color'], 'line-width': ['get', 'glowWidth'], 'line-opacity': ['get', 'glowOpacity'] },
        layout: { 'line-cap': 'round' }
    });
    map.addLayer({
        id: 'fault-lines-core',
        type: 'line',
        source: 'fault-lines',
        paint: { 'line-color': ['get', 'color'], 'line-width': ['get', 'width'], 'line-opacity': ['get', 'opacity'] },
        layout: { 'line-cap': 'round' }
    });

    mapReady = true;
    refreshFaultRisk();
    refreshHeatmap();
    if (_swarmWatchData) updateSwarmPolygons(_swarmWatchData);
    if (_latestThreats) updateZoneBoundaries(_latestThreats);
});

// ── Region name guesser ──────────────────────────────────
const REGION_NAMES = [
    [[34,-119], 'Southern California'],
    [[37,-122], 'San Francisco Bay'],
    [[41,-124], 'Cascadia (N. California)'],
    [[46,-123], 'Pacific Northwest'],
    [[50,-130], 'British Columbia'],
    [[55,-136], 'Gulf of Alaska'],
    [[57,-150], 'Alaska Peninsula'],
    [[53,-167], 'Aleutian Islands'],
    [[51,178],  'Western Aleutians'],
    [[38,142],  'Japan Trench'],
    [[35,140],  'Nankai Trough, Japan'],
    [[30,132],  'Ryukyu Arc, Japan'],
    [[20,122],  'Philippines / Taiwan'],
    [[8,126],   'Mindanao, Philippines'],
    [[-3,120],  'Banda Sea, Indonesia'],
    [[-7,115],  'Java, Indonesia'],
    [[-9,110],  'Sunda Strait'],
    [[-5,100],  'Sumatra, Indonesia'],
    [[22,96],   'Myanmar / Bay of Bengal'],
    [[38,28],   'Aegean Sea, Turkey'],
    [[37,20],   'Greece / Ionian Sea'],
    [[39,35],   'Central Turkey'],
    [[36,48],   'Iran'],
    [[32,57],   'Eastern Iran'],
    [[30,67],   'Pakistan / Afghanistan'],
    [[28,82],   'Nepal / Himalayas'],
    [[65,-18],  'Iceland'],
    [[45,-28],  'Mid-Atlantic Ridge'],
    [[0,-14],   'Equatorial Atlantic'],
    [[-30,-14], 'South Atlantic'],
    [[-55,-5],  'Scotia Arc, S. Atlantic'],
    [[12,42],   'Red Sea / Afar'],
    [[-8,32],   'East African Rift'],
    [[-20,35],  'Mozambique Channel'],
    [[-46,167], 'New Zealand (S. Island)'],
    [[-38,178], 'New Zealand (N. Island)'],
    [[-22,-174],'Tonga Trench'],
    [[-16,-173],'Samoa / Tonga'],
    [[-33,-71], 'Central Chile'],
    [[-18,-70], 'Peru / Chile'],
    [[-5,-80],  'Ecuador / N. Peru'],
    [[2,-78],   'Colombia / Ecuador'],
    [[8,-83],   'Central America'],
    [[14,-93],  'Mexico (S. Coast)'],
    [[20,-106], 'Western Mexico'],
    [[23,-110], 'Baja California'],
    [[28,-113], 'Gulf of California'],
];

function guessRegionName(lat, lon) {
    let best = null, bestDist = Infinity;
    for (const [coords, name] of REGION_NAMES) {
        const dlat = lat - coords[0], dlon = lon - coords[1];
        const d = dlat * dlat + dlon * dlon;
        if (d < bestDist) { bestDist = d; best = name; }
    }
    return best || `${lat.toFixed(1)}, ${lon.toFixed(1)}`;
}

// ── Threat monitor (right sidebar) ───────────────────────
function threatColor(level) {
    if (level === 'ALERT')    return 'rgba(220,70,50,0.85)';
    if (level === 'ELEVATED') return 'rgba(210,160,50,0.8)';
    if (level === 'WATCH')    return 'rgba(160,170,70,0.7)';
    return 'rgba(120,160,180,0.4)';
}

function renderZoneRow(z) {
    const levelCls = `zone-level-${z.level.toLowerCase()}`;
    const isHigh = z.level === 'ELEVATED' || z.level === 'ALERT';

    let signalLines = '';
    if (z.signals && z.signals.length > 0) {
        const maxSigs = isHigh ? 3 : 2;
        const boost = z.signals.filter(s => s.signal.startsWith('Gravitational stress') || s.signal.startsWith('ML forecast'));
        const other = z.signals.filter(s => !s.signal.startsWith('Gravitational stress') && !s.signal.startsWith('ML forecast'));
        const picked = [...other.slice(0, isHigh ? 2 : 1), ...boost].slice(0, maxSigs);
        if (picked.length > 0) {
            signalLines = '<div class="zone-signals">' +
                picked.map(s =>
                    `<div class="zone-signal-line">${s.signal}</div>`
                ).join('') + '</div>';
        }
    }

    return `<div class="zone-row" data-lat="${z.center[0]}" data-lon="${z.center[1]}" data-name="${z.name}">
        <div class="zone-name">${z.name}</div>
        <span class="zone-level ${levelCls}">${z.level}</span>
    </div>${signalLines}`;
}

function updateThreatMonitor(zones) {
    const el = document.getElementById('zone-list');
    if (!el) return;
    if (!zones || zones.length === 0) {
        el.innerHTML = '<div class="panel-muted">No zones scanned</div>';
        return;
    }

    state.topFaultZone = zones[0];
    const alertZones = zones.filter(z => z.level === 'ALERT' || z.level === 'ELEVATED').slice(0, 10);
    const watchZones = zones.filter(z => z.level === 'WATCH');

    let html = '';
    for (const z of alertZones) html += renderZoneRow(z);

    if (alertZones.length === 0) {
        html = '<div class="panel-muted">No active alerts</div>';
    }

    if (watchZones.length > 0) {
        html += `<div class="zone-watch-toggle" id="zone-watch-toggle"><span class="toggle-arrow">&#9662;</span> ${watchZones.length} Watch</div>`;
        html += '<div class="zone-watch-list" id="zone-watch-list">';
        for (const z of watchZones) html += renderZoneRow(z);
        html += '</div>';
    }

    el.innerHTML = html;

    document.getElementById('zone-watch-toggle')?.addEventListener('click', () => {
        const toggle = document.getElementById('zone-watch-toggle');
        const list = document.getElementById('zone-watch-list');
        toggle.classList.toggle('expanded');
        list.classList.toggle('expanded');
    });

    el.querySelectorAll('.zone-row').forEach(item => {
        item.addEventListener('click', () => {
            predictLocation(
                parseFloat(item.dataset.lat),
                parseFloat(item.dataset.lon),
                item.dataset.name
            );
        });
    });
}

function updateZoneBoundaries(zones) {
    _latestThreats = zones;
    if (!mapReady || !zones) return;

    const features = zones
        .filter(z => z.level !== 'NORMAL' && ZONE_BOUNDARIES[z.zone_id])
        .map(z => ({
            type: 'Feature',
            properties: {
                zone_id: z.zone_id,
                name: z.name,
                level: z.level,
                threat_score: z.threat_score || 0,
            },
            geometry: {
                type: 'Polygon',
                coordinates: [ZONE_BOUNDARIES[z.zone_id]],
            }
        }));

    const geo = { type: 'FeatureCollection', features };
    if (map.getSource('zone-boundaries')) {
        map.getSource('zone-boundaries').setData(geo);
    } else {
        map.addSource('zone-boundaries', { type: 'geojson', data: geo });
        map.addLayer({
            id: 'zone-boundaries-fill',
            type: 'fill',
            source: 'zone-boundaries',
            paint: {
                'fill-color': [
                    'match', ['get', 'level'],
                    'ALERT', 'rgba(220,70,50,0.10)',
                    'ELEVATED', 'rgba(230,160,50,0.08)',
                    'WATCH', 'rgba(255,255,255,0.04)',
                    'rgba(0,0,0,0)'
                ],
                'fill-opacity': 1,
            }
        });

        map.on('mouseenter', 'zone-boundaries-fill', (e) => {
            map.getCanvas().style.cursor = 'pointer';
            const p = e.features[0].properties;
            const colors = ZONE_LEVEL_COLORS[p.level] || ZONE_LEVEL_COLORS.WATCH;
            const html = `<div style="font-size:11px"><b>${p.name}</b> · <span style="color:${colors.label};font-weight:600">${p.level}</span></div>`;
            hoverPopup.setHTML(html).setLngLat(e.lngLat).addTo(map);
        });
        map.on('mouseleave', 'zone-boundaries-fill', () => {
            map.getCanvas().style.cursor = '';
            hoverPopup.remove();
        });
        map.on('click', 'zone-boundaries-fill', (e) => {
            const target = e.originalEvent?.target;
            if (target && target !== map.getCanvas() && !target.classList.contains('maplibregl-canvas')) return;
            const interactiveLayers = ['swarm-cells-fill', 'swarm-zone-quakes-dot', 'eq-circles', 'eq-circles-outer'].filter(l => map.getLayer(l));
            if (interactiveLayers.length > 0) {
                const hits = map.queryRenderedFeatures(e.point, { layers: interactiveLayers });
                if (hits.length > 0) return;
            }
            const p = e.features[0].properties;
            const center = ZONE_BOUNDARIES[p.zone_id];
            if (center) {
                const lats = center.map(c => c[1]);
                const lons = center.map(c => c[0]);
                const cLat = (Math.min(...lats) + Math.max(...lats)) / 2;
                const cLon = (Math.min(...lons) + Math.max(...lons)) / 2;
                predictLocation(cLat, cLon, p.name);
            }
        });
    }
}

function updateDartSummary(data) {
    const el = document.getElementById('dart-summary');
    if (!el || !data || !data.stations) return;

    const stations = data.stations;
    const event = stations.filter(s => s.mode === 'event');
    const tsunami = stations.filter(s => s.mode === 'tsunami');
    const elevated = stations.filter(s => s.mode === 'normal' && (s.deviation || 0) >= 1.5);
    const active = [...tsunami, ...event, ...elevated];

    const evtCls = event.length > 0 ? ' dart-count-event' : '';
    const tsuCls = tsunami.length > 0 ? ' dart-count-tsunami' : '';
    const elevCls = elevated.length > 0 ? ' dart-count-elevated' : '';
    let html = `<div class="dart-status-row">
        <span class="dart-count">${stations.length}</span> buoys
        <span style="opacity:0.12;margin:0 2px">/</span>
        <span class="dart-count${evtCls}">${event.length}</span> event
        <span style="opacity:0.12;margin:0 2px">/</span>
        <span class="dart-count${tsuCls}">${tsunami.length}</span> tsunami
        <span style="opacity:0.12;margin:0 2px">/</span>
        <span class="dart-count${elevCls}">${elevated.length}</span> signal
    </div>`;

    if (active.length > 0) {
        html += '<div class="dart-active-list">';
        for (const s of active) {
            const cls = s.mode === 'tsunami' ? 'tsunami' : s.mode === 'event' ? 'event' : 'elevated';
            const label = s.mode === 'normal' ? `${s.deviation.toFixed(1)}σ` : s.mode;
            html += `<div class="dart-active-row ${cls}">
                <span class="dart-active-id">${s.station_id}</span>
                <span class="dart-active-region">${s.region || '—'}</span>
                <span class="dart-active-mode">${label}</span>
            </div>`;
        }
        html += '</div>';
    }

    el.innerHTML = html;
}

function updateVolcanicSummary(data) {
    if (!data || !data.volcanoes) return;
    state._volcanicData = data;
    if (window._updateLegendVolcanicCounts) {
        window._updateLegendVolcanicCounts(data);
    }
}

function _showVolcanoMarker(vid) {
    const data = state._volcanicData;
    if (!data) return;
    const v = data.volcanoes.find(x => String(x.id) === String(vid));
    if (!v) return;
    const key = String(vid);
    if (state.volcanoMarkers[key]) return;

    const levelClass = 'volcano-' + v.level;
    const size = v.level === 'high' ? 14 : 11;
    const el = document.createElement('div');
    el.innerHTML = `<div class="volcano-marker ${levelClass}"></div>`;
    el.style.cursor = 'pointer';

    const m = new maplibregl.Marker({ element: el, anchor: 'center' })
        .setLngLat([v.lon, v.lat])
        .addTo(map);

    const tooltip = `<b>${v.name}</b><br>${v.hotspot_count} hotspots (48h)<br>FRP: ${v.max_frp} MW | ${v.max_brightness}K<br>Level: <b>${v.level.toUpperCase()}</b>`;
    setTooltip(m, tooltip);
    state.volcanoMarkers[key] = m;
}

function _hideVolcanoMarker(vid) {
    const key = String(vid);
    if (state.volcanoMarkers[key]) {
        state.volcanoMarkers[key].remove();
        delete state.volcanoMarkers[key];
    }
}

function updateVolcanoMarkers() {}

function updateConditions(signals) {
    const el = document.getElementById('conditions-panel');
    if (!el) return;

    const items = [
        { key: 'noaa_swpc__kp_index', label: 'Kp Index', unit: '',
          status: v => v >= 5 ? ['Storm', 'cond-storm'] : v >= 4 ? ['Active', 'cond-active'] : ['', ''] },
        { key: 'noaa_swpc__dst_index', label: 'Dst Index', unit: 'nT',
          status: v => v <= -50 ? ['Storm', 'cond-storm'] : v <= -20 ? ['Active', 'cond-active'] : ['', ''] },
    ];

    const swKey = Object.keys(signals).find(k => k.includes('solar_wind') && k.includes('speed'));
    if (swKey && signals[swKey]) {
        items.push({
            key: swKey, label: 'Solar Wind', unit: 'km/s',
            status: v => v >= 600 ? ['High', 'cond-storm'] : v >= 450 ? ['Elevated', 'cond-active'] : ['', ''],
        });
    }

    let html = '';
    for (const item of items) {
        const sig = signals[item.key];
        if (!sig) continue;
        const val = typeof sig.value === 'number' ? (item.unit === 'km/s' ? sig.value.toFixed(0) : sig.value.toFixed(1)) : sig.value;
        const [statusText, statusCls] = item.status(sig.value);
        const statusHtml = statusText ? `<span class="condition-status ${statusCls}">${statusText}</span>` : '';
        const canvasId = 'cond-spark-' + item.key.replace(/[^a-z0-9]/gi, '-');
        html += `<div class="condition-item">
            <div class="condition-header">
                <span class="condition-label">${item.label}</span>
                <span class="condition-value">${val}</span>
                <span class="condition-unit">${item.unit}</span>
                ${statusHtml}
            </div>
            <canvas class="condition-spark" id="${canvasId}"></canvas>
        </div>`;
    }

    el.innerHTML = html;

    requestAnimationFrame(() => {
        for (const item of items) {
            const canvasId = 'cond-spark-' + item.key.replace(/[^a-z0-9]/gi, '-');
            const canvas = document.getElementById(canvasId);
            const history = state.signalHistory[item.key];
            if (canvas && history && history.length > 1) drawSparkline(canvas, history, 'rgba(255,255,255,0.5)');
        }
    });
}

function updateSeismicitySummary(data) {
    const el = document.getElementById('seismicity-summary');
    if (!el || !data) return;

    el.innerHTML = `
        <canvas class="seis-chart" id="seis-hourly-chart"></canvas>
        <div class="seis-stats">
            <div class="seis-stat">
                <span class="seis-stat-value">${data.total_24h}</span>
                <span class="seis-stat-label">Total</span>
            </div>
            <div class="seis-stat">
                <span class="seis-stat-value">${data.m45_24h}</span>
                <span class="seis-stat-label">M4.5+</span>
            </div>
            <div class="seis-stat">
                <span class="seis-stat-value">${data.m50_24h}</span>
                <span class="seis-stat-label">M5+</span>
            </div>
            <div class="seis-stat">
                <span class="seis-stat-value">${data.max_mag_24h}</span>
                <span class="seis-stat-label">Peak</span>
            </div>
        </div>
    `;

    if (data.hourly_counts && data.hourly_counts.length > 1) {
        requestAnimationFrame(() => {
            const canvas = document.getElementById('seis-hourly-chart');
            if (!canvas) return;
            const rect = canvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            const w = rect.width, h = rect.height;
            if (w === 0) return;
            canvas.width = w * dpr;
            canvas.height = h * dpr;
            const ctx = canvas.getContext('2d');
            ctx.scale(dpr, dpr);

            const counts = data.hourly_counts;
            const maxVal = Math.max(...counts, 1);
            const barW = (w - 2) / counts.length;
            const gap = Math.max(1, barW * 0.15);

            for (let i = 0; i < counts.length; i++) {
                const barH = Math.max(1, (counts[i] / maxVal) * (h - 2));
                const x = 1 + i * barW + gap / 2;
                ctx.fillStyle = 'rgba(255,255,255,0.25)';
                ctx.fillRect(x, h - barH, barW - gap, barH);
            }
        });
    }
}

// ── Swarm Watch panel ───────────────────────────────────
const SW_ZONE_LABELS = {
    indonesia: 'Indonesia', japan_kurils: 'Japan & Kurils',
    south_america: 'South America', mexico_ca: 'Mexico & Central America',
    himalaya: 'Himalaya', alaska: 'Alaska', california: 'California',
    philippines: 'Philippines', mediterranean: 'Mediterranean',
    caribbean: 'Caribbean', new_zealand: 'New Zealand',
    png_solomon: 'PNG & Solomon Islands', kamchatka: 'Kamchatka',
};

const SW_ALERT_COLORS = {
    WATCH: 'rgba(230,160,50,0.95)',
    ADVISORY: 'rgba(220,200,60,0.9)',
    NORMAL: 'rgba(160,160,160,0.6)',
};

const ZONE_BOUNDARIES = {
    alaska: [[-172,50],[-140,50],[-140,62],[-155,65],[-172,62],[-172,50]],
    philippines: [[117,5],[128,5],[128,25],[117,25],[117,5]],
    himalaya: [[70,25],[95,25],[95,38],[70,38],[70,25]],
    chile_peru: [[-80,-45],[-65,-45],[-65,-5],[-80,-5],[-80,-45]],
    indonesia: [[95,-12],[140,-12],[140,6],[95,6],[95,-12]],
    japan: [[128,28],[148,28],[148,46],[128,46],[128,28]],
    caribbean: [[-85,10],[-58,10],[-58,22],[-85,22],[-85,10]],
    mediterranean: [[0,32],[42,32],[42,44],[0,44],[0,32]],
    mexico_ca: [[-110,8],[-82,8],[-82,22],[-110,22],[-110,8]],
    norcal: [[-130,38],[-118,38],[-118,50],[-130,50],[-130,38]],
    socal: [[-122,31],[-114,31],[-114,37],[-122,37],[-122,31]],
    iceland: [[-26,62],[-12,62],[-12,67],[-26,67],[-26,62]],
};

const ZONE_LEVEL_COLORS = {
    ALERT: { fill: 'rgba(220,70,50,0.10)', label: 'rgba(220,70,50,0.9)' },
    ELEVATED: { fill: 'rgba(230,160,50,0.08)', label: 'rgba(230,160,50,0.9)' },
    WATCH: { fill: 'rgba(255,255,255,0.04)', label: 'rgba(255,255,255,0.6)' },
};

let _latestThreats = null;
let _swarmWatchData = null;

function updateSwarmWatch(data) {
    _swarmWatchData = data;
    const el = document.getElementById('sw-list');
    if (!el) return;

    const watch = data.watch || [];
    const elevated = watch.filter(w => w.alert_level !== 'NORMAL' && w.model_skill === 'validated');
    const experimental = watch.filter(w => w.model_skill === 'experimental' && w.n_recent_quakes > 0);
    const normalCount = watch.filter(w => w.alert_level === 'NORMAL' && w.model_skill === 'validated').length;

    let html = '';

    if (elevated.length === 0 && experimental.length === 0) {
        html = '<div class="panel-muted">Nothing building — no elevated swarms</div>';
        if (normalCount > 0) {
            html += `<div class="sw-baseline-count">${normalCount} validated swarms at baseline</div>`;
        }
    } else {
        const sortedElevated = [...elevated].sort((a, b) => b.escalation_prob_72h - a.escalation_prob_72h);
        for (const w of sortedElevated) {
            const pct = (w.escalation_prob_72h * 100).toFixed(1);
            const alertClass = w.alert_level.toLowerCase();
            const label = SW_ZONE_LABELS[w.zone] || w.zone;
            const nq = w.n_recent_quakes || 0;
            const dir = w.direction === 'RISING' ? ' ↑' : w.direction === 'FALLING' ? ' ↓' : '';

            html += `<div class="sw-row" data-cell="${w.cell}">
                <div class="sw-header">
                    <span class="sw-zone">${label}</span>
                    <span class="sw-alert sw-alert-${alertClass}">${w.alert_level}</span>
                </div>
                <div class="sw-detail">${pct}% escalation in 72h${dir} · ${nq} quakes</div>
            </div>`;
        }

        if (experimental.length > 0) {
            const topExp = [...experimental].sort((a, b) => b.n_recent_quakes - a.n_recent_quakes).slice(0, 5);
            html += `<div class="sw-baseline-count">${experimental.length} observed swarms (experimental zones)</div>`;
            for (const w of topExp) {
                const label = SW_ZONE_LABELS[w.zone] || w.zone;
                const nq = w.n_recent_quakes || 0;
                const dir = w.direction === 'RISING' ? ' ↑' : w.direction === 'FALLING' ? ' ↓' : '';
                html += `<div class="sw-row sw-row-experimental" data-cell="${w.cell}">
                    <div class="sw-header">
                        <span class="sw-zone">${label}</span>
                        <span class="sw-alert sw-alert-normal">OBSERVED</span>
                    </div>
                    <div class="sw-detail">${nq} quakes${dir} · activity detected</div>
                </div>`;
            }
        }

        if (normalCount > 0) {
            html += `<div class="sw-baseline-count">${normalCount} more at baseline</div>`;
        }
    }

    if (data.generated) {
        const gen = new Date(data.generated.replace(' ', 'T'));
        const timeStr = gen.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit', timeZone: 'UTC' });
        html += `<div class="sw-footer">${data.n_active_swarms || 0} swarms · ${timeStr} UTC</div>`;
    }

    el.innerHTML = html;

    el.querySelectorAll('.sw-row[data-cell]').forEach(row => {
        row.addEventListener('click', () => openSwarmDetail(row.dataset.cell));
    });

    updateSwarmPolygons(data);
}

// ── Chart helpers ────────────────────────────────────────
function drawAreaChart(canvas, data, color, opts = {}) {
    if (!canvas || !data || data.length < 2) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width, h = rect.height;
    if (w === 0 || h === 0) return;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const values = data.map(d => typeof d === 'number' ? d : d.v);
    const min = opts.min != null ? opts.min : Math.min(...values);
    const max = opts.max != null ? opts.max : Math.max(...values);
    const range = max - min || 1;
    const pad = 4;

    if (opts.threshold != null) {
        const ty = h - pad - ((opts.threshold - min) / range) * (h - pad * 2);
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, ty);
        ctx.lineTo(w, ty);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    for (let i = 0; i < values.length; i++) {
        const x = (i / (values.length - 1)) * w;
        const y = h - pad - ((values[i] - min) / range) * (h - pad * 2);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, hexToRgba(color, 0.12));
    grad.addColorStop(1, hexToRgba(color, 0));
    ctx.fillStyle = grad;
    ctx.fill();

    if (opts.labels) {
        ctx.font = '9px Inter, sans-serif';
        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.textAlign = 'left';
        ctx.fillText(opts.labels[0], 2, h - 2);
        ctx.textAlign = 'right';
        ctx.fillText(opts.labels[1], w - 2, h - 2);
    }
}

function drawBarChart(canvas, data, colorFn) {
    if (!canvas || !data || data.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width, h = rect.height;
    if (w === 0 || h === 0) return;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const max = Math.max(...data.map(d => d.count)) || 1;
    const barW = Math.max(2, (w / data.length) - 1);
    const gap = 1;
    for (let i = 0; i < data.length; i++) {
        const barH = Math.max(1, (data[i].count / max) * (h - 4));
        const x = i * (barW + gap);
        const y = h - barH;
        ctx.fillStyle = colorFn ? colorFn(data[i]) : 'rgba(120,160,200,0.3)';
        ctx.beginPath();
        ctx.roundRect(x, y, barW, barH, 1);
        ctx.fill();
    }
}

function drawGaugeArc(canvas, value, max, color) {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const size = rect.width;
    if (size === 0) return;
    canvas.width = size * dpr;
    canvas.height = (size / 2 + 8) * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const cx = size / 2, cy = size / 2;
    const r = (size - 12) / 2;
    const startAngle = Math.PI;
    const endAngle = 2 * Math.PI;
    const pct = Math.min(1, Math.max(0, value / max));

    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, endAngle);
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, startAngle + pct * Math.PI);
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.stroke();

    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, startAngle + pct * Math.PI);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.shadowBlur = 0;
}

// ── Data Analysis Workbench ─────────────────────────────
const WORKBENCH_FEATURES = [
    // Geomagnetic
    { id: 'kp', source: 'noaa_swpc', metric: 'kp_index', label: 'Kp Index', unit: '', category: 'Geomagnetic', thresholds: [{ v: 5, label: 'Storm' }] },
    { id: 'dst', source: 'noaa_swpc', metric: 'dst_index', label: 'Dst Index', unit: 'nT', category: 'Geomagnetic', thresholds: [{ v: -50, label: 'Storm' }] },
    { id: 'goes_bt', source: 'goes_magnetometer', metric: 'goes_mag_total', label: 'GOES |B|', unit: 'nT', category: 'Geomagnetic', thresholds: [] },
    { id: 'goes_hp', source: 'goes_magnetometer', metric: 'goes_mag_hp', label: 'GOES Hp', unit: 'nT', category: 'Geomagnetic', thresholds: [] },
    { id: 'goes_he', source: 'goes_magnetometer', metric: 'goes_mag_he', label: 'GOES He', unit: 'nT', category: 'Geomagnetic', thresholds: [] },
    { id: 'goes_hn', source: 'goes_magnetometer', metric: 'goes_mag_hn', label: 'GOES Hn', unit: 'nT', category: 'Geomagnetic', thresholds: [] },
    { id: 'ground_z', source: 'intermagnet', metric: 'imag_z', label: 'Ground Mag Z', unit: 'nT', category: 'Geomagnetic', thresholds: [] },
    { id: 'ground_x', source: 'intermagnet', metric: 'imag_x', label: 'Ground Mag X', unit: 'nT', category: 'Geomagnetic', thresholds: [] },
    { id: 'ground_y', source: 'intermagnet', metric: 'imag_y', label: 'Ground Mag Y', unit: 'nT', category: 'Geomagnetic', thresholds: [] },
    { id: 'ground_f', source: 'intermagnet', metric: 'imag_f', label: 'Ground Mag F', unit: 'nT', category: 'Geomagnetic', thresholds: [] },
    // Solar
    { id: 'sw_speed', source: 'noaa_swpc', metric: 'solar_wind_speed', label: 'Solar Wind Speed', unit: 'km/s', category: 'Solar', thresholds: [{ v: 500, label: 'Elevated' }] },
    { id: 'sw_density', source: 'noaa_swpc', metric: 'solar_wind_density', label: 'Solar Wind Density', unit: 'p/cm3', category: 'Solar', thresholds: [] },
    { id: 'imf_bt', source: 'noaa_swpc', metric: 'imf_bt', label: 'IMF Bt', unit: 'nT', category: 'Solar', thresholds: [] },
    { id: 'imf_bz', source: 'noaa_swpc', metric: 'imf_bz', label: 'IMF Bz', unit: 'nT', category: 'Solar', thresholds: [{ v: -5, label: 'Southward' }] },
    { id: 'ief', source: 'derived_ief', metric: 'ief', label: 'IEF', unit: 'mV/m', category: 'Solar', thresholds: [] },
    { id: 'proton', source: 'noaa_goes', metric: 'proton_flux', label: 'Proton Flux', unit: 'pfu', category: 'Solar', thresholds: [] },
    { id: 'electron', source: 'noaa_goes', metric: 'electron_flux', label: 'Electron Flux', unit: '', category: 'Solar', thresholds: [] },
    { id: 'neutron', source: 'nmdb_cosmic', metric: 'neutron_count', label: 'Neutron Count', unit: '', category: 'Solar', thresholds: [] },
    { id: 'xray', source: 'goes_xray', metric: 'xray_flux', label: 'X-ray Flux', unit: 'W/m2', category: 'Solar', thresholds: [] },
    // Tidal
    { id: 'tidal_pot', source: 'computed_tides', metric: 'tidal_potential', label: 'Tidal Potential', unit: '', category: 'Tidal', thresholds: [] },
    { id: 'tidal_strain', source: 'computed_tides', metric: 'tidal_strain_rate', label: 'Tidal Strain Rate', unit: '', category: 'Tidal', thresholds: [] },
    // Ocean
    { id: 'water_level', source: 'noaa_tides', metric: 'water_level', label: 'Water Level', unit: 'm', category: 'Ocean', thresholds: [] },
    // Other
    { id: 'olr', source: 'noaa_olr', metric: 'olr', label: 'OLR', unit: 'W/m2', category: 'Other', thresholds: [] },
    { id: 'gravity', source: 'grace_gravity', metric: 'gravity_anomaly', label: 'Gravity Anomaly', unit: 'cm', category: 'Other', thresholds: [] },
    { id: 'pressure', source: 'nws_weather', metric: 'barometric_pressure', label: 'Barometric Pressure', unit: 'mb', category: 'Other', thresholds: [] },
];

const SIGNAL_FEATURE_MAP = {
    'dst':        ['dst'],
    'kp':         ['kp'],
    'solar_wind': ['sw_speed', 'sw_density'],
    'solar wind': ['sw_speed', 'sw_density'],
    'geomag':     ['kp', 'dst', 'goes_bt'],
    'magnetic':   ['goes_bt', 'goes_hp', 'ground_z', 'ground_f'],
    'magnetospheric': ['goes_bt', 'goes_hp'],
    'compression': ['goes_bt', 'goes_hp'],
    'tidal':      ['tidal_pot', 'tidal_strain'],
    'gravitational': ['tidal_pot', 'tidal_strain'],
    'imf':        ['imf_bt', 'imf_bz'],
    'proton':     ['proton'],
    'electron':   ['electron'],
    'x-ray':      ['xray'],
    'xray':       ['xray'],
    'neutron':    ['neutron'],
    'olr':        ['olr'],
    'water':      ['water_level'],
    'pressure':   ['pressure'],
    'gravity':    ['gravity'],
    'ief':        ['ief'],
};

const workbenchState = {
    active: new Set(),
    timeHours: 6,
    zoneName: '',
    zoneData: null,
    chartData: {},
    annotations: {},
};

function saveWbPrefs() {
    try {
        localStorage.setItem('qw_wb_prefs', JSON.stringify({
            features: [...workbenchState.active],
            timeHours: workbenchState.timeHours,
        }));
    } catch (e) {}
}

function loadWbPrefs() {
    try {
        const raw = localStorage.getItem('qw_wb_prefs');
        if (raw) return JSON.parse(raw);
    } catch (e) {}
    return null;
}

function getFeatureById(id) {
    return WORKBENCH_FEATURES.find(f => f.id === id);
}

function buildWorkbenchSidebar() {
    const sidebar = document.getElementById('wb-sidebar');
    const categories = {};
    for (const f of WORKBENCH_FEATURES) {
        if (!categories[f.category]) categories[f.category] = [];
        categories[f.category].push(f);
    }
    let html = '';
    for (const [cat, feats] of Object.entries(categories)) {
        html += `<div class="wb-cat-header" data-cat="${cat}">${cat}<span class="wb-cat-chevron">&#9660;</span></div>`;
        html += `<div class="wb-cat-items" data-cat="${cat}">`;
        for (const f of feats) {
            const activeCls = workbenchState.active.has(f.id) ? ' active' : '';
            html += `<div class="wb-feat-row${activeCls}" data-fid="${f.id}"><div class="wb-feat-check"></div><span class="wb-feat-label">${f.label}</span></div>`;
        }
        html += '</div>';
    }
    sidebar.innerHTML = html;

    sidebar.querySelectorAll('.wb-cat-header').forEach(hdr => {
        hdr.addEventListener('click', () => {
            hdr.classList.toggle('collapsed');
            const items = sidebar.querySelector(`.wb-cat-items[data-cat="${hdr.dataset.cat}"]`);
            if (items) items.classList.toggle('collapsed');
        });
    });

    sidebar.querySelectorAll('.wb-feat-row').forEach(row => {
        row.addEventListener('click', () => {
            const fid = row.dataset.fid;
            if (workbenchState.active.has(fid)) {
                workbenchState.active.delete(fid);
                row.classList.remove('active');
                removeWorkbenchChart(fid);
            } else {
                workbenchState.active.add(fid);
                row.classList.add('active');
                addWorkbenchChart(fid);
            }
            saveWbPrefs();
        });
    });
}

function ensureChartGrid() {
    const main = document.getElementById('wb-main');
    let grid = main.querySelector('.wb-chart-grid');
    if (!grid) {
        main.innerHTML = '';
        grid = document.createElement('div');
        grid.className = 'wb-chart-grid';
        main.appendChild(grid);
    }
    return grid;
}

async function addWorkbenchChart(fid) {
    const feat = getFeatureById(fid);
    if (!feat) return;
    const grid = ensureChartGrid();

    // Remove empty state
    const empty = document.getElementById('wb-main').querySelector('.wb-empty');
    if (empty) empty.remove();

    // Create card
    const card = document.createElement('div');
    card.className = 'wb-chart-card';
    card.id = 'wb-card-' + fid;

    // Build annotation badges
    let annoBadges = '';
    const annos = workbenchState.annotations[fid];
    if (annos && annos.length > 0) {
        annoBadges = '<div class="wb-annotation">' +
            annos.map(a => {
                const cls = a.severity > 0.6 ? '' : a.severity > 0.3 ? ' elevated' : ' watch';
                return `<span class="wb-anno-badge${cls}">${a.label}</span>`;
            }).join('') + '</div>';
    }

    card.innerHTML = `
        <div class="wb-chart-header">
            <span class="wb-chart-title">${feat.label}<span class="wb-chart-unit">${feat.unit}</span><span class="wb-chart-val" id="wb-val-${fid}"></span></span>
            <div class="wb-chart-remove" data-fid="${fid}">&#10005;</div>
        </div>
        ${annoBadges}
        <div class="wb-chart-loading">Loading...</div>
    `;
    grid.appendChild(card);

    card.querySelector('.wb-chart-remove').addEventListener('click', () => {
        workbenchState.active.delete(fid);
        removeWorkbenchChart(fid);
        const row = document.querySelector(`.wb-feat-row[data-fid="${fid}"]`);
        if (row) row.classList.remove('active');
        saveWbPrefs();
    });

    // Fetch data
    await fetchAndRenderChart(fid, card);
}

async function fetchAndRenderChart(fid, card) {
    const feat = getFeatureById(fid);
    if (!feat || !card) return;

    const now = new Date();
    const hours = workbenchState.timeHours;
    const start = new Date(now.getTime() - hours * 3600000);
    const bucketMap = { 6: 1, 24: 3, 48: 5, 168: 15, 720: 60 };
    const bucket = bucketMap[hours] || Math.max(1, Math.round(hours * 60 / 700));
    const url = `/api/samples?source=${encodeURIComponent(feat.source)}&metric=${encodeURIComponent(feat.metric)}&start=${start.toISOString()}&bucket=${bucket}`;

    try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error('API error');
        const samples = await resp.json();

        workbenchState.chartData[fid] = samples;

        if (!samples || samples.length === 0) {
            const loading = card.querySelector('.wb-chart-loading');
            if (loading) loading.textContent = 'No data for this range';
            return;
        }

        const latestVal = samples[samples.length - 1].value;
        const valEl = card.querySelector('.wb-chart-val');
        if (valEl) valEl.textContent = formatAxisVal(latestVal);

        // Replace loading with canvas + labels
        const loading = card.querySelector('.wb-chart-loading');
        if (loading) loading.remove();

        const canvas = document.createElement('canvas');
        canvas.className = 'wb-chart-canvas';
        canvas.id = 'wb-canvas-' + fid;

        const labelRow = document.createElement('div');
        labelRow.className = 'wb-chart-labels';

        // Time labels
        const oldest = new Date(samples[0].timestamp);
        const newest = new Date(samples[samples.length - 1].timestamp);
        labelRow.innerHTML = `<span class="wb-chart-xlabel">${formatChartTime(oldest)}</span><span class="wb-chart-xlabel">${formatChartTime(newest)}</span>`;

        // Remove any existing canvas/labels (for re-render)
        const oldCanvas = card.querySelector('.wb-chart-canvas');
        const oldLabels = card.querySelector('.wb-chart-labels');
        const oldYmin = card.querySelector('.wb-chart-ylabel.bottom');
        const oldYmax = card.querySelector('.wb-chart-ylabel.top');
        if (oldCanvas) oldCanvas.remove();
        if (oldLabels) oldLabels.remove();
        if (oldYmin) oldYmin.remove();
        if (oldYmax) oldYmax.remove();

        card.appendChild(canvas);
        card.appendChild(labelRow);

        // Y-axis labels
        const values = samples.map(s => s.value);
        const minV = Math.min(...values);
        const maxV = Math.max(...values);
        const yMaxEl = document.createElement('span');
        yMaxEl.className = 'wb-chart-ylabel top';
        yMaxEl.textContent = formatAxisVal(maxV);
        const yMinEl = document.createElement('span');
        yMinEl.className = 'wb-chart-ylabel bottom';
        yMinEl.textContent = formatAxisVal(minV);
        card.appendChild(yMaxEl);
        card.appendChild(yMinEl);

        requestAnimationFrame(() => {
            drawWorkbenchChart(canvas, samples, feat);
        });
    } catch (e) {
        const loading = card.querySelector('.wb-chart-loading');
        if (loading) {
            loading.textContent = 'Failed to load';
            loading.style.color = 'rgba(220,70,50,0.4)';
        }
    }
}

function formatAxisVal(v) {
    if (v === 0) return '0';
    if (Math.abs(v) >= 10000) return v.toExponential(1);
    if (Math.abs(v) >= 100) return v.toFixed(0);
    if (Math.abs(v) >= 1) return v.toFixed(1);
    if (Math.abs(v) >= 0.01) return v.toFixed(3);
    return v.toExponential(1);
}

function formatChartTime(date) {
    const h = String(date.getUTCHours()).padStart(2, '0');
    const m = String(date.getUTCMinutes()).padStart(2, '0');
    const d = String(date.getUTCDate()).padStart(2, '0');
    const mo = String(date.getUTCMonth() + 1).padStart(2, '0');
    if (workbenchState.timeHours <= 48) return `${h}:${m}`;
    return `${mo}/${d} ${h}:${m}`;
}

function drawWorkbenchChart(canvas, samples, feat) {
    if (!canvas || !samples || samples.length < 2) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width, h = rect.height;
    if (w === 0 || h === 0) return;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const values = samples.map(s => s.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const padX = 4, padY = 6;

    // Grid lines (4 horizontal)
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 4; i++) {
        const gy = padY + ((h - padY * 2) * i) / 3;
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(w, gy);
        ctx.stroke();
    }

    // Threshold lines
    if (feat.thresholds) {
        for (const t of feat.thresholds) {
            const ty = h - padY - ((t.v - min) / range) * (h - padY * 2);
            if (ty > padY && ty < h - padY) {
                ctx.setLineDash([3, 3]);
                ctx.strokeStyle = 'rgba(220,70,50,0.15)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(0, ty);
                ctx.lineTo(w, ty);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        }
    }

    // Line
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (let i = 0; i < values.length; i++) {
        const x = padX + (i / (values.length - 1)) * (w - padX * 2);
        const y = h - padY - ((values[i] - min) / range) * (h - padY * 2);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Gradient fill
    const lastX = padX + ((values.length - 1) / (values.length - 1)) * (w - padX * 2);
    ctx.lineTo(lastX, h);
    ctx.lineTo(padX, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(255,255,255,0.08)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grad;
    ctx.fill();
}

function removeWorkbenchChart(fid) {
    const card = document.getElementById('wb-card-' + fid);
    if (card) card.remove();
    delete workbenchState.chartData[fid];

    // If no charts remain, show empty state
    if (workbenchState.active.size === 0) {
        const main = document.getElementById('wb-main');
        main.innerHTML = '<div class="wb-empty">Select features from the sidebar to begin analysis</div>';
    }
}

async function refreshAllWorkbenchCharts() {
    for (const fid of workbenchState.active) {
        const card = document.getElementById('wb-card-' + fid);
        if (card) {
            // Show loading on existing chart
            const canvas = card.querySelector('.wb-chart-canvas');
            const labels = card.querySelector('.wb-chart-labels');
            const yMin = card.querySelector('.wb-chart-ylabel.bottom');
            const yMax = card.querySelector('.wb-chart-ylabel.top');
            if (canvas) canvas.remove();
            if (labels) labels.remove();
            if (yMin) yMin.remove();
            if (yMax) yMax.remove();
            const existing = card.querySelector('.wb-chart-loading');
            if (!existing) {
                const loading = document.createElement('div');
                loading.className = 'wb-chart-loading';
                loading.textContent = 'Loading...';
                card.appendChild(loading);
            }
            fetchAndRenderChart(fid, card);
        }
    }
}

function mapSignalsToFeatures(signals) {
    const annotations = {};
    if (!signals || !Array.isArray(signals)) return annotations;

    for (const sig of signals) {
        const sigText = (sig.signal || '').toLowerCase();
        for (const [keyword, featureIds] of Object.entries(SIGNAL_FEATURE_MAP)) {
            if (sigText.includes(keyword)) {
                for (const fid of featureIds) {
                    if (!annotations[fid]) annotations[fid] = [];
                    const alreadyHas = annotations[fid].some(a => a.label === sig.signal);
                    if (!alreadyHas) {
                        annotations[fid].push({
                            label: sig.signal.length > 30 ? sig.signal.slice(0, 28) + '..' : sig.signal,
                            severity: sig.severity || 0.3,
                        });
                    }
                }
            }
        }
    }
    return annotations;
}

function getDefaultFeatures(zoneData) {
    // Default features based on zone signals
    const defaults = new Set(['kp', 'dst', 'sw_speed']);
    if (!zoneData || !zoneData.signals) return defaults;

    for (const sig of zoneData.signals) {
        const sigText = (sig.signal || '').toLowerCase();
        for (const [keyword, featureIds] of Object.entries(SIGNAL_FEATURE_MAP)) {
            if (sigText.includes(keyword)) {
                for (const fid of featureIds) defaults.add(fid);
            }
        }
    }
    // Cap at 6 defaults max
    const arr = [...defaults];
    return new Set(arr.slice(0, 6));
}

async function openWorkbench(zoneName, zoneData) {
    const overlay = document.getElementById('wb-overlay');
    const nameEl = document.getElementById('wb-zone-name');

    workbenchState.zoneName = zoneName || 'Zone Analysis';
    workbenchState.zoneData = zoneData || null;
    workbenchState.chartData = {};

    nameEl.textContent = workbenchState.zoneName;

    workbenchState.annotations = zoneData ? mapSignalsToFeatures(zoneData.signals) : {};

    const saved = loadWbPrefs();
    if (saved && saved.features && saved.features.length > 0) {
        workbenchState.active = new Set(saved.features);
        workbenchState.timeHours = saved.timeHours || 6;
    } else {
        workbenchState.active = getDefaultFeatures(zoneData);
        workbenchState.timeHours = 6;
    }

    buildWorkbenchSidebar();

    document.querySelectorAll('.wb-time-btn').forEach(b => {
        b.classList.toggle('active', parseInt(b.dataset.hours) === workbenchState.timeHours);
    });

    // Clear main and populate charts
    const main = document.getElementById('wb-main');
    main.innerHTML = '';

    overlay.classList.add('visible');

    // Add charts for default features
    for (const fid of workbenchState.active) {
        await addWorkbenchChart(fid);
    }

    // Add nearest station seismograph(s)
    stopAllEmbeddedSeismos();
    const lat = zoneData?.center?.[0];
    const lon = zoneData?.center?.[1];
    if (lat != null && lon != null) {
        const nearest = findNearestStations(lat, lon, 2);
        if (nearest.length > 0) {
            let seismoHtml = '<div class="wb-seismo-section"><div class="wb-seismo-title">Nearest Stations — Live Seismograph</div>';
            for (let i = 0; i < nearest.length; i++) {
                seismoHtml += `<div class="embedded-seismo-wrap" id="wb-seismo-${i}"></div>`;
            }
            seismoHtml += '</div>';
            main.insertAdjacentHTML('beforeend', seismoHtml);
            requestAnimationFrame(() => {
                for (let i = 0; i < nearest.length; i++) {
                    startEmbeddedSeismo(`wb-seismo-${i}`, nearest[i].station, nearest[i].name, '6h');
                }
            });
        }
    }
}

function closeWorkbench() {
    stopAllEmbeddedSeismos();
    const overlay = document.getElementById('wb-overlay');
    overlay.classList.remove('visible');
    workbenchState.active.clear();
    workbenchState.chartData = {};
    workbenchState.annotations = {};
}

function renderMarkdown(md) {
    let html = md
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/\*\*(.+?)\*\*/g, (_, t) => {
            if (/CRITICAL|IMMEDIATE/i.test(t)) return '<strong class="alert-text">' + t + '</strong>';
            if (/ELEVATED|HIGH/i.test(t)) return '<strong class="elevated-text">' + t + '</strong>';
            return '<strong>' + t + '</strong>';
        })
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/^\| *(.+)$/gm, (_, row) => {
            const cells = row.split('|').map(c => c.trim()).filter(Boolean);
            return '<tr>' + cells.map(c => {
                if (/CRITICAL|IMMEDIATE/i.test(c)) return '<td class="alert">' + c + '</td>';
                if (/ELEVATED|HIGH/i.test(c)) return '<td class="elevated">' + c + '</td>';
                return '<td>' + c + '</td>';
            }).join('') + '</tr>';
        })
        .replace(/^- (.+)$/gm, '<li>$1</li>');

    html = html.replace(/(<tr>.*<\/tr>\n?)+/gs, match => {
        const rows = match.trim().split('\n').filter(r => r.startsWith('<tr>'));
        if (rows.length < 2) return match;
        const sepRow = rows[1];
        if (sepRow.includes('---')) {
            const header = rows[0].replace(/<td>/g, '<th>').replace(/<\/td>/g, '</th>');
            return '<table>' + header + rows.slice(2).join('\n') + '</table>';
        }
        return '<table>' + rows.join('\n') + '</table>';
    });

    html = html.replace(/(<li>.*<\/li>\n?)+/gs, match => '<ul>' + match + '</ul>');

    html = html.split('\n\n').map(block => {
        block = block.trim();
        if (!block) return '';
        if (block.startsWith('<h3>') || block.startsWith('<ul>') || block.startsWith('<ol>') || block.startsWith('<table>')) return block;
        return '<p>' + block + '</p>';
    }).join('\n');

    return html;
}


// ── Colors & helpers ─────────────────────────────────────
function magColor(mag) {
    if (mag >= 7.0) return 'rgba(220,80,60,0.9)';
    if (mag >= 6.0) return 'rgba(210,100,60,0.8)';
    if (mag >= 5.0) return 'rgba(200,140,60,0.7)';
    if (mag >= 4.0) return 'rgba(180,160,80,0.6)';
    if (mag >= 3.0) return 'rgba(130,170,200,0.5)';
    return 'rgba(100,150,200,0.4)';
}

function magSize(mag) {
    if (mag >= 7.0) return 14;
    if (mag >= 6.0) return 11;
    if (mag >= 5.0) return 8;
    if (mag >= 4.0) return 6;
    if (mag >= 3.0) return 4;
    return 3;
}

function buildEqMarkerHTML(mag) {
    const color = magColor(mag);
    const core = magSize(mag);
    const orbit1 = core * 3;
    const orbit2 = core * 2.2;
    const glow = mag >= 7.0 ? 10 : mag >= 6.0 ? 7 : mag >= 5.0 ? 4 : mag >= 4.0 ? 2 : 0;
    const dur1 = mag >= 7.0 ? 2.5 : mag >= 6.0 ? 3.5 : mag >= 5.0 ? 5 : 7;
    const dur2 = mag >= 7.0 ? 1.8 : mag >= 6.0 ? 2.5 : mag >= 5.0 ? 3.5 : 5;
    const orbitW = mag >= 6.0 ? 1.5 : 1;
    const o2opacity = mag >= 5.0 ? 0.5 : 0.3;
    const hasOrbits = mag >= 4.0;

    const orbits = hasOrbits
        ? `<div class="eq-orbit eq-orbit-1"></div><div class="eq-orbit eq-orbit-2"></div>`
        : '';

    return `<div class="eq-marker" style="--c:${color};--s:${core}px;--o:${orbit1}px;--o2:${orbit2}px;--glow:${glow}px;--dur:${dur1}s;--dur2:${dur2}s;--ow:${orbitW}px;--o2a:${o2opacity}">`
        + `<div class="eq-core"></div>${orbits}</div>`;
}

function riskColor(score) {
    if (score >= 0.7)  return 'rgba(220,70,50,0.85)';
    if (score >= 0.45) return 'rgba(210,140,50,0.75)';
    if (score >= 0.2)  return 'rgba(160,170,70,0.65)';
    return 'rgba(80,180,120,0.6)';
}

function timeAgo(ts) {
    const diff = (Date.now() - new Date(ts).getTime()) / 1000;
    if (diff < 0) return 'just now';
    if (diff < 60)    return `${Math.round(diff)}s ago`;
    if (diff < 3600)  return `${Math.round(diff / 60)}m ago`;
    if (diff < 86400) return `${(diff / 3600).toFixed(1)}h ago`;
    return `${Math.round(diff / 86400)}d ago`;
}

function hexToRgba(hex, a) {
    if (hex.startsWith('rgba')) return hex;
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function sparkColor(key) {
    if (key.includes('kp')) return '#5a8abf';
    if (key.includes('dst')) return '#7a9b6a';
    if (key.includes('goes')) return '#b08a5a';
    if (key.includes('mag')) return '#8a7aaa';
    return '#6a8a9a';
}

// ── Sparkline ────────────────────────────────────────────
function drawSparkline(canvas, data, color) {
    if (!canvas || !data || data.length < 2) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width, h = rect.height;
    if (w === 0 || h === 0) return;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    const values = data.map(d => typeof d === 'number' ? d : d.v);
    const min = Math.min(...values), max = Math.max(...values);
    const range = max - min || 1;
    const pad = 2;
    const points = [];
    for (let i = 0; i < values.length; i++) {
        const x = (i / (values.length - 1)) * w;
        const y = h - pad - ((values[i] - min) / range) * (h - pad * 2);
        points.push([x, y]);
    }

    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
        if (i === 0) ctx.moveTo(points[i][0], points[i][1]);
        else ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(255,255,255,0.03)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
        if (i === 0) ctx.moveTo(points[i][0], points[i][1]);
        else ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.55)';
    ctx.lineWidth = 1.2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke();
}

// ── Earthquake markers ───────────────────────────────────
function updateEarthquakeMarkers(quakes) {
    const cutoff = new Date(Date.now() - state.filterHours * 3600000).toISOString();
    for (const eq of quakes) {
        if (eq.lat == null || eq.lon == null) continue;
        if (!state.eqMarkers[eq.id]) {
            const el = document.createElement('div');
            el.innerHTML = buildEqMarkerHTML(eq.magnitude);
            el.style.cursor = 'pointer';
            el.addEventListener('click', (e) => { e.stopPropagation(); showEqDetail(eq); });
            const m = new maplibregl.Marker({ element: el, anchor: 'center' })
                .setLngLat([eq.lon, eq.lat]);
            state.eqMarkers[eq.id] = { marker: m, mag: eq.magnitude, ts: eq.timestamp, visible: false };
        }
        const entry = state.eqMarkers[eq.id];
        const layerOn = !window._legendLayerOn || window._legendLayerOn('earthquakes');
        const visible = layerOn && eq.magnitude >= state.filterMag && eq.timestamp >= cutoff;
        if (visible && !entry.visible) { entry.marker.addTo(map); entry.visible = true; }
        if (!visible && entry.visible) { entry.marker.remove(); entry.visible = false; }
    }
}

function applyMapFilter() {
    const cutoff = new Date(Date.now() - state.filterHours * 3600000).toISOString();
    for (const [id, entry] of Object.entries(state.eqMarkers)) {
        const visible = entry.mag >= state.filterMag && entry.ts >= cutoff;
        if (visible && !entry.visible) { entry.marker.addTo(map); entry.visible = true; }
        if (!visible && entry.visible) { entry.marker.remove(); entry.visible = false; }
    }
}

// ── Seismic station markers ──────────────────────────────
function updateStationMarkers(stations) {
    state.seedlinkStations = stations;
    for (const stn of stations) {
        if (stn.lat == null || stn.lon == null) continue;
        const key = stn.station;
        const connected = stn.connected;
        const triggered = stn.triggered;
        const ratio = stn.sta_lta_ratio || 0;

        const isSelected = state.seismoStation === key;
        const color = triggered ? '#dc4632' : connected ? 'rgba(80,200,120,0.7)' : 'rgba(255,255,255,0.15)';
        const glow = triggered ? 'box-shadow:0 0 8px rgba(220,70,50,0.6)' : isSelected ? 'box-shadow:0 0 10px rgba(80,200,120,0.5)' : '';
        const size = triggered ? 10 : isSelected ? 10 : 7;
        const border = isSelected ? 'border:2px solid rgba(80,200,120,0.9)' : 'border:1px solid rgba(255,255,255,0.3)';

        const html = `<div style="width:${size}px;height:${size}px;background:${color};${border};border-radius:2px;${glow};cursor:pointer" title="${stn.name} (${key}) STA/LTA: ${ratio.toFixed(2)}"></div>`;
        const tipHtml = `<b>${stn.name}</b><br>${key}<br>STA/LTA: ${ratio.toFixed(2)}${triggered ? '<br><span style="color:#dc4632">TRIGGERED</span>' : ''}<div style="margin-top:6px;padding-top:5px;border-top:1px solid rgba(255,255,255,0.06);color:rgba(255,255,255,0.35);font-size:10px">click for seismograph</div>`;

        if (state.seismoStation === key) {
            const ampEl = document.getElementById('seismo-amp');
            const ratEl = document.getElementById('seismo-ratio');
            if (ampEl) ampEl.textContent = `peak: ${stn.peak_amplitude?.toFixed(0) || '—'}`;
            if (ratEl) ratEl.textContent = `STA/LTA: ${ratio.toFixed(2)}`;
        }

        if (state.stationMarkers[key]) {
            state.stationMarkers[key].getElement().innerHTML = html;
            state.stationMarkers[key]._tipHtml = tipHtml;
        } else {
            const el = document.createElement('div');
            el.innerHTML = html;
            el.addEventListener('click', () => openSeismograph(key, stn.name));
            const m = new maplibregl.Marker({ element: el, anchor: 'center' })
                .setLngLat([stn.lon, stn.lat])
                .addTo(map);
            setTooltip(m, tipHtml);
            state.stationMarkers[key] = m;
            if (window._legendLayerOn && !window._legendLayerOn('stations')) {
                el.style.display = 'none';
            }
        }
    }
}

// ── Live Seismograph ────────────────────────────────────
function openSeismograph(stationKey, stationName) {
    closeSeismograph();
    state.seismoStation = stationKey;
    state.seismoBuffer = [];
    state.seismoEnvelope = [];
    state.seismoScale = 'live';
    state.seismoMode = 'raw';
    state.seismoDisplayMode = 'waveform';
    state.seismoResonance = null;

    const panel = document.getElementById('seismograph-panel');
    panel.style.display = '';
    document.getElementById('seismo-station').textContent = stationKey;
    document.getElementById('seismo-name').textContent = stationName || '';
    document.getElementById('seismo-amp').textContent = '';
    document.getElementById('seismo-ratio').textContent = '';

    document.querySelectorAll('.seismo-scale-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.scale === 'live');
    });
    document.querySelectorAll('.seismo-mode-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.mode === 'waveform');
    });
    document.getElementById('seismo-scales').style.display = '';

    const canvas = document.getElementById('seismo-canvas');
    canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
    canvas.height = canvas.offsetHeight * (window.devicePixelRatio || 1);

    connectSeismoWs(stationKey);
}

function closeSeismograph() {
    state.seismoStation = null;
    state.seismoBuffer = [];
    state.seismoResonance = null;
    document.getElementById('seismograph-panel').style.display = 'none';
    if (state.seismoInterval) { clearInterval(state.seismoInterval); state.seismoInterval = null; }
    disconnectSeismoWs();
    if (state.seismoRaf) { cancelAnimationFrame(state.seismoRaf); state.seismoRaf = null; }
}

function setSeismoDisplayMode(mode) {
    state.seismoDisplayMode = mode;
    document.querySelectorAll('.seismo-mode-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.mode === mode);
    });

    if (mode === 'resonance') {
        document.getElementById('seismo-scales').style.display = 'none';
        if (state.seismoRaf) { cancelAnimationFrame(state.seismoRaf); state.seismoRaf = null; }
        if (state.seismoInterval) { clearInterval(state.seismoInterval); state.seismoInterval = null; }
        disconnectSeismoWs();
        document.getElementById('seismo-channel').textContent = '2-50 mHz';
        fetchResonance();
        state.seismoInterval = setInterval(fetchResonance, 30000);
    } else {
        document.getElementById('seismo-scales').style.display = '';
        document.getElementById('seismo-channel').textContent = 'BHZ';
        if (state.seismoInterval) { clearInterval(state.seismoInterval); state.seismoInterval = null; }
        setSeismoScale(state.seismoScale || 'live');
    }
}

async function fetchResonance() {
    if (!state.seismoStation) return;
    try {
        const resp = await fetch(`/api/seedlink/resonance/${state.seismoStation}`);
        const data = await resp.json();
        if (data.samples && data.samples.length > 0) {
            state.seismoBuffer = data.samples;
            state.seismoSampleRate = data.sample_rate || 5;
            state.seismoMode = 'raw';
            state.seismoScale = 'live';
            drawSeismograph();
        }
    } catch (e) { /* retry next interval */ }
}

function connectSeismoWs(stationKey) {
    disconnectSeismoWs();
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${location.host}/ws/seismograph/${stationKey}`);
    state.seismoWs = ws;

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'init') {
            state.seismoBuffer = msg.samples || [];
            state.seismoSampleRate = msg.sr || 5;
            state.seismoMode = 'raw';
        } else if (msg.type === 'update') {
            if (state.seismoScale !== 'live') return;
            const sr = msg.sr || state.seismoSampleRate;
            state.seismoSampleRate = sr;
            state.seismoBuffer.push(...(msg.samples || []));
            const maxSamples = Math.ceil(3600 * sr);
            if (state.seismoBuffer.length > maxSamples) {
                state.seismoBuffer = state.seismoBuffer.slice(-maxSamples);
            }
        }
    };
    ws.onclose = () => { state.seismoWs = null; };

    startLiveDrawLoop();
}

function disconnectSeismoWs() {
    if (state.seismoWs) {
        state.seismoWs.close();
        state.seismoWs = null;
    }
}

function startLiveDrawLoop() {
    if (state.seismoRaf) cancelAnimationFrame(state.seismoRaf);
    function tick() {
        if (!state.seismoStation) return;
        if (state.seismoScale === 'live') {
            drawSeismograph();
        }
        state.seismoRaf = requestAnimationFrame(tick);
    }
    state.seismoRaf = requestAnimationFrame(tick);
}

async function fetchWaveform() {
    if (!state.seismoStation) return;
    try {
        const resp = await fetch(`/api/seedlink/waveform/${state.seismoStation}?scale=${state.seismoScale}`);
        const data = await resp.json();
        state.seismoMode = data.mode || 'raw';
        if (data.mode === 'envelope' && data.envelope) {
            state.seismoEnvelope = data.envelope;
            state.seismoDataSeconds = data.data_seconds || 0;
            state.seismoWindowSeconds = data.window_seconds || 0;
            drawSeismograph();
        } else if (data.samples && data.samples.length > 0) {
            state.seismoBuffer = data.samples;
            state.seismoSampleRate = data.sample_rate || 5;
            drawSeismograph();
        }
    } catch (e) { /* retry next interval */ }
}

function setSeismoScale(scale) {
    state.seismoScale = scale;
    document.querySelectorAll('.seismo-scale-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.scale === scale);
    });

    if (scale === 'live') {
        if (state.seismoInterval) { clearInterval(state.seismoInterval); state.seismoInterval = null; }
        state.seismoMode = 'raw';
        if (!state.seismoWs && state.seismoStation) connectSeismoWs(state.seismoStation);
        startLiveDrawLoop();
    } else {
        if (state.seismoRaf) { cancelAnimationFrame(state.seismoRaf); state.seismoRaf = null; }
        disconnectSeismoWs();
        fetchWaveform();
        if (state.seismoInterval) clearInterval(state.seismoInterval);
        state.seismoInterval = setInterval(fetchWaveform, 10000);
    }
}

function drawSeismograph() {
    const canvas = document.getElementById('seismo-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (state.seismoMode === 'envelope') {
        drawEnvelope(ctx, w, h, dpr);
    } else {
        drawWaveform(ctx, w, h, dpr);
    }
}

const SEISMO_FIXED_RANGE = 15;

function computeRMS(samples) {
    if (!samples || samples.length === 0) return 1;
    let sum = 0;
    for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
    return Math.sqrt(sum / samples.length) || 1;
}

function drawYAxisSNR(ctx, w, h, dpr, maxSNR, pad, midY, padR) {
    const right = padR != null ? padR : 10 * dpr;
    const ticks = [5, 10, 15];
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = `${8 * dpr}px "JetBrains Mono", monospace`;
    ctx.textAlign = 'right';
    for (const val of ticks) {
        if (val > maxSNR) break;
        const frac = val / maxSNR;
        const yUp = midY - (h * 0.42 * frac);
        const yDn = midY + (h * 0.42 * frac);
        ctx.fillText(`${val}×`, pad - 3 * dpr, yUp + 3 * dpr);
        ctx.fillText(`-${val}×`, pad - 3 * dpr, yDn + 3 * dpr);
        const lineAlpha = val === 5 ? 0.08 : 0.05;
        ctx.strokeStyle = `rgba(255,255,255,${lineAlpha})`;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad, yUp); ctx.lineTo(ctx.canvas.width - right, yUp); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad, yDn); ctx.lineTo(ctx.canvas.width - right, yDn); ctx.stroke();
    }
    ctx.fillText('0', pad - 3 * dpr, midY + 3 * dpr);
    ctx.textAlign = 'left';
}

function drawWaveform(ctx, w, h, dpr) {
    const samples = state.seismoBuffer;
    if (!samples || samples.length < 2) return;

    const midY = h / 2;
    const pad = 40 * dpr;
    const padR = 10 * dpr;
    const drawW = w - pad - padR;
    const sr = state.seismoSampleRate || 5;
    const windowSec = state.seismoScale === 'live' ? 3600 : 1800;
    const dataSec = samples.length / sr;
    const dataFrac = Math.min(1, dataSec / windowSec);
    const dataStartX = pad + drawW * (1 - dataFrac);
    const maxSNR = SEISMO_FIXED_RANGE;

    // Normalize to SNR (divide by RMS baseline)
    const rms = computeRMS(samples);
    const norm = samples.map(s => s / rms);

    // Horizontal grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 6; i++) {
        const y = (i / 6) * h;
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - padR, y); ctx.stroke();
    }

    // Time grid
    const gridStep = state.seismoScale === 'live' ? 300 : 60;
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = `${9 * dpr}px "JetBrains Mono", monospace`;
    for (let ago = gridStep; ago < windowSec; ago += gridStep) {
        const x = pad + drawW * (1 - ago / windowSec);
        ctx.strokeStyle = 'rgba(255,255,255,0.04)';
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        const label = ago >= 60 ? `-${Math.round(ago / 60)}m` : `-${ago}s`;
        ctx.fillText(label, x + 3 * dpr, h - 3 * dpr);
    }

    // Zero line
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, midY); ctx.lineTo(w - padR, midY); ctx.stroke();

    const scale = (h * 0.42) / maxSNR;
    drawYAxisSNR(ctx, w, h, dpr, maxSNR, pad, midY, padR);

    // Waveform trace
    ctx.strokeStyle = 'rgba(255,255,255,0.55)';
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    for (let i = 0; i < norm.length; i++) {
        const x = dataStartX + (i / norm.length) * (drawW * dataFrac);
        const y = Math.max(0, Math.min(h, midY - norm[i] * scale));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Highlight spikes > 5× baseline
    let hasSpike = false;
    for (let i = 0; i < norm.length; i++) { if (Math.abs(norm[i]) > 5) { hasSpike = true; break; } }
    if (hasSpike) {
        ctx.strokeStyle = 'rgba(220,100,80,0.3)';
        ctx.lineWidth = 2 * dpr;
        ctx.beginPath();
        let drawing = false;
        for (let i = 0; i < norm.length; i++) {
            const x = dataStartX + (i / norm.length) * (drawW * dataFrac);
            const y = Math.max(0, Math.min(h, midY - norm[i] * scale));
            if (Math.abs(norm[i]) > 5) {
                if (!drawing) { ctx.moveTo(x, y); drawing = true; } else ctx.lineTo(x, y);
            } else { drawing = false; }
        }
        ctx.stroke();
    }

    // NOW label
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = `${9 * dpr}px "JetBrains Mono", monospace`;
    ctx.fillText('NOW', w - padR - 24 * dpr, h - 3 * dpr);
}

function drawEnvelope(ctx, w, h, dpr) {
    const env = state.seismoEnvelope;
    if (!env || env.length < 2) return;

    const midY = h / 2;
    const pad = 40 * dpr;
    const padR = 10 * dpr;
    const drawW = w - pad - padR;
    const windowSec = state.seismoWindowSeconds || ({'6h': 21600, '12h': 43200, '24h': 86400}[state.seismoScale] || 21600);
    const dataSec = state.seismoDataSeconds > 0 ? state.seismoDataSeconds : (env.length * 30);
    const dataFrac = Math.min(1, dataSec / windowSec);
    const dataStartX = pad + drawW * (1 - dataFrac);
    const maxSNR = SEISMO_FIXED_RANGE;

    // Compute RMS baseline from envelope means
    let sumSq = 0;
    for (const e of env) { const avg = (Math.abs(e.mn) + Math.abs(e.mx)) / 2; sumSq += avg * avg; }
    const rms = Math.sqrt(sumSq / env.length) || 1;

    // Horizontal grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 6; i++) {
        const y = (i / 6) * h;
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - padR, y); ctx.stroke();
    }

    // Time grid
    const gridStep = state.seismoScale === '24h' ? 3600 : state.seismoScale === '12h' ? 1800 : 600;
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = `${9 * dpr}px "JetBrains Mono", monospace`;
    for (let ago = gridStep; ago < windowSec; ago += gridStep) {
        const x = pad + drawW * (1 - ago / windowSec);
        ctx.strokeStyle = 'rgba(255,255,255,0.04)';
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        const label = ago >= 3600 ? `-${Math.round(ago / 3600)}h` : `-${Math.round(ago / 60)}m`;
        ctx.fillText(label, x + 3 * dpr, h - 3 * dpr);
    }

    // Zero line
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, midY); ctx.lineTo(w - padR, midY); ctx.stroke();

    const scale = (h * 0.42) / maxSNR;
    drawYAxisSNR(ctx, w, h, dpr, maxSNR, pad, midY, padR);

    // Filled envelope — normalized
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    ctx.beginPath();
    for (let i = 0; i < env.length; i++) {
        const x = dataStartX + (i / env.length) * (drawW * dataFrac);
        const y = Math.max(0, Math.min(h, midY - (env[i].mx / rms) * scale));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    for (let i = env.length - 1; i >= 0; i--) {
        const x = dataStartX + (i / env.length) * (drawW * dataFrac);
        const y = Math.max(0, Math.min(h, midY - (env[i].mn / rms) * scale));
        ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();

    // Max outline
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    for (let i = 0; i < env.length; i++) {
        const x = dataStartX + (i / env.length) * (drawW * dataFrac);
        const y = Math.max(0, Math.min(h, midY - (env[i].mx / rms) * scale));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Min outline
    ctx.beginPath();
    for (let i = 0; i < env.length; i++) {
        const x = dataStartX + (i / env.length) * (drawW * dataFrac);
        const y = Math.max(0, Math.min(h, midY - (env[i].mn / rms) * scale));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // NOW label
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = `${9 * dpr}px "JetBrains Mono", monospace`;
    ctx.fillText('NOW', w - padR - 24 * dpr, h - 3 * dpr);
}

// ── Embedded seismograph (workbench + forecast) ─────────
function findNearestStations(lat, lon, count) {
    if (!state.seedlinkStations || state.seedlinkStations.length === 0) return [];
    const scored = state.seedlinkStations
        .filter(s => s.lat != null && s.lon != null && s.connected)
        .map(s => ({ ...s, dist: Math.sqrt((s.lat - lat) ** 2 + (s.lon - lon) ** 2) }))
        .sort((a, b) => a.dist - b.dist);
    return scored.slice(0, count);
}

function drawGenericWaveform(ctx, w, h, dpr, samples, sampleRate, scale) {
    if (!samples || samples.length < 2) {
        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.font = `${10 * dpr}px "JetBrains Mono", monospace`;
        ctx.textAlign = 'center';
        ctx.fillText('Waiting for data...', w / 2, h / 2);
        ctx.textAlign = 'left';
        return;
    }
    const midY = h / 2;
    const pad = 36 * dpr, padR = 8 * dpr;
    const drawW = w - pad - padR;
    const sr = sampleRate || 5;
    const windowSec = (scale === '1h' || scale === 'live') ? 3600 : 1800;
    const dataSec = samples.length / sr;
    const dataFrac = Math.min(1, dataSec / windowSec);
    const dataStartX = pad + drawW * (1 - dataFrac);
    const maxSNR = SEISMO_FIXED_RANGE;

    const rms = computeRMS(samples);
    const norm = samples.map(s => s / rms);

    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) { const y = (i / 4) * h; ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - padR, y); ctx.stroke(); }

    const gridStep = (scale === '1h' || scale === 'live') ? 300 : 60;
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = `${8 * dpr}px "JetBrains Mono", monospace`;
    for (let ago = gridStep; ago < windowSec; ago += gridStep) {
        const x = pad + drawW * (1 - ago / windowSec);
        ctx.strokeStyle = 'rgba(255,255,255,0.04)';
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        const label = ago >= 60 ? `-${Math.round(ago / 60)}m` : `-${ago}s`;
        ctx.fillText(label, x + 2 * dpr, h - 2 * dpr);
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, midY); ctx.lineTo(w - padR, midY); ctx.stroke();

    const sc = (h * 0.42) / maxSNR;
    drawYAxisSNR(ctx, w, h, dpr, maxSNR, pad, midY, padR);

    ctx.strokeStyle = 'rgba(255,255,255,0.55)';
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    for (let i = 0; i < norm.length; i++) {
        const x = dataStartX + (i / norm.length) * (drawW * dataFrac);
        const y = Math.max(0, Math.min(h, midY - norm[i] * sc));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = `${8 * dpr}px "JetBrains Mono", monospace`;
    ctx.fillText('NOW', w - padR - 22 * dpr, h - 2 * dpr);
}

function drawGenericEnvelope(ctx, w, h, dpr, envelope, scale, apiDataSec, apiWindowSec) {
    if (!envelope || envelope.length < 2) {
        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.font = `${10 * dpr}px "JetBrains Mono", monospace`;
        ctx.textAlign = 'center';
        ctx.fillText('Waiting for data...', w / 2, h / 2);
        ctx.textAlign = 'left';
        return;
    }
    const midY = h / 2;
    const pad = 36 * dpr, padR = 8 * dpr;
    const drawW = w - pad - padR;
    const windowSec = apiWindowSec || ({'6h': 21600, '12h': 43200, '24h': 86400}[scale] || 21600);
    const dataSec = apiDataSec > 0 ? apiDataSec : (envelope.length * 30);
    const dataFrac = Math.min(1, dataSec / windowSec);
    const dataStartX = pad + drawW * (1 - dataFrac);
    const maxSNR = SEISMO_FIXED_RANGE;

    let sumSq = 0;
    for (const e of envelope) { const avg = (Math.abs(e.mn) + Math.abs(e.mx)) / 2; sumSq += avg * avg; }
    const rms = Math.sqrt(sumSq / envelope.length) || 1;

    ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) { const y = (i / 4) * h; ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - padR, y); ctx.stroke(); }

    const gridStep = scale === '24h' ? 3600 : scale === '12h' ? 1800 : 600;
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = `${8 * dpr}px "JetBrains Mono", monospace`;
    for (let ago = gridStep; ago < windowSec; ago += gridStep) {
        const x = pad + drawW * (1 - ago / windowSec);
        ctx.strokeStyle = 'rgba(255,255,255,0.04)';
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        const label = ago >= 3600 ? `-${Math.round(ago / 3600)}h` : `-${Math.round(ago / 60)}m`;
        ctx.fillText(label, x + 2 * dpr, h - 2 * dpr);
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, midY); ctx.lineTo(w - padR, midY); ctx.stroke();

    const sc = (h * 0.42) / maxSNR;
    drawYAxisSNR(ctx, w, h, dpr, maxSNR, pad, midY, padR);

    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    ctx.beginPath();
    for (let i = 0; i < envelope.length; i++) {
        const x = dataStartX + (i / envelope.length) * (drawW * dataFrac);
        const y = Math.max(0, Math.min(h, midY - (envelope[i].mx / rms) * sc));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    for (let i = envelope.length - 1; i >= 0; i--) {
        const x = dataStartX + (i / envelope.length) * (drawW * dataFrac);
        const y = Math.max(0, Math.min(h, midY - (envelope[i].mn / rms) * sc));
        ctx.lineTo(x, y);
    }
    ctx.closePath(); ctx.fill();

    ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    for (let i = 0; i < envelope.length; i++) {
        const x = dataStartX + (i / envelope.length) * (drawW * dataFrac);
        const y = Math.max(0, Math.min(h, midY - (envelope[i].mx / rms) * sc));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.beginPath();
    for (let i = 0; i < envelope.length; i++) {
        const x = dataStartX + (i / envelope.length) * (drawW * dataFrac);
        const y = Math.max(0, Math.min(h, midY - (envelope[i].mn / rms) * sc));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = `${8 * dpr}px "JetBrains Mono", monospace`;
    ctx.fillText('NOW', w - padR - 22 * dpr, h - 2 * dpr);
}

function startEmbeddedSeismo(containerId, stationKey, stationName, defaultScale) {
    stopEmbeddedSeismo(containerId);
    const scale = defaultScale || '6h';
    const entry = { station: stationKey, name: stationName, scale: scale, buffer: [], envelope: [], sampleRate: 5, mode: 'raw', interval: null };
    state.embeddedSeismos[containerId] = entry;

    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `
        <div class="embedded-seismo-header">
            <span class="embedded-seismo-station">${stationKey}</span>
            <span class="embedded-seismo-name">${stationName}</span>
            <div class="embedded-seismo-scales">
                <button class="embedded-scale-btn${scale === '6h' ? ' active' : ''}" data-scale="6h" data-cid="${containerId}">6h</button>
                <button class="embedded-scale-btn${scale === '12h' ? ' active' : ''}" data-scale="12h" data-cid="${containerId}">12h</button>
                <button class="embedded-scale-btn${scale === '24h' ? ' active' : ''}" data-scale="24h" data-cid="${containerId}">24h</button>
            </div>
        </div>
        <canvas class="embedded-seismo-canvas" id="${containerId}-canvas"></canvas>`;

    container.querySelectorAll('.embedded-scale-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const cid = btn.dataset.cid;
            const e = state.embeddedSeismos[cid];
            if (!e) return;
            e.scale = btn.dataset.scale;
            container.querySelectorAll('.embedded-scale-btn').forEach(b => b.classList.toggle('active', b.dataset.scale === e.scale));
            fetchEmbeddedWaveform(cid);
        });
    });

    fetchEmbeddedWaveform(containerId);
    entry.interval = setInterval(() => fetchEmbeddedWaveform(containerId), 3000);
}

function stopEmbeddedSeismo(containerId) {
    const entry = state.embeddedSeismos[containerId];
    if (entry) {
        if (entry.interval) clearInterval(entry.interval);
        delete state.embeddedSeismos[containerId];
    }
}

function stopAllEmbeddedSeismos() {
    for (const cid of Object.keys(state.embeddedSeismos)) stopEmbeddedSeismo(cid);
}

async function fetchEmbeddedWaveform(containerId) {
    const entry = state.embeddedSeismos[containerId];
    if (!entry) return;
    try {
        const resp = await fetch(`/api/seedlink/waveform/${entry.station}?scale=${entry.scale}`);
        const data = await resp.json();
        entry.mode = data.mode || 'raw';
        if (data.mode === 'envelope' && data.envelope) {
            entry.envelope = data.envelope;
            entry.dataSeconds = data.data_seconds || 0;
            entry.windowSeconds = data.window_seconds || 0;
        } else if (data.samples && data.samples.length > 0) {
            entry.buffer = data.samples;
            entry.sampleRate = data.sample_rate || 5;
        }
        drawEmbeddedSeismo(containerId);
    } catch (e) { /* retry next interval */ }
}

function drawEmbeddedSeismo(containerId) {
    const entry = state.embeddedSeismos[containerId];
    if (!entry) return;
    const canvas = document.getElementById(`${containerId}-canvas`);
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0) return;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (entry.mode === 'envelope') {
        drawGenericEnvelope(ctx, canvas.width, canvas.height, dpr, entry.envelope, entry.scale, entry.dataSeconds, entry.windowSeconds);
    } else {
        drawGenericWaveform(ctx, canvas.width, canvas.height, dpr, entry.buffer, entry.sampleRate, entry.scale);
    }
}

// ── Tidal sensitivity ripples ────────────────────────────
function updateTidalRipples(data) {
    if (!data || !data.zones) return;
    state.tidalData = data;
    const zones = data.zones;
    const seen = new Set();

    for (const z of zones) {
        if (z.lat == null || z.lon == null) continue;
        seen.add(z.id);

        const threat = z.threat || 0;

        if (threat < 0.05) {
            if (state.tidalMarkers[z.id]) {
                state.tidalMarkers[z.id].remove();
                delete state.tidalMarkers[z.id];
            }
            continue;
        }

        let tierClass, rippleCount, rippleSize, duration;
        if (threat > 0.25) {
            tierClass = 'critical';
            rippleCount = 3;
            rippleSize = 70 + Math.min(40, (threat - 0.25) * 200);
            duration = 2.0;
        } else if (threat > 0.15) {
            tierClass = 'elevated';
            rippleCount = 2;
            rippleSize = 40 + (threat - 0.15) * 300;
            duration = 3.0;
        } else {
            tierClass = 'watch';
            rippleCount = 1;
            rippleSize = 25;
            duration = 4.5;
        }

        let ripples = '';
        const style = `--ripple-size:${rippleSize.toFixed(0)}px;--ripple-duration:${duration}s`;
        ripples += `<div class="tidal-ripple ${tierClass}" style="${style}"></div>`;
        if (rippleCount >= 2) ripples += `<div class="tidal-ripple r2 ${tierClass}" style="${style}"></div>`;
        if (rippleCount >= 3) ripples += `<div class="tidal-ripple r3 ${tierClass}" style="${style}"></div>`;

        const html = `<div class="tidal-ripple-container">
            <div class="tidal-center-dot ${tierClass}"></div>
            ${ripples}
        </div>`;

        const stressPct = ((z.tidal_stress || 0) * 100).toFixed(0);
        const tierLabel = threat > 0.25 ? '<span class="tidal-crit">TIDAL THREAT — ACTIVE</span>'
            : threat > 0.15 ? '<span class="tidal-sig">TIDAL LOADING</span>'
            : '<span class="tidal-watch">TIDAL WATCH</span>';
        const tipHtml = `<b>${z.name}</b><br>` +
            `${tierLabel}<br>` +
            `Tidal forcing: ${stressPct}%<br>` +
            `Sensitivity R=${z.R.toFixed(4)}<br>` +
            `${z.n_quakes} quakes in 30d`;

        if (state.tidalMarkers[z.id]) {
            state.tidalMarkers[z.id].getElement().innerHTML = html;
            state.tidalMarkers[z.id]._tipHtml = tipHtml;
        } else {
            const el = document.createElement('div');
            el.innerHTML = html;
            el.style.cursor = 'pointer';
            const m = new maplibregl.Marker({ element: el, anchor: 'center' })
                .setLngLat([z.lon, z.lat])
                .addTo(map);
            setTooltip(m, tipHtml);
            state.tidalMarkers[z.id] = m;
            if (window._legendLayerOn && !window._legendLayerOn('tidal')) {
                el.style.display = 'none';
            }
        }
    }

    for (const id of Object.keys(state.tidalMarkers)) {
        if (!seen.has(id)) {
            state.tidalMarkers[id].remove();
            delete state.tidalMarkers[id];
        }
    }
}


// ── DART buoy markers ───────────────────────────────────
function updateDartMarkers(data) {
    const stations = data.stations || [];
    const seen = new Set();

    for (const stn of stations) {
        if (stn.lat == null || stn.lon == null) continue;
        const key = stn.station_id;
        seen.add(key);

        const isEvent = stn.mode === 'event';
        const isTsunami = stn.mode === 'tsunami';
        const isElevated = !isEvent && !isTsunami && (stn.deviation || 0) >= 1.5;
        const modeClass = isTsunami ? 'dart-tsunami' : isEvent ? 'dart-event'
                        : isElevated ? 'dart-elevated' : 'dart-normal';

        const html = `<div class="dart-marker ${modeClass}"></div>`;

        const modeLabel = isTsunami ? '<span class="dart-alert">TSUNAMI MODE (1-min)</span>'
                        : isEvent ? '<span class="dart-alert">EVENT MODE (1-min)</span>'
                        : isElevated ? '<span class="dart-alert">ELEVATED SIGNAL</span>'
                        : 'Normal (15-min)';
        const devStr = stn.deviation != null ? `<br>Deviation: ${stn.deviation.toFixed(1)}σ` : '';
        const heightStr = stn.height_m != null ? `<br>Depth: ${stn.height_m.toFixed(1)}m` : '';
        const tipHtml = `<b>DART ${key}</b><br>${stn.region}<br>${modeLabel}${heightStr}${devStr}<br>Last: ${stn.last_reading} UTC`;

        if (state.dartMarkers[key]) {
            state.dartMarkers[key].getElement().innerHTML = html;
            state.dartMarkers[key]._tipHtml = tipHtml;
        } else {
            const el = document.createElement('div');
            el.innerHTML = html;
            el.style.cursor = 'pointer';
            const m = new maplibregl.Marker({ element: el, anchor: 'center' })
                .setLngLat([stn.lon, stn.lat])
                .addTo(map);
            setTooltip(m, tipHtml);
            state.dartMarkers[key] = m;
            if (window._legendLayerOn && !window._legendLayerOn('dart')) {
                el.style.display = 'none';
            }
        }
    }
}

// ── Earthquake feed ──────────────────────────────────────
function updateEqFeed(quakes) {
    const el = document.getElementById('eq-feed');
    const cutoff = new Date(Date.now() - state.filterHours * 3600000).toISOString();
    const filtered = quakes
        .filter(eq => eq.magnitude >= state.filterMag && eq.timestamp >= cutoff)
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    const countEl = document.getElementById('eq-count');
    if (countEl) countEl.textContent = filtered.length + ' events';
    el.innerHTML = filtered.slice(0, 50).map(eq => {
        const color = magColor(eq.magnitude);
        const place = (eq.place || 'Unknown').replace(/^.+? of /, '');
        const depth = eq.depth_km ? `${eq.depth_km.toFixed(0)}km` : '';
        return `<div class="eq-feed-item" data-lat="${eq.lat}" data-lon="${eq.lon}" data-id="${eq.id}">
            <div class="eq-feed-mag" style="color:${color}">M${eq.magnitude.toFixed(1)}</div>
            <div class="eq-feed-info">
                <div class="eq-feed-place">${place}</div>
                <div class="eq-feed-meta">${depth ? depth + ' &middot; ' : ''}${timeAgo(eq.timestamp)}</div>
            </div>
        </div>`;
    }).join('');

    el.querySelectorAll('.eq-feed-item').forEach(item => {
        item.addEventListener('click', () => {
            const lat = parseFloat(item.dataset.lat);
            const lon = parseFloat(item.dataset.lon);
            map.flyTo({ center: [lon, lat], zoom: 7, speed: 1.5 });
            const eq = quakes.find(q => q.id === item.dataset.id);
            if (eq) showEqDetail(eq);
        });
    });
}

// ── Earthquake detail ────────────────────────────────────
function showEqDetail(eq) {
    const el = document.getElementById('eq-detail');
    const content = document.getElementById('detail-content');
    const color = magColor(eq.magnitude);
    const place = (eq.place || 'Unknown location').replace(/'/g, '');
    content.innerHTML = `
        <div class="detail-mag" style="color:${color}">M${eq.magnitude.toFixed(1)}</div>
        <div class="detail-place">${eq.place || 'Unknown location'}</div>
        <div class="detail-grid">
            <div><div class="detail-label">Depth</div><div class="detail-value">${eq.depth_km ? eq.depth_km.toFixed(1) + ' km' : '--'}</div></div>
            <div><div class="detail-label">Time</div><div class="detail-value">${timeAgo(eq.timestamp)}</div></div>
            <div><div class="detail-label">Coords</div><div class="detail-value">${eq.lat.toFixed(2)}, ${eq.lon.toFixed(2)}</div></div>
            <div><div class="detail-label">UTC</div><div class="detail-value">${new Date(eq.timestamp).toISOString().slice(11,16)}</div></div>
        </div>
        <button class="detail-predict-btn" data-lat="${eq.lat}" data-lon="${eq.lon}" data-name="${place}">Analyze region</button>
    `;
    el.classList.add('visible');
}

function hideEqDetail() {
    document.getElementById('eq-detail').classList.remove('visible');
}

// ── Signal cards (now feeds conditions panel) ───────────
function updateSignalCards(signals) {
    updateConditions(signals);
}

// ── Risk arc SVG ─────────────────────────────────────────
function riskArcSvg(score, color, size) {
    const r = (size - 6) / 2;
    const cx = size / 2, cy = size / 2;
    const circumference = Math.PI * r;
    const filled = circumference * score;
    const pct = (score * 100).toFixed(0);
    return `<svg class="risk-arc-svg" width="${size}" height="${size / 2 + 4}" viewBox="0 0 ${size} ${size / 2 + 4}">
        <path d="M 3 ${cy} A ${r} ${r} 0 0 1 ${size - 3} ${cy}"
              fill="none" stroke="rgba(255,255,255,0.04)" stroke-width="3" stroke-linecap="round"/>
        <path d="M 3 ${cy} A ${r} ${r} 0 0 1 ${size - 3} ${cy}"
              fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round"
              stroke-dasharray="${filled} ${circumference}"
              style="filter:drop-shadow(0 0 3px ${color})"/>
        <text x="${cx}" y="${cy - 4}" text-anchor="middle" fill="${color}"
              font-size="${Math.round(size * 0.22)}" font-weight="600" font-family="Inter,sans-serif">${pct}%</text>
    </svg>`;
}

// ── Geo trend helper ─────────────────────────────────────
function geoTrend(key) {
    const hist = state.signalHistory[key];
    if (!hist || hist.length < 4) return { arrow: '', cls: 'trend-flat' };
    const recent = hist.slice(-3).reduce((s, d) => s + (typeof d === 'number' ? d : d.v), 0) / 3;
    const older = hist.slice(-8, -3);
    if (older.length === 0) return { arrow: '', cls: 'trend-flat' };
    const prev = older.reduce((s, d) => s + (typeof d === 'number' ? d : d.v), 0) / older.length;
    const delta = recent - prev;
    const pctChange = prev !== 0 ? Math.abs(delta / prev) : 0;
    if (pctChange < 0.05) return { arrow: '—', cls: 'trend-flat' };
    return delta > 0 ? { arrow: '▲', cls: 'trend-up' } : { arrow: '▼', cls: 'trend-down' };
}

function geoStatusLabel(key, value) {
    if (key.includes('kp')) {
        if (value >= 7) return { text: 'Severe storm', cls: 'geo-storm' };
        if (value >= 5) return { text: 'Storm', cls: 'geo-storm' };
        if (value >= 4) return { text: 'Active', cls: 'geo-active' };
        return { text: 'Quiet', cls: 'geo-quiet' };
    }
    if (key.includes('dst')) {
        if (value < -100) return { text: 'Intense storm', cls: 'geo-storm' };
        if (value < -50) return { text: 'Moderate storm', cls: 'geo-storm' };
        if (value < -30) return { text: 'Weak disturbance', cls: 'geo-active' };
        return { text: 'Quiet', cls: 'geo-quiet' };
    }
    return { text: '', cls: 'geo-quiet' };
}

function geoBarPct(key, value) {
    if (key.includes('kp')) return Math.min(100, (value / 9) * 100);
    if (key.includes('dst')) return Math.min(100, (Math.abs(value) / 150) * 100);
    return 30;
}

// ── Footer prediction (actionable) ───────────────────────
async function predictLocation(lat, lon, name) {
    const zoneName = name || guessRegionName(lat, lon);
    try {
        const resp = await fetch('/api/threats');
        if (resp.ok) {
            const allThreats = await resp.json();
            const zoneData = allThreats.find(z => z.name === zoneName) || allThreats.find(z => {
                const d = Math.abs(z.center[0] - lat) + Math.abs(z.center[1] - lon);
                return d < 15;
            }) || null;
            openWorkbench(zoneName, zoneData);
        } else {
            openWorkbench(zoneName, null);
        }
    } catch (e) {
        openWorkbench(zoneName, null);
    }
}

// ── Predictive fault risk (from API) ─────────────────────
async function refreshFaultRisk() {
    if (!mapReady) return;
    try {
        const resp = await fetch('/api/predict/faults');
        if (!resp.ok) return;
        const data = await resp.json();

        const heatFeatures = [];
        const lineFeatures = [];

        for (const seg of data.segments) {
            const a = seg.a, b = seg.b;
            const risk = seg.risk;

            const steps = 6;
            for (let s = 0; s <= steps; s++) {
                const t = s / steps;
                const lat = a[0] + (b[0] - a[0]) * t;
                const lon = a[1] + (b[1] - a[1]) * t;
                if (risk > 0.2) {
                    heatFeatures.push({
                        type: 'Feature',
                        geometry: { type: 'Point', coordinates: [lon, lat] },
                        properties: { weight: Math.pow(risk, 2) }
                    });
                }
            }

            if (risk > 0.25) {
                const t = Math.min(1, (risk - 0.25) / 0.55);
                const r = Math.round(200 + t * 55);
                const g = Math.round(160 * (1 - t));
                const bv = Math.round(40 * (1 - t));
                const opacity = 0.2 + t * 0.6;
                const weight = 1 + t * 2.5;

                lineFeatures.push({
                    type: 'Feature',
                    geometry: { type: 'LineString', coordinates: [[a[1], a[0]], [b[1], b[0]]] },
                    properties: {
                        color: `rgb(${r},${g},${bv})`,
                        width: weight,
                        opacity: opacity,
                        glowWidth: weight + 4 + t * 6,
                        glowOpacity: opacity * 0.3,
                    }
                });
            }
        }

        const fhSrc = map.getSource('fault-heat');
        if (fhSrc) fhSrc.setData({ type: 'FeatureCollection', features: heatFeatures });
        const flSrc = map.getSource('fault-lines');
        if (flSrc) flSrc.setData({ type: 'FeatureCollection', features: lineFeatures });
    } catch (e) {
        console.error('Fault risk error:', e);
    }
}

// ── Heatmap (seismicity-based) ───────────────────────────
async function refreshHeatmap() {
    if (!mapReady) return;
    try {
        const thirtyDays = new Date(Date.now() - 30 * 86400000).toISOString();
        const resp = await fetch(`/api/earthquakes?start=${thirtyDays}&min_mag=2.5&limit=500`);
        if (!resp.ok) return;
        const quakes = await resp.json();
        if (!Array.isArray(quakes) || quakes.length === 0) return;
        const features = quakes
            .filter(q => q.lat != null && q.lon != null)
            .map(q => {
                const age = (Date.now() - new Date(q.timestamp).getTime()) / (30 * 86400000);
                const recency = Math.max(0, 1 - age);
                const magWeight = Math.pow((q.magnitude - 2) / 6, 1.5);
                return {
                    type: 'Feature',
                    geometry: { type: 'Point', coordinates: [q.lon, q.lat] },
                    properties: { weight: Math.min(1, magWeight * (0.3 + 0.7 * recency)) }
                };
            });
        const src = map.getSource('eq-heat');
        if (src) src.setData({ type: 'FeatureCollection', features });
    } catch (e) {
        console.error('Heatmap error:', e);
    }
}

// ── Swarm Watch Detail Overlay ──────────────────────────
function openSwarmDetail(cellId) {
    if (!_swarmWatchData) return;
    const overlay = document.getElementById('fc-overlay');
    overlay.classList.add('open');

    const cell = _swarmWatchData.watch.find(w => w.cell === cellId);
    if (!cell) {
        document.getElementById('fc-content').innerHTML = '<div class="fc-loading">No data for this swarm</div>';
        return;
    }

    const quakes = (cell.quakes || []).map(q => ({
        ...q, magnitude: q.mag, timestamp: new Date(q.time).getTime()
    })).sort((a, b) => b.timestamp - a.timestamp);

    renderSwarmDetail(cell, _swarmWatchData, quakes);
}

function closeSwarmDetail() {
    stopAllEmbeddedSeismos();
    document.getElementById('fc-overlay').classList.remove('open');
}

document.getElementById('fc-close')?.addEventListener('click', closeSwarmDetail);
document.getElementById('fc-overlay')?.addEventListener('click', (e) => {
    if (e.target.id === 'fc-overlay') closeSwarmDetail();
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.getElementById('fc-overlay')?.classList.contains('open')) {
        closeSwarmDetail();
    }
});

function renderSwarmDetail(cell, data, zoneQuakes) {
    const content = document.getElementById('fc-content');
    const label = SW_ZONE_LABELS[cell.zone] || cell.zone;
    const isExperimental = cell.model_skill === 'experimental';
    const pct = (cell.escalation_prob_72h * 100).toFixed(1);
    const effectiveLevel = isExperimental ? 'OBSERVED' : cell.alert_level;
    const alertClass = isExperimental ? 'normal' : cell.alert_level.toLowerCase();
    const baseRate = (data.base_rate_72h * 100).toFixed(1);
    const gaugeColor = isExperimental ? 'rgba(160,160,160,0.5)' : (SW_ALERT_COLORS[cell.alert_level] || 'rgba(255,255,255,0.7)');

    const circumference = 2 * Math.PI * 52;
    const offset = circumference * (1 - (isExperimental ? 0 : cell.escalation_prob_72h));

    const latC = cell.centroid[0];
    const lonC = cell.centroid[1];
    const locStr = `${Math.abs(latC).toFixed(1)}°${latC >= 0 ? 'N' : 'S'}, ${Math.abs(lonC).toFixed(1)}°${lonC >= 0 ? 'E' : 'W'}`;
    const dirLabel = cell.direction === 'RISING' ? '↑ Rising' : cell.direction === 'FALLING' ? '↓ Falling' : '→ Steady';
    const volcHtml = cell.nearby_volcanic_alert
        ? `<div class="sw-seis-stat"><span class="sw-seis-num" style="color:rgba(220,70,50,0.9)">${cell.nearby_volcanic_alert.volcano}</span><span class="sw-seis-label">${cell.nearby_volcanic_alert.level} volcanic alert</span></div>`
        : '';

    const totalModelQuakes = cell.n_recent_quakes || 0;
    if (!zoneQuakes) zoneQuakes = [];

    const magBuckets = { '2-3': 0, '3-4': 0, '4-5': 0, '5+': 0 };
    for (const eq of zoneQuakes) {
        if (eq.magnitude >= 5) magBuckets['5+']++;
        else if (eq.magnitude >= 4) magBuckets['4-5']++;
        else if (eq.magnitude >= 3) magBuckets['3-4']++;
        else magBuckets['2-3']++;
    }
    const maxBucket = Math.max(1, ...Object.values(magBuckets));
    const magBarColors = { '2-3': 'rgba(255,255,255,0.5)', '3-4': 'rgba(255,255,255,0.5)', '4-5': 'rgba(255,255,255,0.5)', '5+': 'rgba(220,70,50,0.9)' };
    let magBarsHtml = '';
    for (const [range, count] of Object.entries(magBuckets)) {
        const pctW = Math.max(2, (count / maxBucket) * 100);
        magBarsHtml += `<div class="sw-mag-row">
            <span class="sw-mag-label">M${range}</span>
            <div class="sw-mag-bar-track"><div class="sw-mag-bar-fill" style="width:${pctW}%;background:${magBarColors[range]}"></div></div>
            <span class="sw-mag-count">${count}</span>
        </div>`;
    }

    // Timeline data for seismograph canvas (last 48h)
    const _tlQuakes = [...zoneQuakes].sort((a, b) => a.timestamp - b.timestamp);
    const _tlDataAttr = JSON.stringify(_tlQuakes.map(q => [q.timestamp, q.magnitude]));

    let recentListHtml = '';
    const showQuakes = zoneQuakes;
    for (const eq of showQuakes) {
        const mc = magColor(eq.magnitude);
        const place = (eq.place || 'Unknown').replace(/^.+? of /, '');
        const depth = eq.depth_km ? `${eq.depth_km.toFixed(0)}km` : '';
        recentListHtml += `<div class="sw-quake-item">
            <span class="sw-quake-mag" style="color:${mc}">M${eq.magnitude.toFixed(1)}</span>
            <span class="sw-quake-place">${place}</span>
            <span class="sw-quake-meta">${depth}${depth ? ' · ' : ''}${timeAgo(eq.timestamp)}</span>
        </div>`;
    }

    const heroQuestion = isExperimental
        ? 'Swarm activity observed — experimental zone'
        : `Active swarm — ${pct}% escalation risk (72h)`;
    const heroDesc = isExperimental
        ? 'This zone has not been validated for forecasting. Showing observed activity only.'
        : 'Calibrated probability of M5+ mainshock within 72 hours';

    content.innerHTML = `
        <div class="fc-hero">
            ${isExperimental ? '' : `<div class="fc-gauge">
                <svg viewBox="0 0 120 120">
                    <circle class="fc-gauge-track" cx="60" cy="60" r="52"/>
                    <circle class="fc-gauge-fill" cx="60" cy="60" r="52"
                        stroke="${gaugeColor}"
                        stroke-dasharray="${circumference}"
                        stroke-dashoffset="${circumference}"/>
                </svg>
                <div class="fc-gauge-pct">
                    ${pct}<span class="fc-gauge-label">%</span>
                </div>
            </div>`}
            <div class="fc-hero-info">
                <div class="fc-hero-zone">${label}</div>
                <div class="sw-hero-question">${heroQuestion}</div>
                <div class="sw-hero-desc">${heroDesc}</div>
                <span class="sw-hero-alert sw-alert sw-alert-${alertClass}">${effectiveLevel}</span>
                ${isExperimental
                    ? `<div class="sw-hero-lift">${locStr} · ${dirLabel}</div>`
                    : `<div class="sw-hero-lift">${cell.lift_vs_base.toFixed(1)}× baseline (baseline: ${baseRate}%) · ${locStr} · ${dirLabel}</div>`
                }
            </div>
        </div>

        <div class="fc-grid">
            <div class="fc-main">
                <div class="fc-section">
                    <div class="fc-section-title">Seismicity Overview</div>
                    <div class="sw-seismicity-summary">
                        <div class="sw-seis-stat">
                            <span class="sw-seis-num">${totalModelQuakes}</span>
                            <span class="sw-seis-label">quakes in swarm</span>
                        </div>
                        <div class="sw-seis-stat">
                            <span class="sw-seis-num">${dirLabel}</span>
                            <span class="sw-seis-label">trend</span>
                        </div>
                        ${volcHtml}
                    </div>
                </div>

                <div class="fc-section">
                    <div class="fc-section-title">Magnitude Distribution</div>
                    <div class="sw-mag-chart">${magBarsHtml}</div>
                </div>

                <div class="fc-section">
                    <div class="fc-section-title">Timeline (48h)</div>
                    <div class="sw-timeline-chart">
                        <canvas id="sw-tl-canvas" data-quakes='${_tlDataAttr}' style="width:100%;height:48px"></canvas>
                        <div class="sw-tl-axis">
                            <span>48h ago</span><span>24h ago</span><span>now</span>
                        </div>
                    </div>
                </div>

                ${recentListHtml ? `<div class="fc-section">
                    <div class="fc-section-title">Recent Quakes in Zone</div>
                    <div class="sw-quake-list">${recentListHtml}</div>
                </div>` : ''}

                <div class="fc-section">
                    <div class="fc-section-title">Swarm Location</div>
                    <div class="sw-cells-list"><div class="sw-cell-row">
                        <span class="sw-cell-loc">${locStr}</span>
                        ${isExperimental ? '' : `<span class="sw-cell-prob">${pct}%</span>
                        <span class="sw-cell-lift">${cell.lift_vs_base.toFixed(1)}×</span>`}
                    </div></div>
                </div>
            </div>

            <div class="fc-side">
                ${isExperimental ? '' : `<div class="fc-section">
                    <div class="fc-section-title">Calibration Context</div>
                    <div class="sw-calibration">
                        <div class="sw-cal-row">
                            <span class="sw-cal-label">Baseline (any swarm)</span>
                            <span class="sw-cal-val">${baseRate}%</span>
                        </div>
                        <div class="sw-cal-row">
                            <span class="sw-cal-label">This cell</span>
                            <span class="sw-cal-val">${pct}%</span>
                        </div>
                        <div class="sw-cal-row">
                            <span class="sw-cal-label">Lift vs baseline</span>
                            <span class="sw-cal-val">${cell.lift_vs_base.toFixed(1)}×</span>
                        </div>
                    </div>
                </div>`}

                <div class="fc-section">
                    <div class="fc-section-title">What This Means</div>
                    <div class="sw-context">
                        ${isExperimental ? `
                        <div class="sw-context-item">This zone is experimental — the model has not been validated here</div>
                        <div class="sw-context-item">Showing observed swarm activity, not a forecast</div>
                        <div class="sw-context-item">Validated zones: Alaska, South America, New Zealand, Japan/Kurils</div>
                        ` : `
                        <div class="sw-context-item">~94% of active swarms fizzle without producing a large event</div>
                        <div class="sw-context-item">This model identifies the ~6% that escalate to M5+ mainshocks</div>
                        <div class="sw-context-item">Probabilities are calibrated — ${pct}% means approximately ${pct}% actually escalate</div>
                        <div class="sw-context-item">${effectiveLevel} is a watch posture, not a prediction</div>
                        `}
                    </div>
                </div>

                <div class="fc-section">
                    <div class="fc-section-title">Alert Levels</div>
                    <div class="sw-alert-legend">
                        <div class="sw-legend-row"><span class="sw-alert sw-alert-watch">WATCH</span><span>Notably elevated — top 5%</span></div>
                        <div class="sw-legend-row"><span class="sw-alert sw-alert-advisory">ADVISORY</span><span>Worth watching — top 20%</span></div>
                        <div class="sw-legend-row"><span class="sw-alert sw-alert-normal">NORMAL</span><span>Baseline risk</span></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="fc-seismo-section" id="fc-seismo-section"></div>
    `;

    stopAllEmbeddedSeismos();
    requestAnimationFrame(() => {
        const fill = content.querySelector('.fc-gauge-fill');
        if (fill) fill.style.strokeDashoffset = offset;

        const tlCanvas = document.getElementById('sw-tl-canvas');
        if (tlCanvas) {
            const rect = tlCanvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            const cw = rect.width, ch = rect.height;
            tlCanvas.width = cw * dpr;
            tlCanvas.height = ch * dpr;
            const ctx = tlCanvas.getContext('2d');
            ctx.scale(dpr, dpr);

            let qdata = [];
            try { qdata = JSON.parse(tlCanvas.dataset.quakes || '[]'); } catch(e) {}

            const tNow = Date.now();
            const t48 = tNow - 48 * 60 * 60 * 1000;

            // Bucket into 96 slots (30min each) for smooth line
            const nSlots = 96;
            const slotMs = (48 * 60 * 60 * 1000) / nSlots;
            const slots = new Array(nSlots).fill(0);
            for (const [ts, mag] of qdata) {
                const si = Math.min(nSlots - 1, Math.max(0, Math.floor((ts - t48) / slotMs)));
                slots[si] += mag;
            }
            const maxSlot = Math.max(1, ...slots);

            // Draw baseline
            ctx.strokeStyle = 'rgba(255,255,255,0.08)';
            ctx.lineWidth = 1;
            const baseY = ch * 0.5;
            ctx.beginPath();
            ctx.moveTo(0, baseY);
            ctx.lineTo(cw, baseY);
            ctx.stroke();

            // Draw seismograph line
            ctx.strokeStyle = 'rgba(255,255,255,0.6)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i < nSlots; i++) {
                const x = (i / (nSlots - 1)) * cw;
                const amp = (slots[i] / maxSlot) * (ch * 0.45);
                const y = baseY - amp;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Fill under the line
            ctx.lineTo(cw, baseY);
            ctx.lineTo(0, baseY);
            ctx.closePath();
            ctx.fillStyle = 'rgba(255,255,255,0.04)';
            ctx.fill();

            // Draw individual quake ticks
            ctx.fillStyle = 'rgba(255,255,255,0.3)';
            for (const [ts, mag] of qdata) {
                const x = ((ts - t48) / (tNow - t48)) * cw;
                const tickH = Math.max(2, (mag / 6) * ch * 0.4);
                ctx.fillRect(x, baseY - tickH, 1, tickH);
            }
        }

        const avgLat = cell.centroid[0];
        const avgLon = cell.centroid[1];
        const nearest = findNearestStations(avgLat, avgLon, 2);
        const section = document.getElementById('fc-seismo-section');
        if (nearest.length > 0 && section) {
            let html = '<div class="fc-section-title">Nearest Stations — Live Seismograph</div>';
            for (let i = 0; i < nearest.length; i++) {
                html += `<div class="embedded-seismo-wrap" id="fc-seismo-${i}"></div>`;
            }
            section.innerHTML = html;
            for (let i = 0; i < nearest.length; i++) {
                startEmbeddedSeismo(`fc-seismo-${i}`, nearest[i].station, nearest[i].name, '6h');
            }
        }
    });
}

let _swarmZoneHandlers = false;

function updateSwarmPolygons(data) {
    if (!mapReady) return;

    const watch = data.watch || [];

    // One extent per validated, non-NORMAL cell
    const extentFeatures = watch
        .filter(w => w.extent && w.alert_level !== 'NORMAL' && w.model_skill === 'validated')
        .map(w => ({
            type: 'Feature',
            properties: {
                alert_level: w.alert_level,
                cell: w.cell,
                zone: w.zone,
                zone_label: SW_ZONE_LABELS[w.zone] || w.zone,
                n_quakes: w.n_recent_quakes || 0,
            },
            geometry: {
                type: 'Polygon',
                coordinates: [[
                    [w.extent[2], w.extent[0]],
                    [w.extent[3], w.extent[0]],
                    [w.extent[3], w.extent[1]],
                    [w.extent[2], w.extent[1]],
                    [w.extent[2], w.extent[0]],
                ]]
            }
        }));

    const extentGeo = { type: 'FeatureCollection', features: extentFeatures };
    if (map.getSource('swarm-cells')) {
        map.getSource('swarm-cells').setData(extentGeo);
    } else {
        map.addSource('swarm-cells', { type: 'geojson', data: extentGeo });
        map.addLayer({
            id: 'swarm-cells-fill',
            type: 'fill',
            source: 'swarm-cells',
            paint: {
                'fill-color': [
                    'match', ['get', 'alert_level'],
                    'WATCH', 'rgba(230,160,50,0.10)',
                    'ADVISORY', 'rgba(220,200,60,0.08)',
                    'rgba(160,160,160,0.03)'
                ],
                'fill-opacity': 0.8,
            }
        });
        map.addLayer({
            id: 'swarm-cells-border',
            type: 'line',
            source: 'swarm-cells',
            paint: {
                'line-color': [
                    'match', ['get', 'alert_level'],
                    'WATCH', 'rgba(230,160,50,0.4)',
                    'ADVISORY', 'rgba(220,200,60,0.3)',
                    'rgba(160,160,160,0.1)'
                ],
                'line-width': 1,
                'line-dasharray': [3, 2],
            }
        });

        map.on('mouseenter', 'swarm-cells-fill', (e) => {
            map.getCanvas().style.cursor = 'pointer';
            const p = e.features[0].properties;
            const color = SW_ALERT_COLORS[p.alert_level] || 'rgba(160,160,160,0.6)';
            const html = `<div style="font-size:11px"><b>${p.zone_label}</b> · <span style="color:${color};font-weight:600">${p.alert_level}</span> · ${p.n_quakes} quakes</div>`;
            hoverPopup.setHTML(html).setLngLat(e.lngLat).addTo(map);
        });
        map.on('mouseleave', 'swarm-cells-fill', () => {
            map.getCanvas().style.cursor = '';
            hoverPopup.remove();
        });
        map.on('click', 'swarm-cells-fill', (e) => {
            openSwarmDetail(e.features[0].properties.cell);
        });
        _swarmZoneHandlers = true;
    }

    // Render quakes from embedded quakes[] arrays on the map
    _renderSwarmQuakes(watch);
}

function _renderSwarmQuakes(watch) {
    const features = [];
    for (const w of watch) {
        if (!w.quakes || !w.quakes.length || w.alert_level === 'NORMAL' || w.model_skill === 'experimental') continue;
        for (const eq of w.quakes) {
            const mag = eq.mag || 0;
            let color, radius;
            if (mag >= 5) { color = 'rgba(220,70,50,0.9)'; radius = 6; }
            else { color = 'rgba(255,255,255,0.5)'; radius = Math.max(2, mag); }
            features.push({
                type: 'Feature',
                properties: { mag, color, radius, cell: w.cell,
                    place: eq.place || '', depth: eq.depth_km || 0, time: eq.time || '' },
                geometry: { type: 'Point', coordinates: [eq.lon, eq.lat] }
            });
        }
    }
    const geo = { type: 'FeatureCollection', features };
    if (map.getSource('swarm-zone-quakes')) {
        map.getSource('swarm-zone-quakes').setData(geo);
    } else {
        map.addSource('swarm-zone-quakes', { type: 'geojson', data: geo });
        map.addLayer({
            id: 'swarm-zone-quakes-dot',
            type: 'circle',
            source: 'swarm-zone-quakes',
            paint: {
                'circle-radius': ['get', 'radius'],
                'circle-color': ['get', 'color'],
                'circle-stroke-width': 0.5,
                'circle-stroke-color': 'rgba(255,255,255,0.15)',
            }
        });
        map.on('mouseenter', 'swarm-zone-quakes-dot', (e) => {
            map.getCanvas().style.cursor = 'pointer';
            const p = e.features[0].properties;
            const place = (p.place || '').replace(/^.+? of /, '');
            const depth = p.depth ? `${Math.round(p.depth)}km` : '';
            const ago = p.time ? timeAgo(new Date(p.time).getTime()) : '';
            hoverPopup.setHTML(`<div style="font-size:11px"><b>M${p.mag.toFixed(1)}</b> ${place}<br>${depth}${depth && ago ? ' · ' : ''}${ago}</div>`)
                .setLngLat(e.lngLat).addTo(map);
        });
        map.on('mouseleave', 'swarm-zone-quakes-dot', () => {
            map.getCanvas().style.cursor = '';
            hoverPopup.remove();
        });
    }
}

// ── SSE Stream ───────────────────────────────────────────
function connectStream() {
    const es = new EventSource(CONFIG.STREAM_URL);

    es.addEventListener('earthquakes', (e) => {
        try {
            const quakes = JSON.parse(e.data);
            checkNewQuakes(quakes);
            state.earthquakes = quakes;
            updateEarthquakeMarkers(quakes);
            updateEqFeed(quakes);
            if (!state._heatmapLoaded) {
                state._heatmapLoaded = true;
                refreshHeatmap();
            }
        } catch (err) { console.error('EQ parse:', err); }
    });

    es.addEventListener('threats', (e) => {
        try {
            const threats = JSON.parse(e.data);
            updateThreatMonitor(threats);
            updateZoneBoundaries(threats);
        } catch (err) { console.error('Threat parse:', err); }
    });

    es.addEventListener('signals', (e) => {
        try {
            const signals = JSON.parse(e.data);
            state.signals = signals;
            for (const [key, sig] of Object.entries(signals)) {
                if (!state.signalHistory[key]) state.signalHistory[key] = [];
                const hist = state.signalHistory[key];
                const last = hist.length > 0 ? hist[hist.length - 1] : null;
                if (!last || last.t !== sig.timestamp) {
                    hist.push({ t: sig.timestamp, v: sig.value });
                    if (hist.length > 200)
                        state.signalHistory[key] = hist.slice(-150);
                }
            }
            updateSignalCards(signals);
        } catch (err) { console.error('Signal parse:', err); }
    });

    es.addEventListener('dart', (e) => {
        try {
            const data = JSON.parse(e.data);
            updateDartMarkers(data);
            updateDartSummary(data);
        } catch (err) { console.error('DART parse:', err); }
    });

    es.addEventListener('volcanic', (e) => {
        try {
            const data = JSON.parse(e.data);
            updateVolcanicSummary(data);
            try { updateVolcanoMarkers(data); } catch (merr) { console.error('Volcano markers:', merr); }
        } catch (err) { console.error('Volcanic parse:', err); }
    });

    es.addEventListener('tidal', (e) => {
        try {
            const data = JSON.parse(e.data);
            updateTidalRipples(data);
        } catch (err) { console.error('Tidal parse:', err); }
    });

    es.addEventListener('seedlink', (e) => {
        try {
            const data = JSON.parse(e.data);
            updateStationMarkers(data.stations || []);
            if (data.detections && data.detections.length) {
                const now = Date.now();
                for (const det of data.detections) {
                    const key = det.source + det.timestamp;
                    if (det.estimated_magnitude >= 5.0
                        && !state.seenQuakeIds.has(key)
                        && now - state.lastSeedlinkAlert > 300000) {
                        state.seenQuakeIds.add(key);
                        state.lastSeedlinkAlert = now;
                        playQuakeAlert(det.estimated_magnitude);
                    } else {
                        state.seenQuakeIds.add(key);
                    }
                }
            }
        } catch (err) { console.error('SeedLink parse:', err); }
    });

    es.addEventListener('tier2-watch', (e) => {
        try {
            const data = JSON.parse(e.data);
            updateSwarmWatch(data);
        } catch (err) { console.error('Swarm watch parse:', err); }
    });

    es.onopen = () => {
        state.streamConnected = true;
        const dot = document.querySelector('.brand-dot');
        const label = document.getElementById('signal-status');
        dot.classList.add('live');
        label.textContent = 'live';
        label.style.color = 'rgba(80,200,120,0.6)';
    };

    es.onerror = () => {
        state.streamConnected = false;
        const dot = document.querySelector('.brand-dot');
        const label = document.getElementById('signal-status');
        dot.classList.remove('live');
        label.textContent = 'reconnecting';
        label.style.color = 'rgba(255,255,255,0.2)';
        es.close();
        setTimeout(connectStream, CONFIG.RECONNECT_DELAY);
    };
}

// ── Clock ────────────────────────────────────────────────
function updateClock() {
    const now = new Date();
    const h = String(now.getUTCHours()).padStart(2, '0');
    const m = String(now.getUTCMinutes()).padStart(2, '0');
    const s = String(now.getUTCSeconds()).padStart(2, '0');
    document.getElementById('hud-time').textContent = `${h}:${m}:${s}`;
}

// ── Map click ────────────────────────────────────────────
function setupMapClick() {
    map.on('click', () => {
        const detail = document.getElementById('eq-detail');
        if (detail.classList.contains('visible')) {
            hideEqDetail();
        }
    });
}

// ── Event delegation ─────────────────────────────────────
function setupDelegation() {
    document.getElementById('detail-close').addEventListener('click', hideEqDetail);
    document.getElementById('eq-detail').addEventListener('click', (e) => {
        const btn = e.target.closest('.detail-predict-btn');
        if (btn) {
            hideEqDetail();
            predictLocation(parseFloat(btn.dataset.lat), parseFloat(btn.dataset.lon), btn.dataset.name || '');
            return;
        }
    });

    // Seismograph
    const seismoClose = document.getElementById('seismo-close');
    if (seismoClose) seismoClose.addEventListener('click', closeSeismograph);
    document.querySelectorAll('.seismo-scale-btn').forEach(btn => {
        btn.addEventListener('click', () => setSeismoScale(btn.dataset.scale));
    });
    document.querySelectorAll('.seismo-mode-btn').forEach(btn => {
        btn.addEventListener('click', () => setSeismoDisplayMode(btn.dataset.mode));
    });

    // Workbench close
    document.getElementById('wb-close').addEventListener('click', closeWorkbench);

    // Workbench time filter buttons
    document.querySelectorAll('.wb-time-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.wb-time-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            workbenchState.timeHours = parseInt(btn.dataset.hours);
            saveWbPrefs();
            refreshAllWorkbenchCharts();
        });
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (state.seismoStation) { closeSeismograph(); e.stopPropagation(); return; }
            const wb = document.getElementById('wb-overlay');
            if (wb.classList.contains('visible')) {
                closeWorkbench();
                e.stopPropagation();
            }
        }
    });
}

// ── Footer resize ────────────────────────────────────────
function setupFooterResize() {
    const hud = document.querySelector('.hud');
    const handle = document.querySelector('.footer-handle');
    let dragging = false, startY = 0, startH = 0;
    const MIN_H = 6, MAX_H = 400;

    handle.addEventListener('mousedown', (e) => {
        dragging = true;
        startY = e.clientY;
        startH = parseInt(getComputedStyle(hud).gridTemplateRows.split(' ').pop()) || 160;
        document.body.style.cursor = 'row-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        const newH = Math.max(MIN_H, Math.min(MAX_H, startH + (startY - e.clientY)));
        hud.style.gridTemplateRows = `1fr ${newH}px`;
        map.resize();
    });
    window.addEventListener('mouseup', () => {
        if (!dragging) return;
        dragging = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    });
    handle.addEventListener('dblclick', () => {
        const curH = parseInt(getComputedStyle(hud).gridTemplateRows.split(' ').pop()) || 160;
        hud.style.gridTemplateRows = `1fr ${curH <= MIN_H + 10 ? 160 : MIN_H}px`;
        map.resize();
    });
}

// ── Bootstrap ────────────────────────────────────────────
async function loadSignalHistory() {
    try {
        const resp = await fetch('/api/signals/history?hours=24');
        if (!resp.ok) return;
        const data = await resp.json();
        for (const [key, info] of Object.entries(data))
            state.signalHistory[key] = info.data || [];
    } catch (e) { console.error('History load:', e); }
}

// ── Init ─────────────────────────────────────────────────
async function init() {
    setupMapClick();
    setupDelegation();

    setInterval(updateClock, 1000);
    updateClock();

    connectStream();

    // Fire all initial data loads in parallel
    Promise.all([
        loadSignalHistory(),
        fetch('/api/threats').then(r => r.ok ? r.json() : null).then(d => { if (d) { updateThreatMonitor(d); updateZoneBoundaries(d); } }).catch(() => {}),
        fetch('/api/seismicity/summary').then(r => r.ok ? r.json() : null).then(d => d && updateSeismicitySummary(d)).catch(() => {}),
        fetch('/api/dart/status').then(r => r.ok ? r.json() : null).then(d => { if (d) { updateDartMarkers(d); updateDartSummary(d); } }).catch(() => {}),
        fetch('/api/tidal/sensitivity').then(r => r.ok ? r.json() : null).then(d => d && updateTidalRipples(d)).catch(() => {}),
        fetch('/api/volcanic/activity').then(r => r.ok ? r.json() : null).then(d => { if (d) { updateVolcanicSummary(d); updateVolcanoMarkers(d); } }).catch(() => {}),
        fetch('/api/seedlink/stations').then(r => r.ok ? r.json() : null).then(d => { if (d) updateStationMarkers(d); }).catch(() => {}),
        fetch('/api/tier2-watch').then(r => r.ok ? r.json() : null).then(d => { if (d) updateSwarmWatch(d); }).catch(() => {}),
    ]);
    setInterval(refreshFaultRisk, CONFIG.FAULT_RISK_INTERVAL);
    setInterval(refreshHeatmap, CONFIG.HEATMAP_INTERVAL);
    setInterval(async () => {
        try {
            const resp = await fetch('/api/seismicity/summary');
            if (resp.ok) updateSeismicitySummary(await resp.json());
        } catch (e) {}
    }, 60000);
    setInterval(async () => {
        try {
            const resp = await fetch('/api/tier2-watch');
            if (resp.ok) updateSwarmWatch(await resp.json());
        } catch (e) {}
    }, 300000);

    const savedMag = localStorage.getItem('qw_filterMag');
    const savedHours = localStorage.getItem('qw_filterHours');
    if (savedMag) state.filterMag = parseFloat(savedMag);
    if (savedHours) state.filterHours = parseInt(savedHours);
    document.querySelectorAll('.eq-filter-btn[data-mag]').forEach(btn => {
        if (parseFloat(btn.dataset.mag) === state.filterMag) {
            document.querySelectorAll('.eq-filter-btn[data-mag]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        }
        btn.addEventListener('click', () => {
            document.querySelectorAll('.eq-filter-btn[data-mag]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.filterMag = parseFloat(btn.dataset.mag);
            localStorage.setItem('qw_filterMag', state.filterMag);
            if (state.earthquakes.length) updateEqFeed(state.earthquakes);
            applyMapFilter();
        });
    });
    document.querySelectorAll('.eq-filter-btn[data-hours]').forEach(btn => {
        if (parseInt(btn.dataset.hours) === state.filterHours) {
            document.querySelectorAll('.eq-filter-btn[data-hours]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        }
        btn.addEventListener('click', () => {
            document.querySelectorAll('.eq-filter-btn[data-hours]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.filterHours = parseInt(btn.dataset.hours);
            localStorage.setItem('qw_filterHours', state.filterHours);
            if (state.earthquakes.length) updateEqFeed(state.earthquakes);
            applyMapFilter();
        });
    });
}

// ── Ionosphere TEC overlay ───────────────────────────────
async function loadIonosphere() {
    if (!mapReady) return;
    try {
        const resp = await fetch('/api/ionosphere/tec');
        if (!resp.ok) return;
        const data = await resp.json();
        if (!data.features || !data.features.length) return;

        const timeEl = document.getElementById('iono-time');
        if (timeEl && data.features[0]?.properties?.time_tag) {
            const t = new Date(data.features[0].properties.time_tag);
            timeEl.textContent = t.toISOString().slice(11, 16) + ' UTC';
        } else if (timeEl && data.time_tag) {
            const t = new Date(data.time_tag);
            timeEl.textContent = t.toISOString().slice(11, 16) + ' UTC';
        }

        for (const f of data.features) {
            f.properties._weight = Math.min(1, (f.properties.tec || 0) / 50);
        }

        if (map.getSource('iono-tec')) {
            map.getSource('iono-tec').setData(data);
        } else {
            map.addSource('iono-tec', { type: 'geojson', data });
            map.addLayer({
                id: 'iono-heatmap',
                type: 'heatmap',
                source: 'iono-tec',
                maxzoom: 7,
                paint: {
                    'heatmap-weight': ['get', '_weight'],
                    'heatmap-intensity': ['interpolate', ['linear'], ['zoom'],
                        0, 0.3,
                        2, 0.35,
                        4, 0.4,
                        6, 0.45,
                    ],
                    'heatmap-radius': ['interpolate', ['linear'], ['zoom'],
                        0, 35,
                        2, 35,
                        4, 40,
                        6, 50,
                    ],
                    'heatmap-opacity': ['interpolate', ['linear'], ['zoom'],
                        0, 0.5,
                        3, 0.5,
                        6, 0.4,
                    ],
                    'heatmap-color': [
                        'interpolate', ['linear'], ['heatmap-density'],
                        0,    'rgba(0,0,0,0)',
                        0.15, 'rgba(30,60,140,0.3)',
                        0.35, 'rgba(40,120,200,0.45)',
                        0.55, 'rgba(50,180,150,0.5)',
                        0.75, 'rgba(180,200,50,0.6)',
                        0.9,  'rgba(225,140,30,0.7)',
                        1.0,  'rgba(205,50,30,0.75)',
                    ],
                },
                layout: { visibility: 'visible' },
            }, 'plate-boundaries');
        }
    } catch (e) {
        console.error('Ionosphere load:', e);
    }
}

// ── Weather Layers ───────────────────────────────────────
const WX_EVENT_COLORS = {
    'Tornado Warning':              { fill: 'rgba(255,60,120,0.12)',  line: 'rgba(255,60,120,0.9)',  pulse: true },
    'Tornado Watch':                { fill: 'rgba(255,60,120,0.06)',  line: 'rgba(255,60,120,0.4)',  pulse: false },
    'Severe Thunderstorm Warning':  { fill: 'rgba(255,120,170,0.10)', line: 'rgba(255,120,170,0.8)', pulse: true },
    'Severe Thunderstorm Watch':    { fill: 'rgba(255,120,170,0.05)', line: 'rgba(255,120,170,0.35)',pulse: false },
    'Hurricane Warning':            { fill: 'rgba(180,120,255,0.12)', line: 'rgba(180,120,255,0.9)', pulse: true },
    'Hurricane Watch':              { fill: 'rgba(180,120,255,0.06)', line: 'rgba(180,120,255,0.4)', pulse: false },
    'Tropical Storm Warning':       { fill: 'rgba(180,120,255,0.10)', line: 'rgba(180,120,255,0.7)', pulse: true },
    'Tropical Storm Watch':         { fill: 'rgba(180,120,255,0.05)', line: 'rgba(180,120,255,0.35)',pulse: false },
    'Flash Flood Warning':          { fill: 'rgba(90,200,255,0.10)',  line: 'rgba(90,200,255,0.8)',  pulse: true },
    'Flash Flood Watch':            { fill: 'rgba(90,200,255,0.05)',  line: 'rgba(90,200,255,0.35)', pulse: false },
    'Storm Surge Warning':          { fill: 'rgba(90,200,255,0.12)',  line: 'rgba(90,200,255,0.9)',  pulse: true },
    'Storm Surge Watch':            { fill: 'rgba(90,200,255,0.06)',  line: 'rgba(90,200,255,0.4)',  pulse: false },
};
const WX_DEFAULT_COLOR = { fill: 'rgba(255,120,170,0.08)', line: 'rgba(255,120,170,0.5)', pulse: false };

const SPC_RISK_COLORS = {
    'TSTM': { fill: 'rgba(80,160,80,0.05)',  line: 'rgba(80,160,80,0.3)' },
    'MRGL': { fill: 'rgba(80,160,80,0.07)',  line: 'rgba(80,160,80,0.45)' },
    'SLGT': { fill: 'rgba(220,170,0,0.08)',   line: 'rgba(220,170,0,0.5)' },
    'ENH':  { fill: 'rgba(230,130,30,0.09)',  line: 'rgba(230,130,30,0.55)' },
    'MDT':  { fill: 'rgba(220,60,40,0.10)',   line: 'rgba(220,60,40,0.6)' },
    'HIGH': { fill: 'rgba(255,60,120,0.12)',  line: 'rgba(255,60,120,0.7)' },
};

let _wxRadarFrame = 0;
let _wxRadarTimer = null;

async function loadWeatherWarnings() {
    if (!mapReady) return;
    try {
        const resp = await fetch('/api/weather/alerts');
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.error) return;
        const feats = data.features || [];

        const countEl = document.getElementById('wx-warn-count');
        if (countEl) countEl.textContent = feats.length || '';

        for (const f of feats) {
            const ev = f.properties.event || '';
            const c = WX_EVENT_COLORS[ev] || WX_DEFAULT_COLOR;
            f.properties._fill = c.fill;
            f.properties._line = c.line;
            f.properties._pulse = c.pulse ? 1 : 0;
            f.properties._dasharray = ev.includes('Watch') ? '6,4' : '0,0';
        }

        const geojson = { type: 'FeatureCollection', features: feats };
        if (map.getSource('wx-warnings')) {
            map.getSource('wx-warnings').setData(geojson);
        } else {
            map.addSource('wx-warnings', { type: 'geojson', data: geojson });
            map.addLayer({
                id: 'wx-warnings-fill', type: 'fill', source: 'wx-warnings',
                paint: { 'fill-color': ['get', '_fill'], 'fill-opacity': 1 },
            }, 'plate-boundaries');
            map.addLayer({
                id: 'wx-warnings-line', type: 'line', source: 'wx-warnings',
                paint: {
                    'line-color': ['get', '_line'],
                    'line-width': 1.5,
                    'line-opacity': ['case', ['==', ['get', '_pulse'], 1], 0.9, 0.5],
                },
                layout: { 'line-cap': 'round' },
            }, 'plate-boundaries');

            map.on('click', 'wx-warnings-fill', (e) => {
                if (e.originalEvent.target !== map.getCanvas()) return;
                const f = e.features[0];
                if (!f) return;
                const p = f.properties;
                const ev = p.event || 'Weather Alert';
                const c = WX_EVENT_COLORS[ev] || WX_DEFAULT_COLOR;
                const pillColor = c.line;
                new maplibregl.Popup({ className: 'wx-popup', maxWidth: '320px' })
                    .setLngLat(e.lngLat)
                    .setHTML(`
                        <div style="font-family:Inter,sans-serif;color:rgba(255,255,255,0.9)">
                            <div style="display:inline-block;padding:2px 8px;border-radius:3px;background:${pillColor};font-size:11px;font-weight:600;letter-spacing:0.5px;margin-bottom:6px">${ev.toUpperCase()}</div>
                            <div style="font-size:12px;line-height:1.4;margin-top:4px;color:rgba(255,255,255,0.75)">${p.headline || ''}</div>
                            <div style="font-size:11px;margin-top:4px;color:rgba(255,255,255,0.5)">${p.areaDesc || ''}</div>
                        </div>
                    `)
                    .addTo(map);
            });
            map.on('mouseenter', 'wx-warnings-fill', () => { map.getCanvas().style.cursor = 'pointer'; });
            map.on('mouseleave', 'wx-warnings-fill', () => { map.getCanvas().style.cursor = ''; });
        }
    } catch (e) { console.error('Weather warnings:', e); }
}

async function loadWeatherRadar() {
    if (!mapReady) return;
    try {
        const resp = await fetch('/api/weather/radar');
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.error || !data.radar) return;

        const frames = data.radar.past || [];
        if (!frames.length) return;
        const host = data.host;

        if (map.getSource('wx-radar-0')) {
            return;
        }

        // last ~8 frames: snappier loop, fewer layers. 512px tiles + nearest resampling
        // = crisp (not blurry). Inactive frames stay at 0.01 opacity (not 0) so MapLibre
        // never evicts their tiles — kills the "disappears for a few seconds" reload gap.
        const recent = frames.slice(-8);
        const ACTIVE = 0.6, KEEP = 0.01;
        recent.forEach((frame, i) => {
            const url = `${host}${frame.path}/512/{z}/{x}/{y}/8/1_1.png`;
            map.addSource(`wx-radar-${i}`, { type: 'raster', tiles: [url], tileSize: 512 });
            map.addLayer({
                id: `wx-radar-${i}`, type: 'raster', source: `wx-radar-${i}`,
                paint: {
                    'raster-opacity': i === recent.length - 1 ? ACTIVE : KEEP,
                    'raster-opacity-transition': { duration: 200 },  // smooth crossfade
                    'raster-resampling': 'nearest',                  // crisp, not blurry
                    'raster-fade-duration': 0,                       // no tile-load fade flicker
                },
                layout: { visibility: 'visible' },
            }, 'plate-boundaries');
        });

        _wxRadarFrame = recent.length - 1;
        _wxRadarTimer = setInterval(() => {
            const prev = _wxRadarFrame;
            _wxRadarFrame = (_wxRadarFrame + 1) % recent.length;
            try {
                map.setPaintProperty(`wx-radar-${prev}`, 'raster-opacity', KEEP);
                map.setPaintProperty(`wx-radar-${_wxRadarFrame}`, 'raster-opacity', ACTIVE);
            } catch(e) {}
        }, 650);
    } catch (e) { console.error('Weather radar:', e); }
}

function stopWeatherRadar() {
    if (_wxRadarTimer) { clearInterval(_wxRadarTimer); _wxRadarTimer = null; }
}

async function loadWeatherOutlook() {
    if (!mapReady) return;
    try {
        const resp = await fetch('/api/weather/outlook');
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.error) return;
        const feats = (data.features || []).filter(f => f.geometry);

        for (const f of feats) {
            const label = f.properties.LABEL || '';
            const c = SPC_RISK_COLORS[label] || SPC_RISK_COLORS['TSTM'];
            f.properties._fill = c.fill;
            f.properties._line = c.line;
        }

        const geojson = { type: 'FeatureCollection', features: feats };
        if (map.getSource('wx-outlook')) {
            map.getSource('wx-outlook').setData(geojson);
        } else {
            map.addSource('wx-outlook', { type: 'geojson', data: geojson });
            map.addLayer({
                id: 'wx-outlook-fill', type: 'fill', source: 'wx-outlook',
                paint: { 'fill-color': ['get', '_fill'], 'fill-opacity': 1 },
            }, 'plate-boundaries');
            map.addLayer({
                id: 'wx-outlook-line', type: 'line', source: 'wx-outlook',
                paint: { 'line-color': ['get', '_line'], 'line-width': 1.2, 'line-opacity': 0.7 },
                layout: { 'line-cap': 'round' },
            }, 'plate-boundaries');

            map.on('click', 'wx-outlook-fill', (e) => {
                if (e.originalEvent.target !== map.getCanvas()) return;
                const f = e.features[0];
                if (!f) return;
                const p = f.properties;
                const label = p.LABEL2 || p.LABEL || 'Outlook';
                const c = SPC_RISK_COLORS[p.LABEL] || SPC_RISK_COLORS['TSTM'];
                new maplibregl.Popup({ className: 'wx-popup', maxWidth: '280px' })
                    .setLngLat(e.lngLat)
                    .setHTML(`
                        <div style="font-family:Inter,sans-serif;color:rgba(255,255,255,0.9)">
                            <div style="display:inline-block;padding:2px 8px;border-radius:3px;background:${c.line};font-size:11px;font-weight:600;letter-spacing:0.5px">${label.toUpperCase()}</div>
                            <div style="font-size:11px;margin-top:6px;color:rgba(255,255,255,0.5)">SPC Day 1 Convective Outlook</div>
                        </div>
                    `)
                    .addTo(map);
            });
        }
    } catch (e) { console.error('Weather outlook:', e); }
}

async function loadWeatherHurricanes() {
    if (!mapReady) return;
    try {
        const resp = await fetch('/api/weather/hurricanes');
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.error) return;
        const storms = data.activeStorms || [];

        const countEl = document.getElementById('wx-hurricane-count');
        if (countEl) countEl.textContent = storms.length || '';

        for (const key of Object.keys(state._hurricaneMarkers || {})) {
            state._hurricaneMarkers[key].remove();
        }
        state._hurricaneMarkers = {};

        for (const storm of storms) {
            if (!storm.latitudeNumeric || !storm.longitudeNumeric) continue;
            const el = document.createElement('div');
            el.className = 'wx-hurricane-marker';
            el.innerHTML = `<div class="wx-hurricane-glyph"></div><div class="wx-hurricane-label">${storm.name || ''}</div>`;
            const marker = new maplibregl.Marker({ element: el })
                .setLngLat([storm.longitudeNumeric, storm.latitudeNumeric])
                .addTo(map);
            state._hurricaneMarkers[storm.id || storm.name] = marker;
        }
    } catch (e) { console.error('Weather hurricanes:', e); }
}

state._hurricaneMarkers = {};

// ── Map Legend ────────────────────────────────────────────
(function initLegend() {
    const legendEl = document.getElementById('map-legend');
    const toggle = document.getElementById('legend-toggle');
    const close = document.getElementById('legend-close');
    const panel = document.getElementById('legend-panel');
    if (!legendEl) return;

    toggle.addEventListener('click', () => legendEl.classList.add('open'));
    close.addEventListener('click', () => legendEl.classList.remove('open'));

    // Close on outside click
    document.addEventListener('click', (e) => {
        if (legendEl.classList.contains('open') && !legendEl.contains(e.target)) {
            legendEl.classList.remove('open');
        }
    });

    // Layer visibility state
    const layerState = {
        earthquakes: true,
        heatmap: true,
        plates: true,
        faults: true,
        stations: true,
        dart: true,
        tidal: true,
        'volc-high': false,
        'volc-elevated': false,
        'volc-active': false,
        ionosphere: false,
        'wx-radar': false,
        'wx-warnings': false,
        'wx-outlook': false,
        'wx-hurricanes': false,
    };

    function setMarkerVisibility(markers, visible) {
        for (const key of Object.keys(markers)) {
            const m = markers[key];
            const mk = m && m.marker ? m.marker : m;
            if (!mk || !mk.getElement) continue;
            mk.getElement().style.display = visible ? '' : 'none';
        }
    }

    function applyLayer(layer, on) {
        layerState[layer] = on;

        if (!mapReady) return;

        switch (layer) {
            case 'earthquakes':
                setMarkerVisibility(state.eqMarkers, on);
                break;
            case 'heatmap':
                try { map.setLayoutProperty('eq-heatmap', 'visibility', on ? 'visible' : 'none'); } catch(e) {}
                break;
            case 'plates':
                try { map.setLayoutProperty('plate-boundaries', 'visibility', on ? 'visible' : 'none'); } catch(e) {}
                break;
            case 'faults':
                try {
                    map.setLayoutProperty('fault-heatmap', 'visibility', on ? 'visible' : 'none');
                    map.setLayoutProperty('fault-lines-glow', 'visibility', on ? 'visible' : 'none');
                    map.setLayoutProperty('fault-lines-core', 'visibility', on ? 'visible' : 'none');
                } catch(e) {}
                break;
            case 'stations':
                setMarkerVisibility(state.stationMarkers, on);
                break;
            case 'dart':
                setMarkerVisibility(state.dartMarkers, on);
                break;
            case 'tidal':
                setMarkerVisibility(state.tidalMarkers, on);
                break;
            case 'ionosphere':
                try {
                    map.setLayoutProperty('iono-heatmap', 'visibility', on ? 'visible' : 'none');
                } catch(e) {}
                if (on && !map.getSource('iono-tec')) loadIonosphere();
                break;
            case 'wx-warnings':
                if (on && !map.getSource('wx-warnings')) { loadWeatherWarnings(); }
                try {
                    map.setLayoutProperty('wx-warnings-fill', 'visibility', on ? 'visible' : 'none');
                    map.setLayoutProperty('wx-warnings-line', 'visibility', on ? 'visible' : 'none');
                } catch(e) {}
                break;
            case 'wx-radar':
                if (on && !map.getSource('wx-radar-0')) { loadWeatherRadar(); }
                else if (!on) { stopWeatherRadar(); }
                if (map.getSource('wx-radar-0')) {
                    let i = 0;
                    while (map.getSource(`wx-radar-${i}`)) {
                        try { map.setLayoutProperty(`wx-radar-${i}`, 'visibility', on ? 'visible' : 'none'); } catch(e) {}
                        i++;
                    }
                    if (on && !_wxRadarTimer) loadWeatherRadar();
                }
                break;
            case 'wx-outlook':
                if (on && !map.getSource('wx-outlook')) { loadWeatherOutlook(); }
                try {
                    map.setLayoutProperty('wx-outlook-fill', 'visibility', on ? 'visible' : 'none');
                    map.setLayoutProperty('wx-outlook-line', 'visibility', on ? 'visible' : 'none');
                } catch(e) {}
                break;
            case 'wx-hurricanes':
                if (on) { loadWeatherHurricanes(); }
                else {
                    for (const key of Object.keys(state._hurricaneMarkers || {})) {
                        state._hurricaneMarkers[key].remove();
                    }
                    state._hurricaneMarkers = {};
                }
                break;
            case 'volc-high':
            case 'volc-elevated':
            case 'volc-active':
                _applyVolcanicFilter();
                break;
        }
    }

    function _applyVolcanicFilter() {
        const data = state._volcanicData;
        if (!data || !data.volcanoes) return;

        const levelMap = {
            'volc-high': 'high',
            'volc-elevated': 'elevated',
            'volc-active': 'active',
        };

        const enabledLevels = new Set();
        for (const [k, v] of Object.entries(levelMap)) {
            if (layerState[k]) enabledLevels.add(v);
        }

        // Remove markers for levels that are off
        for (const key of Object.keys(state.volcanoMarkers)) {
            const v = data.volcanoes.find(x => String(x.id) === key);
            if (!v || !enabledLevels.has(v.level)) {
                _hideVolcanoMarker(key);
            }
        }

        // Add markers for levels that are on
        for (const v of data.volcanoes) {
            if (enabledLevels.has(v.level)) {
                _showVolcanoMarker(v.id);
            }
        }
    }

    // Wire up toggle switches
    panel.querySelectorAll('.legend-item').forEach(item => {
        const layer = item.dataset.layer;
        const cb = item.querySelector('input[type="checkbox"]');
        if (!layer || !cb) return;

        cb.addEventListener('change', () => {
            applyLayer(layer, cb.checked);
        });
    });

    // Expose so volcanic data updates can refresh counts
    window._updateLegendVolcanicCounts = function(data) {
        if (!data || !data.volcanoes) return;
        const high = data.volcanoes.filter(v => v.level === 'high').length;
        const elevated = data.volcanoes.filter(v => v.level === 'elevated').length;
        const active = data.volcanoes.filter(v => v.level === 'active').length;
        const hEl = document.getElementById('volc-high-count');
        const eEl = document.getElementById('volc-elevated-count');
        const aEl = document.getElementById('volc-active-count');
        if (hEl) hEl.textContent = high;
        if (eEl) eEl.textContent = elevated;
        if (aEl) aEl.textContent = active;
    };

    // Expose layer state check for marker update functions
    window._legendLayerOn = function(layer) {
        return layerState[layer] !== false;
    };
})();

init();
