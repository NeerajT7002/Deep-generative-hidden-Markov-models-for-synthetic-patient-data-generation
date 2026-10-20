// ========================================================================================
// DG-HMM NEURAL ICU COMMAND CENTER: NEXT-GEN INTERACTIVE SUITE
// Group 6 | Probabilistic Reasoning (Semester 5)
// Momenzadeh & Oshaghi (2026), BMC Medical Informatics & Decision Making
// ========================================================================================

let currentSyntheticData = null;
let audioEnabled = false;
let audioCtx = null;

// Charts
let chartHemodynamics = null;
let chartMetabolic = null;
let chartRadar = null;

const STATE_NAMES = ['Stable', 'Moderate', 'Guarded', 'Critical'];
const STATE_COLORS = ['#059669', '#0284c7', '#d97706', '#dc2626'];

document.addEventListener('DOMContentLoaded', () => {
    initAudio();
    initParticleBackground();
    initECGMonitor();
    initTabs();
    initCharts();
    initRadarChart();
    initManifoldTopology();
    loadBenchmarkMetrics();
    initInlineDatasetViewer();
    generateSyntheticPatient();

    // Event Listeners
    document.getElementById('btnGenerate').addEventListener('click', () => {
        playBeep(880, 'triangle', 0.08);
        generateSyntheticPatient();
    });
    document.getElementById('btnOverlayReal').addEventListener('click', () => {
        playBeep(660, 'sine', 0.06);
        overlayRealPatient();
    });
    document.getElementById('btnDownloadCSV').addEventListener('click', downloadPatientCSV);
    const btnReplay = document.getElementById('btnReplayPath');
    if (btnReplay) {
        btnReplay.addEventListener('click', () => {
            playBeep(640, 'triangle', 0.08);
            animatePatientTrajectory();
        });
    }

    // Audio Toggle
    const audioBtn = document.getElementById('btnAudioToggle');
    audioBtn.addEventListener('click', () => {
        audioEnabled = !audioEnabled;
        if (audioEnabled && !audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        audioBtn.classList.toggle('active', audioEnabled);
        document.getElementById('audioIcon').innerText = audioEnabled ? '🔊' : '🔇';
        document.getElementById('audioLabel').innerText = audioEnabled ? 'AUDIO ON' : 'AUDIO OFF';
        if (audioEnabled) playBeep(784, 'sine', 0.1);
    });

    // Preset buttons
    document.querySelectorAll('.preset-pill').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.preset-pill').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            playBeep(600, 'sine', 0.05);
            applyPreset(btn.dataset.preset);
        });
    });
});

// =========================================================================
// 1. WEB AUDIO API SYNTHESIZER
// =========================================================================
function initAudio() {
    // Lazy initialized on first user interaction
}

function playBeep(freq = 440, type = 'sine', duration = 0.08) {
    if (!audioEnabled) return;
    try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
        // Audio policy
    }
}

// =========================================================================
// 2. 60 FPS CRT HOSPITAL ECG WAVEFORM MONITOR (LEAD II)
// =========================================================================
function initECGMonitor() {
    const canvas = document.getElementById('ecgWaveCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function resize() {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    let sweepX = 0;
    const speed = 2.4;
    const points = [];
    const maxPoints = 500;

    function getECGHeight(t) {
        const cycle = t % 120;
        // P-wave
        if (cycle > 20 && cycle < 35) return Math.sin((cycle - 20) / 15 * Math.PI) * 7;
        // QRS complex
        if (cycle > 45 && cycle < 48) return -6;
        if (cycle >= 48 && cycle < 53) return 40; // R spike
        if (cycle >= 53 && cycle < 58) return -12; // S trough
        // T-wave
        if (cycle > 70 && cycle < 95) return Math.sin((cycle - 70) / 25 * Math.PI) * 11;
        return (Math.random() - 0.5) * 1.5; // Baseline noise
    }

    let t = 0;
    function animate() {
        t += 1;
        sweepX = (sweepX + speed) % canvas.width;

        const cy = canvas.height / 2;
        const val = cy - getECGHeight(t);
        points.push({ x: sweepX, y: val });
        if (points.length > maxPoints) points.shift();

        // Clear trailing fade
        ctx.fillStyle = 'rgba(4, 8, 14, 0.12)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw green phosphor ECG beam
        ctx.beginPath();
        ctx.strokeStyle = '#00e676';
        ctx.lineWidth = 2.2;
        ctx.shadowBlur = 12;
        ctx.shadowColor = '#00e676';

        for (let i = 0; i < points.length - 1; i++) {
            const p1 = points[i];
            const p2 = points[i + 1];
            if (Math.abs(p2.x - p1.x) < 50) {
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
            }
        }
        ctx.stroke();

        // Scanning vertical beam
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(0, 230, 118, 0.4)';
        ctx.lineWidth = 2;
        ctx.moveTo(sweepX, 0);
        ctx.lineTo(sweepX, canvas.height);
        ctx.stroke();

        requestAnimationFrame(animate);
    }
    animate();
}

// =========================================================================
// 3. AMBIENT PARTICLES
// =========================================================================
function initParticleBackground() {
    const canvas = document.getElementById('bgParticleCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    const particles = Array.from({ length: 45 }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        radius: Math.random() * 2 + 1,
        color: Math.random() > 0.5 ? 'rgba(0, 242, 254, ' : 'rgba(168, 85, 247, '
    }));

    function loop() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = p.color + '0.6)';
            ctx.fill();

            // Connect nearby
            for (let j = i + 1; j < particles.length; j++) {
                const p2 = particles[j];
                const dist = Math.hypot(p2.x - p.x, p2.y - p.y);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.strokeStyle = `rgba(0, 242, 254, ${0.15 * (1 - dist / 120)})`;
                    ctx.lineWidth = 0.8;
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(loop);
    }
    loop();
}

// =========================================================================
// 4. TAB NAVIGATION
// =========================================================================
function initTabs() {
    const buttons = document.querySelectorAll('.nav-tab-item');
    const panes = document.querySelectorAll('.tab-pane');

    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            playBeep(700, 'sine', 0.04);
            buttons.forEach(b => b.classList.remove('active'));
            panes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const target = btn.dataset.tab;
            const pane = document.getElementById(target);
            if (pane) pane.classList.add('active');

            if (target === 'manifoldTab') {
                setTimeout(() => {
                    renderManifoldTrajectory(currentSyntheticData ? currentSyntheticData.records : null);
                    animatePatientTrajectory();
                }, 60);
            }
        });
    });
}

// =========================================================================
// 5. CHART.JS INITIALIZATION
// =========================================================================
function initCharts() {
    const hours = Array.from({ length: 24 }, (_, i) => `H${i + 1}`);

    const baseOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: { labels: { color: '#94a3b8', font: { family: 'Outfit', size: 11, weight: '600' } } },
            tooltip: { backgroundColor: 'rgba(6, 9, 16, 0.95)', borderColor: 'rgba(0, 242, 254, 0.3)', borderWidth: 1, titleColor: '#00f2fe', bodyFont: { family: 'JetBrains Mono' } }
        },
        scales: {
            x: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } } },
            y: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } } }
        }
    };

    // Hemodynamics
    const ctxHemo = document.getElementById('chartHemodynamics').getContext('2d');
    chartHemodynamics = new Chart(ctxHemo, {
        type: 'line',
        data: {
            labels: hours,
            datasets: [
                { label: 'Heart Rate (bpm)', borderColor: '#00f2fe', backgroundColor: 'rgba(0, 242, 254, 0.08)', fill: true, borderWidth: 2.4, tension: 0.35, data: [] },
                { label: 'Systolic BP (mmHg)', borderColor: '#a855f7', backgroundColor: 'transparent', borderWidth: 2.4, tension: 0.35, data: [] },
                { label: 'Diastolic BP (mmHg)', borderColor: '#f43f5e', backgroundColor: 'transparent', borderWidth: 2.0, borderDash: [4, 4], tension: 0.35, data: [] }
            ]
        },
        options: baseOptions
    });

    // Metabolic
    const ctxMeta = document.getElementById('chartMetabolic').getContext('2d');
    chartMetabolic = new Chart(ctxMeta, {
        type: 'line',
        data: {
            labels: hours,
            datasets: [
                { label: 'SpO2 Oxygenation (%)', borderColor: '#00e676', backgroundColor: 'rgba(0, 230, 118, 0.08)', fill: true, borderWidth: 2.4, tension: 0.35, data: [] },
                { label: 'Respiratory Rate', borderColor: '#f59e0b', backgroundColor: 'transparent', borderWidth: 2.0, tension: 0.35, data: [] },
                { label: 'Temperature (°F)', borderColor: '#ec4899', backgroundColor: 'transparent', borderWidth: 1.8, borderDash: [3, 3], tension: 0.35, data: [] }
            ]
        },
        options: baseOptions
    });
}

function initRadarChart() {
    const ctxRadar = document.getElementById('chartRadar').getContext('2d');
    chartRadar = new Chart(ctxRadar, {
        type: 'radar',
        data: {
            labels: ['Heart Rate', 'Systolic BP', 'Diastolic BP', 'Pulse Press.', 'SpO2', 'Resp Rate'],
            datasets: [{
                label: 'Mean Balance Score',
                data: [80, 75, 70, 85, 95, 65],
                backgroundColor: 'rgba(0, 242, 254, 0.2)',
                borderColor: '#00f2fe',
                borderWidth: 2,
                pointBackgroundColor: '#00e676',
                pointRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.06)' },
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    pointLabels: { color: '#94a3b8', font: { family: 'Outfit', size: 10 } },
                    ticks: { display: false }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// =========================================================================
// 6. SYNTHETIC PATIENT GENERATION
// =========================================================================
async function generateSyntheticPatient() {
    const btn = document.getElementById('btnGenerate');
    btn.disabled = true;
    btn.innerHTML = 'Synthesizing De Novo...';

    try {
        const res = await fetch('/api/generate_patient', { method: 'POST' });
        const data = await res.json();
        currentSyntheticData = data.records;

        const hr = data.records.map(r => r['Heart Rate']);
        const sbp = data.records.map(r => r['Systolic BP']);
        const dbp = data.records.map(r => r['Diastolic BP']);
        const spo2 = data.records.map(r => r['SpO2']);
        const rr = data.records.map(r => r['Respiratory Rate']);
        const temp = data.records.map(r => r['Temperature']);

        chartHemodynamics.data.datasets[0].data = hr;
        chartHemodynamics.data.datasets[1].data = sbp;
        chartHemodynamics.data.datasets[2].data = dbp;
        if (chartHemodynamics.data.datasets.length > 3) chartHemodynamics.data.datasets = chartHemodynamics.data.datasets.slice(0, 3);
        chartHemodynamics.update();

        chartMetabolic.data.datasets[0].data = spo2;
        chartMetabolic.data.datasets[1].data = rr;
        chartMetabolic.data.datasets[2].data = temp;
        chartMetabolic.update();

        // Update CRT Monitor Top Readout
        document.getElementById('crtHeartRate').innerText = Math.round(hr[0]);
        document.getElementById('crtBP').innerText = `${Math.round(sbp[0])}/${Math.round(dbp[0])}`;
        document.getElementById('crtSpO2').innerText = `${spo2[0].toFixed(1)}%`;
        document.getElementById('crtRR').innerText = rr[0].toFixed(1);

        // Update Bio Panel
        const randId = Math.floor(Math.random() * 900) + 100;
        document.getElementById('bioPatientID').innerText = `Synthetic Patient #${randId}`;
        document.getElementById('samplePathSummary').innerText = data.transition_summary;

        const meanPP = (data.records.reduce((a, b) => a + b['Pulse Pressure'], 0) / 24).toFixed(1);
        const meanShock = (data.records.reduce((a, b) => a + b['Shock Index'], 0) / 24).toFixed(2);
        document.getElementById('valPulsePressure').innerText = meanPP;
        document.getElementById('valShockIndex').innerText = meanShock;

        // Radar values
        chartRadar.data.datasets[0].data = [
            Math.min(100, (hr[0] / 100) * 80),
            Math.min(100, (sbp[0] / 130) * 80),
            Math.min(100, (dbp[0] / 80) * 80),
            Math.min(100, (meanPP / 60) * 80),
            spo2[0],
            Math.min(100, (rr[0] / 20) * 80)
        ];
        chartRadar.update();

        // Manifold
        renderManifoldTrajectory(data.records);

    } catch (err) {
        console.error('Generation error:', err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-sparkle">✦</span> <span>SYNTHESIZE NEW PATIENT</span>';
    }
}

// Overlay Real ICU Patient
async function overlayRealPatient() {
    try {
        const res = await fetch('/api/real_sample');
        const data = await res.json();
        const realHR = data.records.map(r => r['Heart Rate']);
        const realSBP = data.records.map(r => r['Systolic BP']);

        if (chartHemodynamics.data.datasets.length === 3) {
            chartHemodynamics.data.datasets.push({
                label: `Real Patient #${data.patient_index} HR`,
                borderColor: 'rgba(255, 255, 255, 0.45)',
                backgroundColor: 'transparent',
                borderWidth: 1.6,
                borderDash: [5, 5],
                data: realHR
            });
            chartHemodynamics.data.datasets.push({
                label: `Real Patient #${data.patient_index} SBP`,
                borderColor: 'rgba(216, 180, 254, 0.45)',
                backgroundColor: 'transparent',
                borderWidth: 1.6,
                borderDash: [5, 5],
                data: realSBP
            });
        } else {
            chartHemodynamics.data.datasets[3].data = realHR;
            chartHemodynamics.data.datasets[4].data = realSBP;
            chartHemodynamics.data.datasets[3].label = `Real Patient #${data.patient_index} HR`;
        }
        chartHemodynamics.update();
    } catch (e) {
        console.error(e);
    }
}


// =========================================================================
// 8. 2D CONTINUOUS LATENT MANIFOLD & MARKOV TOPOLOGY
// =========================================================================
const CLINICAL_TRANSITIONS = {
    '0_0': {
        from: 'Stable (State 0)', to: 'Stable (State 0)', prob: '72.0%',
        desc: 'High clinical persistence. A stable patient has a 72% likelihood of maintaining normal hemodynamics and metabolic stability in the subsequent hour.'
    },
    '0_1': {
        from: 'Stable (State 0)', to: 'Moderate (State 1)', prob: '18.0%',
        desc: 'Early physiological stress. Patient begins showing mild tachycardia, compensatory hyperventilation, or early metabolic distress.'
    },
    '0_2': {
        from: 'Stable (State 0)', to: 'Guarded (State 2)', prob: '8.0%',
        desc: 'Accelerated deterioration. Patient develops trending hypotension and signs of systemic inflammatory response (SIRS).'
    },
    '0_3': {
        from: 'Stable (State 0)', to: 'Critical (State 3)', prob: '2.0%',
        desc: 'Acute catastrophic crash. Sudden severe collapse (e.g. massive pulmonary embolism, acute myocardial infarction, or sudden airway compromise).'
    },
    '1_0': {
        from: 'Moderate (State 1)', to: 'Stable (State 0)', prob: '15.0%',
        desc: 'Positive treatment response. Oxygen titration or fluid resuscitation stabilizes vital signs back to normal physiological baseline.'
    },
    '1_1': {
        from: 'Moderate (State 1)', to: 'Moderate (State 1)', prob: '65.0%',
        desc: 'Compensated equilibrium. Patient remains under active monitoring with mild vital sign abnormalities.'
    },
    '1_2': {
        from: 'Moderate (State 1)', to: 'Guarded (State 2)', prob: '16.0%',
        desc: 'Disease progression. Patient requires escalating medical therapy, such as initiating low-dose vasopressors or telemetry expansion.'
    },
    '1_3': {
        from: 'Moderate (State 1)', to: 'Critical (State 3)', prob: '4.0%',
        desc: 'Rapid decompensation. Patient fails to compensate, showing sharp drop in oxygenation or blood pressure.'
    },
    '2_0': {
        from: 'Guarded (State 2)', to: 'Stable (State 0)', prob: '8.0%',
        desc: 'Rapid reversal. Effective clinical resuscitation rapidly restores hemodynamics without sustained organ strain.'
    },
    '2_1': {
        from: 'Guarded (State 2)', to: 'Moderate (State 1)', prob: '19.0%',
        desc: 'Gradual recovery. Mean arterial pressure (MAP) stabilizes and vasopressor requirements decrease.'
    },
    '2_2': {
        from: 'Guarded (State 2)', to: 'Guarded (State 2)', prob: '58.0%',
        desc: 'Protracted intensive care. Patient remains guarded under continuous vasoactive infusion and close surveillance.'
    },
    '2_3': {
        from: 'Guarded (State 2)', to: 'Critical (State 3)', prob: '28.0%',
        desc: 'High deterioration alert! A guarded patient has a 28% risk of plunging into refractory septic shock in the next hour unless aggressive clinical intervention is performed.'
    },
    '3_0': {
        from: 'Critical (State 3)', to: 'Stable (State 0)', prob: '2.0%',
        desc: 'Emergency rescue. Rare immediate stabilization following invasive resuscitation or surgical intervention.'
    },
    '3_1': {
        from: 'Critical (State 3)', to: 'Moderate (State 1)', prob: '5.0%',
        desc: 'Post-resuscitation rebound. Emergency measures successfully lift patient out of critical shock into moderate observation.'
    },
    '3_2': {
        from: 'Critical (State 3)', to: 'Guarded (State 2)', prob: '28.0%',
        desc: 'Successful stabilization. Surviving Sepsis bundle protocol lifts patient from imminent collapse to guarded status.'
    },
    '3_3': {
        from: 'Critical (State 3)', to: 'Critical (State 3)', prob: '65.0%',
        desc: 'Refractory shock persistence. Patient requires maximum mechanical and vasopressor support to maintain vital perfusion.'
    }
};

let currentManifoldPoints = [];
let manifoldAnimTimer = null;

function initManifoldTopology() {
    renderMarkovMatrix();
    setTimeout(() => {
        renderManifoldTrajectory();
    }, 120);
}

function renderMarkovMatrix() {
    const box = document.getElementById('matrixGridBox');
    if (!box) return;
    box.innerHTML = '';

    const rowLabels = ['S0 (Stable)', 'S1 (Mod)', 'S2 (Grd)', 'S3 (Crit)'];
    const rowClasses = ['s0', 's1', 's2', 's3'];

    const matrix = [
        [0.72, 0.18, 0.08, 0.02],
        [0.15, 0.65, 0.16, 0.04],
        [0.08, 0.19, 0.58, 0.15],
        [0.02, 0.05, 0.28, 0.65]
    ];

    for (let i = 0; i < 4; i++) {
        // Row Label
        const rLabel = document.createElement('div');
        rLabel.className = `matrix-row-label ${rowClasses[i]}`;
        rLabel.innerText = rowLabels[i];
        box.appendChild(rLabel);

        // 4 Columns
        for (let j = 0; j < 4; j++) {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            const val = matrix[i][j];
            cell.innerText = (val * 100).toFixed(0) + '%';
            cell.dataset.row = i;
            cell.dataset.col = j;

            if (i === j) {
                cell.classList.add('diagonal-cell');
            }

            const updateInspector = () => {
                document.querySelectorAll('.matrix-cell').forEach(c => c.classList.remove('active-cell'));
                cell.classList.add('active-cell');

                const key = `${i}_${j}`;
                const info = CLINICAL_TRANSITIONS[key];
                if (info) {
                    const tiBadge = document.getElementById('tiBadge');
                    const tiProb = document.getElementById('tiProb');
                    const tiTrans = document.getElementById('tiTransition');
                    const tiDesc = document.getElementById('tiDesc');
                    if (tiBadge) tiBadge.innerText = `ROW ${i} → COL ${j}`;
                    if (tiProb) tiProb.innerText = `${info.prob} Probability`;
                    if (tiTrans) tiTrans.innerText = `${info.from} → ${info.to}`;
                    if (tiDesc) tiDesc.innerText = info.desc;
                }
            };

            cell.addEventListener('mouseenter', () => {
                playBeep(420 + i * 70 + j * 40, 'sine', 0.03);
                updateInspector();
            });

            cell.addEventListener('click', updateInspector);

            // Default highlight: Row 2, Col 3 (Guarded -> Critical 28%)
            if (i === 2 && j === 3) {
                cell.classList.add('active-cell');
                updateInspector();
            }

            box.appendChild(cell);
        }
    }
}

function renderManifoldTrajectory(records = null) {
    const canvas = document.getElementById('manifoldCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const container = canvas.parentElement;
    const w = container && container.clientWidth > 100 ? container.clientWidth : 800;
    const h = container && container.clientHeight > 100 ? container.clientHeight : 480;
    canvas.width = w;
    canvas.height = h;

    ctx.clearRect(0, 0, w, h);

    // 1. Subtle Background Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = 0; x < w; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }
    for (let y = 0; y < h; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Principal Clinical Coordinate Axes (Crosshairs)
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.12)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    // Horizontal Axis
    ctx.beginPath();
    ctx.moveTo(40, h * 0.52);
    ctx.lineTo(w - 40, h * 0.52);
    ctx.stroke();
    // Vertical Axis
    ctx.beginPath();
    ctx.moveTo(w * 0.5, 30);
    ctx.lineTo(w * 0.5, h - 30);
    ctx.stroke();
    ctx.setLineDash([]);

    // Axis Labels
    ctx.fillStyle = 'rgba(148, 163, 184, 0.55)';
    ctx.font = '500 11px Outfit, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('← Hemodynamic Stability', 45, h * 0.52 - 8);
    ctx.textAlign = 'right';
    ctx.fillText('Circulatory Decompensation →', w - 45, h * 0.52 - 8);

    ctx.textAlign = 'center';
    ctx.fillText('↑ Normal Physiological Baseline', w * 0.5, 24);
    ctx.fillText('↓ Severe Metabolic & Respiratory Strain', w * 0.5, h - 12);

    // 2. 4 Clinical Cluster Halos in Latent Space
    const clusters = [
        { 
            id: 0, 
            name: 'State 0: Stable', 
            vitals: 'HR 65-85 | SBP 120 | SpO₂ 99%',
            x: w * 0.28, y: h * 0.30, 
            color: 'rgba(0, 230, 118, ', 
            hex: '#00e676' 
        },
        { 
            id: 1, 
            name: 'State 1: Moderate', 
            vitals: 'HR 85-105 | SBP 110 | SpO₂ 94%',
            x: w * 0.72, y: h * 0.30, 
            color: 'rgba(0, 242, 254, ', 
            hex: '#00f2fe' 
        },
        { 
            id: 2, 
            name: 'State 2: Guarded', 
            vitals: 'HR 105-125 | SBP 95 | SpO₂ 91%',
            x: w * 0.70, y: h * 0.72, 
            color: 'rgba(245, 158, 11, ', 
            hex: '#f59e0b' 
        },
        { 
            id: 3, 
            name: 'State 3: Critical', 
            vitals: 'HR > 130 | SBP 75 | SpO₂ 85%',
            x: w * 0.28, y: h * 0.72, 
            color: 'rgba(244, 63, 94, ', 
            hex: '#f43f5e' 
        }
    ];

    // Subtle Directed Disease Progression Arrows between clusters
    const drawDirectedArrow = (x1, y1, x2, y2, color, dashed = false) => {
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1.5;
        if (dashed) ctx.setLineDash([4, 4]);
        else ctx.setLineDash([]);
        
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        const angle = Math.atan2(y2 - y1, x2 - x1);
        const arrowLen = 8;
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - arrowLen * Math.cos(angle - Math.PI / 6), y2 - arrowLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(x2 - arrowLen * Math.cos(angle + Math.PI / 6), y2 - arrowLen * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
        ctx.setLineDash([]);
    };

    // Progression tendencies
    drawDirectedArrow(clusters[0].x + 40, clusters[0].y, clusters[1].x - 40, clusters[1].y, 'rgba(0, 242, 254, 0.2)');
    drawDirectedArrow(clusters[1].x, clusters[1].y + 35, clusters[2].x, clusters[2].y - 35, 'rgba(245, 158, 11, 0.25)');
    drawDirectedArrow(clusters[2].x - 40, clusters[2].y, clusters[3].x + 40, clusters[3].y, 'rgba(244, 63, 94, 0.35)', true);
    drawDirectedArrow(clusters[3].x, clusters[3].y - 35, clusters[0].x, clusters[0].y + 35, 'rgba(0, 230, 118, 0.2)');

    // Draw Halos & Text
    clusters.forEach(c => {
        // Outer halo
        const grad = ctx.createRadialGradient(c.x, c.y, 8, c.x, c.y, 90);
        grad.addColorStop(0, c.color + '0.35)');
        grad.addColorStop(0.6, c.color + '0.12)');
        grad.addColorStop(1, c.color + '0.0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(c.x, c.y, 90, 0, Math.PI * 2);
        ctx.fill();

        // Inner glowing core
        ctx.beginPath();
        ctx.arc(c.x, c.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = c.hex;
        ctx.shadowColor = c.hex;
        ctx.shadowBlur = 12;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Labels
        ctx.fillStyle = '#f8fafc';
        ctx.font = '700 13px Outfit, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(c.name, c.x, c.y + 24);

        ctx.fillStyle = 'rgba(148, 163, 184, 0.85)';
        ctx.font = '500 10.5px JetBrains Mono, monospace';
        ctx.fillText(c.vitals, c.x, c.y + 40);
    });

    // 3. Fallback default trajectory if none provided yet
    if (!records || records.length === 0) {
        if (currentSyntheticData && currentSyntheticData.records) {
            records = currentSyntheticData.records;
        } else {
            const defaultStates = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 2,2,2,2, 1,1,0,0];
            records = defaultStates.map((s, idx) => ({ hour: idx + 1, state_id: s }));
        }
    }

    // Build Trajectory Coordinates
    currentManifoldPoints = records.map((r, i) => {
        const c = clusters[r.state_id] || clusters[0];
        const angle = i * 0.9;
        const radius = 16 + (i % 3) * 6;
        const jitterX = Math.cos(angle) * radius;
        const jitterY = Math.sin(angle) * radius;
        return {
            hour: r.hour || i + 1,
            state_id: r.state_id,
            state_name: STATE_NAMES[r.state_id] || 'Stable',
            x: c.x + jitterX,
            y: c.y + jitterY
        };
    });

    // Draw Smooth Trajectory Curve
    if (currentManifoldPoints.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = '#00f2fe';
        ctx.lineWidth = 2.8;
        ctx.shadowColor = '#00f2fe';
        ctx.shadowBlur = 14;

        ctx.moveTo(currentManifoldPoints[0].x, currentManifoldPoints[0].y);
        for (let i = 1; i < currentManifoldPoints.length; i++) {
            const prev = currentManifoldPoints[i - 1];
            const curr = currentManifoldPoints[i];
            const midX = (prev.x + curr.x) / 2;
            const midY = (prev.y + curr.y) / 2;
            ctx.quadraticCurveTo(prev.x, prev.y, midX, midY);
        }
        const last = currentManifoldPoints[currentManifoldPoints.length - 1];
        ctx.lineTo(last.x, last.y);
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Draw Milestone Checkpoints
        currentManifoldPoints.forEach((pt, i) => {
            const isKeyHour = (i === 0 || i === 5 || i === 11 || i === 17 || i === 23);
            const r = isKeyHour ? 6 : 3.5;

            ctx.beginPath();
            ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
            ctx.fillStyle = i === 0 ? '#00e676' : (i === 23 ? '#00f2fe' : (pt.state_id === 3 ? '#f43f5e' : '#38bdf8'));
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = isKeyHour ? 1.5 : 1;
            ctx.stroke();

            // Milestone Text Badges
            if (isKeyHour) {
                const label = i === 0 ? 'H1 (Admit)' : (i === 23 ? 'H24 (End)' : `H${i + 1}`);
                ctx.fillStyle = '#f8fafc';
                ctx.font = '700 10px JetBrains Mono, monospace';
                ctx.textAlign = 'center';
                ctx.fillText(label, pt.x, pt.y - 10);
            }
        });
    }
}

function animatePatientTrajectory() {
    if (!currentManifoldPoints || currentManifoldPoints.length === 0) {
        renderManifoldTrajectory();
    }
    if (currentManifoldPoints.length === 0) return;

    if (manifoldAnimTimer) clearInterval(manifoldAnimTimer);

    let step = 0;
    const totalSteps = currentManifoldPoints.length;

    manifoldAnimTimer = setInterval(() => {
        if (step >= totalSteps) {
            clearInterval(manifoldAnimTimer);
            manifoldAnimTimer = null;
            return;
        }

        const pt = currentManifoldPoints[step];
        const hudHour = document.getElementById('hudActiveHour');
        const hudState = document.getElementById('hudActiveState');
        if (hudHour) hudHour.innerText = `Trajectory: Hour ${pt.hour} of 24`;
        if (hudState) hudState.innerText = `Current Region: ${pt.state_name} (State ${pt.state_id})`;

        // Redraw canvas with moving highlight orb
        renderManifoldTrajectory();
        const canvas = document.getElementById('manifoldCanvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            // Draw active pulsing traveler orb
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 11, 0, Math.PI * 2);
            ctx.fillStyle = '#00f2fe';
            ctx.shadowColor = '#00f2fe';
            ctx.shadowBlur = 25;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 18, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(0, 242, 254, 0.7)';
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.shadowBlur = 0;
        }

        playBeep(480 + pt.state_id * 100, 'triangle', 0.04);
        step++;
    }, 180);
}

// =========================================================================
// 9. DATASET VIEWER & METRICS
// =========================================================================
async function loadBenchmarkMetrics() {
    try {
        const res = await fetch('/api/metrics');
        const data = await res.json();

        const elRule = document.getElementById('metricRuleComp');
        if (elRule) elRule.innerText = `${data.rule_compliance.ours.toFixed(1)}%`;
        const elBounds = document.getElementById('metricBoundsAdh');
        if (elBounds) elBounds.innerText = `${data.bounds_adherence.ours.toFixed(1)}%`;
        const elCorr = document.getElementById('metricCorr');
        if (elCorr) elCorr.innerText = `${data.correlation.ours.toFixed(4)}`;
        const elDtw = document.getElementById('metricDtw');
        if (elDtw) elDtw.innerText = `${data.dtw.ours.toFixed(4)}`;

        const tbody = document.getElementById('benchmarkTableBody');
        const rows = [
            { dim: 'Statistical Fidelity', metric: 'Kolmogorov-Smirnov (KS)', ours: data.ks_score.ours.toFixed(4), paper: data.ks_score.paper, note: 'Captures marginal distributions across 6 vitals' },
            { dim: 'Statistical Fidelity', metric: 'Correlation Preservation', ours: data.correlation.ours.toFixed(4), paper: data.correlation.paper, note: 'Evaluated directly from neural decoder' },
            { dim: 'Temporal Dynamics', metric: 'DTW Alignment Distance', ours: data.dtw.ours.toFixed(4), paper: data.dtw.paper, note: 'Sequence alignment distance (lower is closer)' },
            { dim: 'Clinical Validity', metric: 'Clinical Rule Compliance', ours: `${data.rule_compliance.ours.toFixed(1)}%`, paper: data.rule_compliance.paper, note: 'SBP > DBP + 10 mmHg & shock index limits' },
            { dim: 'Clinical Validity', metric: 'Physiological Bounds Adherence', ours: `${data.bounds_adherence.ours.toFixed(1)}%`, paper: data.bounds_adherence.paper, note: 'Human physiological reference range compliance' },
            { dim: 'Privacy Protection', metric: 'MIA Attack Resistance', ours: `${data.mia_resistance.ours.toFixed(1)}%`, paper: data.mia_resistance.paper, note: 'Empirical membership inference resistance' },
            { dim: 'Privacy Protection', metric: 'Minimum Record Distance (RMS)', ours: data.min_distance.ours.toFixed(4), paper: data.min_distance.paper, note: 'Indicates synthetic records do not duplicate training samples' }
        ];

        tbody.innerHTML = rows.map(r => `
            <tr>
                <td><strong>${r.dim}</strong></td>
                <td>${r.metric}</td>
                <td><span class="gauge-badge success" style="font-size: 0.8rem;">${r.ours}</span></td>
                <td><span style="font-family: var(--font-mono); color: var(--text-secondary);">${r.paper}</span></td>
                <td style="color: var(--text-muted); font-size: 0.82rem;">${r.note}</td>
            </tr>
        `).join('');

    } catch (e) {
        console.error(e);
    }
}

function initInlineDatasetViewer() {
    const selPat = document.getElementById('inlineSelectPatient');
    if (selPat && selPat.children.length <= 1) {
        for (let i = 1; i <= 60; i++) {
            const opt = document.createElement('option');
            opt.value = i;
            opt.innerText = `Patient #${i}`;
            selPat.appendChild(opt);
        }
    }
    if (selPat) selPat.addEventListener('change', loadInlineDataset);
    const selState = document.getElementById('inlineSelectState');
    if (selState) selState.addEventListener('change', loadInlineDataset);

    loadInlineDataset();
}

async function loadInlineDataset() {
    const selPat = document.getElementById('inlineSelectPatient');
    const selState = document.getElementById('inlineSelectState');
    const patVal = selPat ? selPat.value : 'all';
    const stateVal = selState ? selState.value : 'all';

    try {
        const res = await fetch(`/api/cohort_dataset?patient_id=${patVal}&state_id=${stateVal}`);
        const data = await res.json();
        const tbody = document.getElementById('inlineCohortTableBody');
        const countDisp = document.getElementById('inlineRowCount');

        if (countDisp) countDisp.innerText = `Displaying ${data.records.length} observations (of 1,440 total)`;
        if (!tbody) return;

        tbody.innerHTML = data.records.slice(0, 200).map(r => {
            const sId = r.hidden_state_id;
            const badgeClasses = ['s0', 's1', 's2', 's3'];
            const pp = r['Pulse Pressure'] !== undefined ? r['Pulse Pressure'] : (r['Systolic BP'] - r['Diastolic BP']).toFixed(1);
            const shock = r['Shock Index'] !== undefined ? r['Shock Index'] : (r['Heart Rate'] / Math.max(r['Systolic BP'], 1.0)).toFixed(2);

            return `
                <tr>
                    <td><strong>#${r.synthetic_patient_id}</strong></td>
                    <td>H${r.hour}</td>
                    <td><span class="legend-badge-item ${badgeClasses[sId]}" style="padding: 2px 6px; border-radius: 4px; font-size: 0.72rem;">${STATE_NAMES[sId]}</span></td>
                    <td style="font-family: var(--font-mono);">${parseFloat(r['Heart Rate']).toFixed(1)}</td>
                    <td style="font-family: var(--font-mono);">${parseFloat(r['Systolic BP']).toFixed(1)}</td>
                    <td style="font-family: var(--font-mono);">${parseFloat(r['Diastolic BP']).toFixed(1)}</td>
                    <td style="font-family: var(--font-mono); color: var(--emerald);">${pp}</td>
                    <td style="font-family: var(--font-mono); color: #38bdf8;">${shock}</td>
                    <td style="font-family: var(--font-mono); color: var(--emerald);">${parseFloat(r['SpO2']).toFixed(1)}%</td>
                </tr>
            `;
        }).join('');
    } catch (e) {
        console.error(e);
    }
}

function downloadPatientCSV() {
    if (!currentSyntheticData || currentSyntheticData.length === 0) return;
    const headers = Object.keys(currentSyntheticData[0]).join(',');
    const rows = currentSyntheticData.map(r => Object.values(r).join(',')).join('\n');
    const csvContent = "data:text/csv;charset=utf-8," + headers + "\n" + rows;
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `synthetic_icu_patient_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
