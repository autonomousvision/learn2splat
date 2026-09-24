(function () {
    const PATH = 'static/videos/main/';
    const MANIFEST = 'static/videos/manifest.json';

    // Scene groups, in the order their buttons appear. A scene joins the first group whose
    // `match` accepts the dataset name recorded for it in the manifest.
    const GROUPS = [
        // named as the hero's Setting pills are, so one setting reads the same across the page,
        // and ordered to match: dense leads there, so it leads here and opens by default
        {id: 'dense', label: 'Dense views · SfM init', match: d => /dense/.test(d)},
        {id: 'sparse', label: 'Sparse views · ReSplat init', match: d => /sparse/.test(d)}
    ];

    // One <video> per stream; a scene uses as many as it has methods, so the four-method sparse
    // scenes fill all of them and the three-method dense ones leave the last idle.
    const VIDEO_IDS = ['vidLT', 'vidRT', 'vidLB', 'vidRB'];

    const root = document.getElementById('videoTeaser');
    const scenesRoot = document.getElementById('videoTeaserScenes');
    const canvas = document.getElementById('videoTeaserCanvas');
    if (!root || !scenesRoot || !canvas) return;

    const ctx = canvas.getContext('2d', {alpha: false});
    if (!ctx) return;

    function isMobile() {
        return window.innerWidth <= 960;
    }

    function clamp(v, a, b) {
        return Math.max(a, Math.min(b, v));
    }

    class VideoStream {
        constructor(videoEl) {
            this.el = videoEl;
            this.base = null;
            this.label = '';
            this.readyResolver = null;
            this.readyPromise = null;
            this.bindReadyHandlers();
            this.el.muted = true;
            this.el.playsInline = true;
            this.el.loop = true;
            this.el.preload = 'auto';
        }

        bindReadyHandlers() {
            const resolveReady = () => {
                if (!this.readyResolver) return;
                clearTimeout(this._readyTimeout);
                this._readyTimeout = null;
                this.readyResolver();
                this.readyResolver = null;
                this.readyPromise = null;
            };
            this.el.addEventListener('canplaythrough', resolveReady);
            this.el.addEventListener('canplay', resolveReady);
        }

        load(base, label, sceneId) {
            this.base = base;
            this.label = label;
            const url = PATH + base + '_' + sceneId + '.mp4';

            // Always create a fresh promise so rapid scene switches don't share state
            clearTimeout(this._readyTimeout);
            this._readyTimeout = null;
            this.readyPromise = new Promise(resolve => {
                this.readyResolver = resolve;
            });

            if (this._loadedUrl !== url) {
                this._loadedUrl = url;
                this.el.src = url;
                this.el.load();
            } else if (this.el.readyState >= 3) {
                // Already buffered — resolve immediately
                this.readyResolver();
                this.readyResolver = null;
                this.readyPromise = null;
                return Promise.resolve();
            }

            // Fallback: resolve after 3 s so a failed canplay never permanently stalls playback
            this._readyTimeout = setTimeout(() => {
                this._readyTimeout = null;
                if (this.readyResolver) {
                    this.readyResolver();
                    this.readyResolver = null;
                    this.readyPromise = null;
                }
            }, 3000);

            return this.readyPromise;
        }

        playFromStart() {
            try {
                this.el.currentTime = 0;
            } catch (e) {
                // no-op
            }
            const p = this.el.play();
            if (p && typeof p.catch === 'function') p.catch(function () {
            });
            return p;
        }
    }

    const allStreams = VIDEO_IDS
        .map(id => document.getElementById(id))
        .filter(Boolean)
        .map(el => new VideoStream(el));

    if (!allStreams.length) return;

    let xPos = 0.5;
    let yPos = 0.5;
    let scenes = [];          // every scene in the manifest
    let labels = {};          // method key -> what the page prints over its panel
    let groups = [];          // [{id, label, scenes: [...]}], only those with scenes
    let activeGroup = null;
    let activeScene = null;
    let streams = [];         // the streams the active scene uses, in method order

    // ── Panel layout ──────────────────────────────────────────────────────────
    // Where each method's panel sits, for as many methods as the scene has. Four split into
    // quadrants about the pointer; three give the first method the full left column and split
    // the right one; two split left/right. The pointer drives every boundary, so one drag
    // reveals all of them.
    function layout(n, cvW, cvH) {
        const col = cvW * xPos;
        const row = cvH * yPos;
        if (n <= 1) return [{x: 0, y: 0, w: cvW, h: cvH}];
        if (n === 2) {
            return [{x: 0, y: 0, w: col, h: cvH},
                    {x: col, y: 0, w: cvW - col, h: cvH}];
        }
        if (n === 3) {
            return [{x: 0, y: 0, w: col, h: cvH},
                    {x: col, y: 0, w: cvW - col, h: row},
                    {x: col, y: row, w: cvW - col, h: cvH - row}];
        }
        return [{x: 0, y: 0, w: col, h: row},
                {x: col, y: 0, w: cvW - col, h: row},
                {x: 0, y: row, w: col, h: cvH - row},
                {x: col, y: row, w: cvW - col, h: cvH - row}];
    }

    function drawSplitLines(n, cvW, cvH) {
        const col = cvW * xPos;
        const row = cvH * yPos;
        ctx.strokeStyle = '#aaa';
        ctx.lineWidth = 2;
        ctx.beginPath();
        if (n >= 2) {
            ctx.moveTo(col, 0);
            ctx.lineTo(col, cvH);
        }
        if (n === 3) {
            ctx.moveTo(col, row);
            ctx.lineTo(cvW, row);
        } else if (n >= 4) {
            ctx.moveTo(0, row);
            ctx.lineTo(cvW, row);
        }
        ctx.stroke();
    }

    // ── Scene selection ───────────────────────────────────────────────────────
    function sceneById(id) {
        return scenes.find(s => s.id === id) || null;
    }

    function groupOf(scene) {
        const g = GROUPS.find(g => g.match(scene.dataset || ''));
        return g ? g.id : GROUPS[0].id;
    }

    function setScene(sceneId, force) {
        const scene = sceneById(sceneId);
        if (!scene || (!force && activeScene && scene.id === activeScene.id)) return;
        activeScene = scene;

        const shouldPlay = streams.some(s => !s.el.paused) || !streams.length;
        // A scene with fewer methods leaves the spare video elements idle rather than showing
        // a stale frame from the scene before it.
        allStreams.forEach(s => s.el.pause());
        streams = scene.methods.slice(0, allStreams.length).map((m, i) => allStreams[i]);

        Promise.all(streams.map((s, i) =>
            s.load(scene.methods[i], labels[scene.methods[i]] || scene.methods[i], scene.id)
        )).then(function () {
            if (!shouldPlay) return;
            streams.forEach(s => s.playFromStart());
        });
        renderButtons();
        renderNote();
    }

    function setGroup(groupId) {
        const group = groups.find(g => g.id === groupId);
        if (!group) return;
        activeGroup = group;
        setScene(group.scenes[0].id, true);
    }

    // ── Toolbar ───────────────────────────────────────────────────────────────
    // The group row is created here rather than in the page markup, so a site that ships the
    // widget's HTML unchanged still gets the switcher.
    const groupsRoot = document.createElement('div');
    groupsRoot.className = 'video-teaser-scenes video-teaser-groups';
    groupsRoot.setAttribute('role', 'radiogroup');
    groupsRoot.setAttribute('aria-label', 'Scene type selector');
    scenesRoot.parentNode.insertBefore(groupsRoot, scenesRoot);

    function buttonsHtml(items, activeId) {
        return items.map(function (item) {
            const active = item.id === activeId;
            const btn = '<button type="button" class="video-teaser-scene-btn' +
                (active ? ' is-active' : '') + '" data-id="' + item.id +
                '" aria-pressed="' + (active ? 'true' : 'false') + '">' + item.label + '</button>';
            // the dense group carries the asterisk the note under the widget answers; it sits
            // outside the pill, where ReSplat's green reads against the page
            if (item.id !== 'dense') return btn;
            return '<span class="video-btn-wrap">' + btn +
                '<span class="video-pill-mark" aria-hidden="true">*</span></span>';
        }).join('');
    }

    const COUNT_WORDS = {2: 'both', 3: 'all three', 4: 'all four'};

    function renderNote() {
        const note = root.querySelector('.video-teaser-note');
        if (!note || !activeScene) return;
        const n = activeScene.methods.length;
        note.textContent = 'Drag the split lines to compare ' +
            (COUNT_WORDS[n] || ('all ' + n)) + ' methods in one view; click to freeze them.';
    }

    function renderButtons() {
        groupsRoot.innerHTML = buttonsHtml(groups, activeGroup && activeGroup.id);
        scenesRoot.innerHTML = buttonsHtml(
            (activeGroup ? activeGroup.scenes : []).map(s => ({id: s.id, label: s.id})),
            activeScene && activeScene.id);
        groupsRoot.querySelectorAll('.video-teaser-scene-btn').forEach(btn => {
            btn.addEventListener('click', () => setGroup(btn.dataset.id));
        });
        scenesRoot.querySelectorAll('.video-teaser-scene-btn').forEach(btn => {
            btn.addEventListener('click', () => setScene(btn.dataset.id));
        });
    }

    function setupKeyboard(container, onPick) {
        container.addEventListener('keydown', function (e) {
            const nav = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'];
            if (nav.indexOf(e.key) === -1) return;
            const buttons = Array.from(container.querySelectorAll('.video-teaser-scene-btn'));
            const idx = buttons.indexOf(document.activeElement);
            if (!buttons.length || idx === -1) return;
            e.preventDefault();
            let next = idx;
            if (e.key === 'Home') next = 0;
            if (e.key === 'End') next = buttons.length - 1;
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (idx + 1) % buttons.length;
            if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (idx - 1 + buttons.length) % buttons.length;
            const nextBtn = buttons[next];
            if (!nextBtn) return;
            onPick(nextBtn.dataset.id);
            nextBtn.focus();
        });
    }

    // ── Canvas ────────────────────────────────────────────────────────────────
    // The canvas takes the active scene's own aspect: the sparse DL3DV videos are 960x512 but
    // the dense ones are 3:2, and a fixed 960x512 canvas stretched them.
    function videoAspect() {
        const ref = streams.length ? streams[0].el : null;
        return ref && ref.videoWidth && ref.videoHeight ? ref.videoHeight / ref.videoWidth : 512 / 960;
    }

    function setCanvasSize() {
        const ratio = window.devicePixelRatio || 1;
        const logicalW = isMobile() ? Math.max(320, Math.round(Math.min(root.clientWidth - 24, 960))) : 960;
        const logicalH = Math.round(logicalW * videoAspect());
        canvas.style.width = logicalW + 'px';
        canvas.style.height = logicalH + 'px';
        canvas.width = Math.round(logicalW * ratio);
        canvas.height = Math.round(logicalH * ratio);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    allStreams.forEach(s => s.el.addEventListener('loadedmetadata', function () {
        if (streams[0] === s) setCanvasSize();
    }));

    function updatePointer(clientX, clientY) {
        const rect = canvas.getBoundingClientRect();
        xPos = clamp((clientX - rect.left) / rect.width, 0, 1);
        yPos = clamp((clientY - rect.top) / rect.height, 0, 1);
    }

    // The split lines follow the pointer; a click freezes them where they are so a comparison can
    // be studied (and pointed at) without the panels moving. Clicking again releases them.
    let splitFrozen = false;

    if (!isMobile()) {
        canvas.addEventListener('mousemove', function (e) {
            if (splitFrozen) return;
            updatePointer(e.clientX, e.clientY);
        });
        canvas.addEventListener('click', function (e) {
            splitFrozen = !splitFrozen;
            if (!splitFrozen) updatePointer(e.clientX, e.clientY);
            canvas.style.cursor = splitFrozen ? 'default' : 'crosshair';
        });
    }

    // The iteration each frame belongs to, read off the active scene's own label track: scenes
    // are logged at different steps and run to different lengths, so the track is per scene.
    function currentIterLabel() {
        const map = activeScene && activeScene.iterations;
        if (!map || !map.length || !streams.length) return '';
        const ref = streams[0].el;
        if (!ref.duration) return '';
        const idx = clamp(Math.floor(ref.currentTime / ref.duration * map.length), 0, map.length - 1);
        let val = map[idx];
        if (val === 'orbit') {
            // Mid-sweep: keep printing the step the sweep was spliced in at.
            for (let i = idx - 1; i >= 0; i--) {
                if (map[i] !== 'orbit') {
                    val = map[i];
                    break;
                }
            }
        }
        return val === 'orbit' ? '' : 't = ' + val;
    }

    function drawStream(stream, sx, sy, sw, sh, dx, dy, dw, dh) {
        if (!stream.el || stream.el.readyState < 2) {
            ctx.fillStyle = '#000';
            ctx.fillRect(dx, dy, dw, dh);
            return;
        }
        try {
            ctx.drawImage(stream.el, sx, sy, sw, sh, dx, dy, dw, dh);
        } catch (e) {
            ctx.fillStyle = '#000';
            ctx.fillRect(dx, dy, dw, dh);
        }
    }

    // Our method's name is drawn in the theme's purple, the colour it has everywhere else on the page.
    const OURS_BASE = 'learn2splat';
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#9E5EBB';

    function chipPath(x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath();
    }

    // Each name sits in a rounded chip, as the page's own pills do: ours in the theme purple, the
    // baselines in translucent ink. White on a solid ground stays readable over any frame, which
    // neither the old black outline nor bare purple text managed against bright sky.
    const CHIP_INK = 'rgba(17, 19, 28, .62)';

    function drawLabel(text, x, y, sub, color) {
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.font = '600 14px Inter, sans-serif';
        const padX = 8, padY = 5, h = 25;
        ctx.fillStyle = color || CHIP_INK;
        chipPath(x, y, ctx.measureText(text).width + padX * 2, h, 7);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.fillText(text, x + padX, y + padY);
        if (sub) {
            ctx.font = '11px Inter, sans-serif';
            ctx.fillStyle = CHIP_INK;
            chipPath(x, y + h + 5, ctx.measureText(sub).width + 11, 19, 6);
            ctx.fill();
            ctx.fillStyle = '#fff';
            ctx.fillText(sub, x + 5.5, y + h + 9);
        }
    }

    function drawLoop() {
        const ratio = window.devicePixelRatio || 1;
        const cvW = canvas.width / ratio;
        const cvH = canvas.height / ratio;
        ctx.clearRect(0, 0, cvW, cvH);

        if (streams.length) {
            const ref = streams[0].el;
            const vw = ref.videoWidth || 960;
            const vh = ref.videoHeight || 512;
            const sx = vw / cvW;
            const sy = vh / cvH;
            const rects = layout(streams.length, cvW, cvH);
            const iter = currentIterLabel();
            const pad = 16;

            rects.forEach(function (r, i) {
                drawStream(streams[i], r.x * sx, r.y * sy, r.w * sx, r.h * sy, r.x, r.y, r.w, r.h);
            });
            rects.forEach(function (r, i) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(r.x, r.y, r.w, r.h);
                ctx.clip();
                drawLabel(streams[i].label, r.x + pad, r.y + pad, iter,
                    streams[i].base === OURS_BASE ? accent : null);
                ctx.restore();
            });
            if (!isMobile()) drawSplitLines(streams.length, cvW, cvH);
        }

        syncPlaybackUI();
        requestAnimationFrame(drawLoop);
    }

    // Every stream plays the same timeline, so one starting, pausing or drifting brings the
    // others with it.
    function wireSync() {
        allStreams.forEach(source => {
            source.el.addEventListener('play', function () {
                streams.forEach(other => {
                    if (other === source || !other.el.paused) return;
                    const p = other.el.play();
                    if (p && typeof p.catch === 'function') p.catch(function () {
                    });
                });
            });

            source.el.addEventListener('pause', function () {
                streams.forEach(other => {
                    if (other === source || other.el.paused) return;
                    other.el.pause();
                });
            });

            source.el.addEventListener('timeupdate', function () {
                if (source.el.paused) return;
                streams.forEach(other => {
                    if (other === source) return;
                    if (Math.abs(other.el.currentTime - source.el.currentTime) > 0.2) {
                        other.el.currentTime = source.el.currentTime;
                    }
                });
            });
        });
    }

    const playPauseBtn = document.getElementById('videoTeaserPlayPause');
    const scrub = document.getElementById('videoTeaserScrub');
    const timeLabel = document.getElementById('videoTeaserTime');
    let paused = false;
    // true while the scrubber is being dragged, so playback does not fight the handle
    let scrubbing = false;
    let seekInFlight = false;
    let pendingSeek = null;   // newest drag target, applied when the running seek finishes

    const SCRUB_MAX = 1000;

    // Every stream shares one timeline, so a seek moves all of them together.
    function applySeek(frac) {
        seekInFlight = true;
        streams.forEach(function (s) {
            const d = s.el.duration;
            if (!d) return;
            try {
                s.el.currentTime = clamp(frac, 0, 1) * d;
            } catch (e) {
                // a stream that has not loaded its metadata yet simply keeps its position
            }
        });
    }

    // These MP4s carry only a couple of keyframes, so each seek decodes a long way forward.
    // Restarting one on every pointer move cancels the last, and nothing paints until the drag
    // ends; instead the newest target waits and is applied once the previous seek completes, so
    // frames keep arriving as fast as decoding allows.
    function seekFraction(frac) {
        if (seekInFlight) {
            pendingSeek = frac;
            return;
        }
        applySeek(frac);
    }

    allStreams.forEach(function (s) {
        s.el.addEventListener('seeked', function () {
            if (!streams.length || s !== streams[0]) return;
            seekInFlight = false;
            if (pendingSeek === null) return;
            const next = pendingSeek;
            pendingSeek = null;
            applySeek(next);
        });
    });

    // Called from the draw loop: the handle follows playback, and the readout names the
    // optimization iteration rather than a timestamp, which is what the frames are indexed by.
    function syncPlaybackUI() {
        const ref = streams.length ? streams[0].el : null;
        if (timeLabel) timeLabel.textContent = ref ? currentIterLabel() : '';
        if (!scrub || scrubbing || !ref || !ref.duration) return;
        scrub.value = String(Math.round(ref.currentTime / ref.duration * SCRUB_MAX));
    }

    if (scrub) {
        scrub.addEventListener('input', function () {
            scrubbing = true;
            seekFraction(Number(scrub.value) / SCRUB_MAX);
        });
        // pointerup lands on the handle, change on a keyboard step; both end the drag
        ['change', 'pointerup', 'pointercancel', 'blur'].forEach(function (ev) {
            scrub.addEventListener(ev, function () {
                scrubbing = false;
                // land exactly where the handle was left, even if a seek was still running
                if (!seekInFlight && pendingSeek !== null) {
                    const f = pendingSeek;
                    pendingSeek = null;
                    applySeek(f);
                }
            });
        });
    }

    function updatePlayPauseBtn() {
        if (!playPauseBtn) return;
        playPauseBtn.textContent = paused ? '▶' : '▮▮';
        playPauseBtn.setAttribute('aria-label', paused ? 'Play' : 'Pause');
    }

    if (playPauseBtn) {
        playPauseBtn.addEventListener('click', function () {
            paused = !paused;
            if (paused) {
                streams.forEach(s => s.el.pause());
            } else {
                streams.forEach(s => s.el.play().catch(function () {}));
            }
            updatePlayPauseBtn();
        });
    }

    setCanvasSize();
    setupKeyboard(groupsRoot, setGroup);
    setupKeyboard(scenesRoot, setScene);
    wireSync();
    window.addEventListener('resize', setCanvasSize, {passive: true});
    requestAnimationFrame(drawLoop);

    // The manifest is what the render pipeline writes next to the MP4s: which scenes exist,
    // which methods each one has, and where its frames sit in the optimization.
    fetch(MANIFEST, {cache: 'no-cache'})
        .then(r => r.json())
        .then(function (data) {
            labels = data.labels || {};
            scenes = (data.scenes || []).filter(s => s.methods && s.methods.length);
            if (!scenes.length) throw new Error('manifest lists no playable scene');

            groups = GROUPS
                .map(g => ({id: g.id, label: g.label,
                            scenes: scenes.filter(s => groupOf(s) === g.id)}))
                .filter(g => g.scenes.length);

            const wanted = sceneById(root.dataset.defaultScene)
                || (groups[0] && groups[0].scenes[0]) || scenes[0];
            activeGroup = groups.find(g => g.scenes.indexOf(wanted) !== -1) || groups[0];
            setScene(wanted.id, true);
        })
        .catch(function (err) {
            console.warn('video teaser: could not load ' + MANIFEST, err);
        });
})();
