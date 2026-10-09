(() => {
  const projects = Array.isArray(window.CLIPPER_PROJECTS) ? window.CLIPPER_PROJECTS : [];
  const $ = id => document.getElementById(id);
  const track = $('stage-track'), list = $('stage-list'), frame = $('device-frame'), ticks = $('device-ticks');
  const note = $('stage-note'), tags = $('stage-tags'), rail = $('rail'), head = document.querySelector('.work-head-copy');
  const viewer = $('viewer'), vvideo = $('viewer-video'), vclose = $('viewer-close');
  if (!track || !list || !frame || !viewer || !vvideo || !vclose || !projects.length) return;

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  const wide = window.matchMedia('(min-width: 900px)');
  const el = (t, c, x) => { const n = document.createElement(t); if (c) n.className = c; if (x) n.textContent = x; return n; };
  const n = projects.length;
  track.style.setProperty('--n', n);
  let lastTrigger = null, active = -1, pauseTimer = 0;

  /* ---- viewer ---- */
  function openProject(p, trigger) {
    lastTrigger = trigger || null;
    $('viewer-meta').textContent = p.category + ' / ' + p.duration;
    $('viewer-title').textContent = p.title;
    $('viewer-note').textContent = p.note || '';
    $('viewer-rights').textContent = p.rights || '';
    const f = $('viewer-file'); f.href = p.video;
    vvideo.poster = p.poster || '';
    vvideo.src = p.video;
    if (!viewer.open) viewer.showModal();
    vvideo.currentTime = 0;
    const pr = vvideo.play(); if (pr && pr.catch) pr.catch(() => {});
    history.replaceState(null, '', '#' + p.slug);
  }
  function closeViewer() { if (viewer.open) viewer.close(); }
  viewer.addEventListener('close', () => {
    vvideo.pause(); vvideo.removeAttribute('src'); vvideo.load();
    history.replaceState(null, '', '#work');
    if (lastTrigger && lastTrigger.focus) lastTrigger.focus();
  });
  vclose.addEventListener('click', closeViewer);
  viewer.addEventListener('click', e => { if (e.target === viewer) closeViewer(); });

  /* ---- stage rows, device videos, ticks ---- */
  list.textContent = '';
  const bar = el('span', 'stage-bar'); bar.setAttribute('aria-hidden', 'true');
  list.appendChild(bar);
  const rows = [], vids = [], tickEls = [];
  projects.forEach((p, i) => {
    const li = el('li', 'stage-item');
    const b = el('button', 'stage-row'); b.type = 'button';
    b.append(el('span', 'stage-num', String(i + 1).padStart(2, '0')), el('span', 'stage-name', p.title), el('span', 'stage-meta', p.category + ' / ' + p.duration));
    b.addEventListener('click', () => { if (wide.matches) scrollToIndex(i); else openProject(p, b); });
    li.appendChild(b); list.appendChild(li); rows.push(b);

    const v = el('video', 'device-video');
    v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'none'; v.poster = p.cover || p.poster || '';
    v.setAttribute('aria-hidden', 'true'); v.tabIndex = -1;
    frame.appendChild(v); vids.push(v);

    const t = el('span', 'tick'); ticks.appendChild(t); tickEls.push(t);
  });
  const hit = el('button', 'device-hit'); hit.type = 'button';
  const chip = el('span', 'device-chip', 'Watch with sound'); hit.appendChild(chip);
  hit.addEventListener('click', () => openProject(projects[Math.max(active, 0)], hit));
  frame.appendChild(hit);

  function setActive(i) {
    if (i === active) return;
    active = i;
    list.style.setProperty('--i', i);
    rows.forEach((r, k) => { r.classList.toggle('is-active', k === i); r.setAttribute('aria-current', k === i ? 'true' : 'false'); });
    tickEls.forEach((t, k) => t.classList.toggle('is-on', k <= i));
    hit.setAttribute('aria-label', 'Watch ' + projects[i].title + ' with sound');
    clearTimeout(pauseTimer);
    const v = vids[i];
    if (!v.getAttribute('src')) v.src = projects[i].video;
    vids.forEach((x, k) => x.classList.toggle('is-active', k === i));
    if (!reduce.matches) { const pr = v.play(); if (pr && pr.catch) pr.catch(() => {}); }
    pauseTimer = setTimeout(() => vids.forEach((x, k) => { if (k !== i) x.pause(); }), 450);
    note.classList.add('is-swap');
    setTimeout(() => {
      note.textContent = projects[i].note || '';
      tags.textContent = '';
      [projects[i].category, projects[i].duration, projects[i].sourceLabel].filter(Boolean).forEach(x => tags.appendChild(el('li', 'work-tag', x)));
      note.classList.remove('is-swap');
    }, reduce.matches ? 0 : 140);
  }

  /* ---- scroll progress -> active index (desktop only) ---- */
  let ticking = false;
  function onScroll() {
    if (ticking) return; ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      if (!wide.matches) return;
      const r = track.getBoundingClientRect();
      const span = Math.max(1, r.height - window.innerHeight);
      const p = Math.min(1, Math.max(0, -r.top / span));
      setActive(Math.min(n - 1, Math.floor(p * n)));
    });
  }
  function scrollToIndex(i) {
    const top = track.getBoundingClientRect().top + window.scrollY;
    const span = track.offsetHeight - window.innerHeight;
    window.scrollTo({ top: top + span * ((i + 0.5) / n), behavior: reduce.matches ? 'auto' : 'smooth' });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  wide.addEventListener('change', onScroll);

  /* ---- mobile rail ---- */
  if (rail) {
    rail.textContent = '';
    projects.forEach(p => {
      const li = el('li', 'rail-item');
      const b = el('button', 'rail-card'); b.type = 'button'; b.setAttribute('aria-label', 'Watch ' + p.title);
      const img = el('img'); img.src = p.cover || p.poster; img.alt = ''; img.loading = 'lazy';
      const v = el('video', 'rail-video'); v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'none'; v.setAttribute('aria-hidden', 'true');
      const cap = el('span', 'rail-cap'); cap.append(el('strong', '', p.title), el('small', '', p.category + ' / ' + p.duration));
      b.append(img, v, cap);
      b.addEventListener('click', () => openProject(p, b));
      li.appendChild(b); rail.appendChild(li);
      p._rv = v;
    });
    if ('IntersectionObserver' in window && !reduce.matches) {
      const io = new IntersectionObserver(es => es.forEach(e => {
        const v = e.target.querySelector('.rail-video'); if (!v || wide.matches) return;
        if (e.isIntersecting && e.intersectionRatio >= 0.6) {
          if (!v.getAttribute('src')) v.src = projects[[...rail.children].indexOf(e.target)].video;
          v.classList.add('is-live'); const pr = v.play(); if (pr && pr.catch) pr.catch(() => {});
        } else { v.pause(); v.classList.remove('is-live'); }
      }), { root: rail, threshold: [0, 0.6] });
      [...rail.children].forEach(c => io.observe(c));
    }
  }

  /* ---- heading blur-in ---- */
  if (head) requestAnimationFrame(() => requestAnimationFrame(() => head.classList.add('is-in')));

  /* ---- init + deep link ---- */
  setActive(0);
  onScroll();
  const slug = location.hash.slice(1), di = projects.findIndex(p => p.slug === slug);
  if (di >= 0) openProject(projects[di], null);
})();
