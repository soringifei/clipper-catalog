(() => {
  const projects = Array.isArray(window.CLIPPER_PROJECTS) ? window.CLIPPER_PROJECTS : [];
  const list = document.getElementById('project-list');
  const viewer = document.getElementById('viewer');
  const video = document.getElementById('viewer-video');
  const closeButton = document.getElementById('viewer-close');
  if (!list || !viewer || !video || !closeButton || projects.length === 0) return;

  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  };
  let lastTrigger = null;

  function openProject(project, trigger) {
    if (typeof viewer.showModal !== 'function') {
      window.location.href = project.video;
      return;
    }
    lastTrigger = trigger;
    document.getElementById('viewer-meta').textContent = `${project.category} / ${project.duration}`;
    document.getElementById('viewer-title').textContent = project.title;
    document.getElementById('viewer-file').href = project.video;
    video.poster = project.poster;
    const source = element('source');
    source.src = project.video;
    source.type = 'video/mp4';
    video.replaceChildren(source);
    if (project.captions) {
      const track = element('track');
      track.kind = 'captions';
      track.label = 'English';
      track.srclang = 'en';
      track.src = project.captions;
      video.append(track);
    }
    video.append(document.createTextNode('Your browser cannot play this video.'));
    video.load();
    viewer.showModal();
    closeButton.focus();
  }

  const cards = projects.map((project, index) => {
    const article = element('article', 'catalog-card');
    article.id = project.slug;
    const meta = element('div', 'card-meta');
    meta.append(element('span', 'card-index', `${String(index + 1).padStart(2, '0')} / ${project.category}`));
    meta.append(element('span', '', project.duration));

    const frame = element('div', project.coverText === false ? 'card-frame card-frame--finished-art' : 'card-frame');
    const visual = element('button', 'card-visual');
    visual.type = 'button';
    visual.setAttribute('aria-label', `Watch ${project.title}, ${project.duration}`);
    const poster = element('img');
    poster.src = project.cover;
    poster.style.objectPosition = project.coverPosition;
    poster.alt = '';
    poster.loading = index === 0 ? 'eager' : 'lazy';
    poster.decoding = 'async';
    visual.append(poster);
    const view = element('span', 'card-view', 'PLAY  ↗');
    visual.append(view);
    visual.addEventListener('click', () => openProject(project, visual));
    frame.append(visual);
    if (project.coverText !== false) {
      const overlay = element('div', 'cover-overlay');
      overlay.append(element('span', 'cover-subject', project.subject));
      const headline = element('h3', 'cover-headline');
      headline.append(element('span', '', project.coverLead), element('span', 'cover-accent', project.coverAccent));
      overlay.append(headline);
      frame.append(overlay);
    }
    const info = element('div', 'card-info');
    if (project.coverText === false) info.append(element('h3', 'card-title', project.title));
    info.append(element('p', 'card-note', project.note));
    const permalink = element('a', 'card-permalink', 'Link to this edit ↗');
    permalink.href = `#${project.slug}`;
    permalink.setAttribute('aria-label', `Link to ${project.title}`);
    info.append(permalink);
    article.append(meta, frame, info);
    return article;
  });
  list.replaceChildren(...cards);
  const linkedCard = document.getElementById(window.location.hash.slice(1));
  if (linkedCard?.classList.contains('catalog-card')) {
    requestAnimationFrame(() => linkedCard.scrollIntoView({ block: 'start' }));
  }

  closeButton.addEventListener('click', () => viewer.close());
  viewer.addEventListener('click', event => {
    if (event.target === viewer) viewer.close();
  });
  viewer.addEventListener('close', () => {
    video.pause();
    video.replaceChildren(document.createTextNode('Your browser cannot play this video.'));
    video.removeAttribute('poster');
    video.load();
    lastTrigger?.focus();
  });
})();
