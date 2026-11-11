// Use the engraver's final note positions for jianpu beams, ties and barlines.
// The ABC itself retains readable numeric duration marks for other ABC readers.
function renderCombinedScore(target, abc, options = {}, ends) {
  // Present key, meter and tempo together. Keep Q in the parsed tune for MIDI.
  const tonic = abc.match(/^P:1=([^\n]+)/m)?.[1] || abc.match(/^K:(\S+)/m)?.[1];
  const meter = abc.match(/^M:(\S+)/m)?.[1];
  const displayAbc = abc.replace(/^P:1=[^\n]+\n/gm, "").replace(/^Q:/m, "%%printtempo 0\nQ:");
  const tunes = ABCJS.renderAbc(target, displayAbc, {
    responsive: "resize", oneSvgPerLine: true, staffwidth: 740, paddingbottom: 22,
    ...options,
    afterParsing(tune) {
      let index = 0, held = new Set();
      for (const line of tune.lines) {
        for (const event of line.staff?.[0]?.voices?.[0] || []) {
          if (event.el_type !== "note") continue;
          const outgoing = new Set();
          event.pitches?.forEach((p, i) => {
            p.endTie = ends?.[index]?.[i] ?? held.has(p.pitch);
            if (p.startTie) outgoing.add(p.pitch);
          });
          held = outgoing;
          index++;
        }
      }
    }
  });
  function stroke(parent, d, width = 1) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    for (const [k,v] of Object.entries({d, fill:"none", stroke:"currentColor", "stroke-width":width}))
      path.setAttribute(k,v);
    parent.appendChild(path);
  }
  for (const line of tunes[0]?.lines || []) {
    const voice = line.staff?.[0]?.voices?.[0] || [];
    // The engraver may move individual annotation rows to avoid collisions.
    // Set one baseline per system so a displaced hole never looks like another voice.
    const numberedY = voice.flatMap(e => {
      const text = e.abselem?.children.filter(c => c.type === "text" && c.graphelem?.tagName === "text") || [];
      return text.length === 2 ? [Math.min(...text.map(c=>+c.graphelem.getAttribute("y")))] : [];
    });
    const baseline = numberedY.length ? Math.min(...numberedY) + 14 : 0;
    // abcjs reserves space for left-aligned labels. Once centered on the
    // noteheads, each label must stay inside its own cell in BOTH rows.
    const columns = voice.filter(e => e.abselem && ["note","bar"].includes(e.el_type));
    const cells = new Map();
    const anchor = e => e.abselem.x + (e.el_type === "note" ? 4 : 0);
    columns.forEach((e,i) => {
      if (e.el_type !== "note") return;
      const x = anchor(e);
      const margin = neighbour => !neighbour ? Infinity :
        Math.abs(x-anchor(neighbour)) / (neighbour.el_type === "bar" ? 1 : 2) - 2;
      cells.set(e, [margin(columns[i-1]), margin(columns[i+1])]);
    });
    let previous, beam, rowY;
    for (const event of voice) {
      const abs = event.abselem;
      if (!abs) continue;
      if (event.el_type === "bar") {
        if (rowY !== undefined) stroke(previous.parent, `M${abs.x} ${rowY-14}v18`, .85);
        beam = null;
        continue;
      }
      if (event.el_type !== "note") continue;
      const labels = abs.children.filter(c => c.type === "text" && c.graphelem?.tagName === "text")
        .sort((a,b) => +a.graphelem.getAttribute("y") - +b.graphelem.getAttribute("y"));
      if (labels.length !== 2) continue;
      const [degree, hole] = labels.map(c => c.graphelem);
      const parent = degree.parentNode;
      const x = abs.x + 4, y = baseline;
      degree.setAttribute("y", y);
      hole.setAttribute("y", y + 28);
      rowY = y;
      const value = degree.textContent;
      const count = value.includes("̳") ? 2 : value.includes("̲") ? 1 : 0;
      for (const text of [degree,hole]) {
        text.setAttribute("text-anchor","middle");
        text.setAttribute("x",x);
        for (const child of text.querySelectorAll("tspan")) child.setAttribute("x",x);
      }
      // Start the first digit at the notehead; continuation dashes extend right.
      if (value.includes(" –")) {
        degree.setAttribute("text-anchor","start");
        degree.setAttribute("transform","translate(-4,0)");
      }
      degree.querySelector("tspan").textContent = value.replace(/[̲̳]/g,"");
      const [leftSpace,rightSpace] = cells.get(event);
      for (const text of [degree,hole]) {
        const space = text.getAttribute("text-anchor") === "start" ?
          rightSpace + 4 : 2 * Math.min(leftSpace,rightSpace);
        const span = text.querySelector("tspan");
        if (span && span.getComputedTextLength() > space) {
          span.setAttribute("textLength", Math.max(1,space));
          span.setAttribute("lengthAdjust","spacingAndGlyphs");
        }
      }
      if (count) {
        const joined = beam && !event.startBeam && previous?.beamed;
        for (let i=0;i<count;i++) {
          const left = joined && i < beam.count ? beam.x : x-4;
          stroke(parent,`M${left} ${y+3+i*2.8}H${x+4}`,1.15);
        }
      }
      const tiesIn = event.pitches?.some(p=>p.endTie);
      const high = y-18;
      if (tiesIn) {
        const left = previous?.x ?? x-12;
        stroke(parent,`M${left+2} ${high} Q${(left+x)/2} ${high-8} ${x-2} ${high}`);
      }
      beam = count ? {x,count} : null;
      previous = {x,parent,beamed:!event.endBeam && count>0,
                  tied:event.pitches?.some(p=>p.startTie)};
    }
    if (previous?.tied) stroke(previous.parent,`M${previous.x+2} ${rowY-18} q8 -5 14 -3`);
  }
  const parent = typeof target === "string" ? document.getElementById(target) : target;
  const systems = Array.from(parent.querySelectorAll("svg"));
  for (const [i, svg] of systems.entries()) {
    // Empty rest annotations can report (0,0) even far down a long tune.
    // Measure only visible marks, in this system's own SVG coordinates.
    const matrix = svg.getScreenCTM();
    const width = svg.viewBox.baseVal.width;
    let inkTop = Infinity, inkBottom = -Infinity;
    if (matrix?.d > 0) {
      for (const mark of svg.querySelectorAll("path,text,line,rect,ellipse,circle,polygon,polyline")) {
        if (mark.tagName === "text" && !mark.textContent.trim()) continue;
        const rect = mark.getBoundingClientRect();
        if (!rect.width && !rect.height) continue;
        inkTop = Math.min(inkTop, (rect.top - matrix.f) / matrix.d);
        inkBottom = Math.max(inkBottom, (rect.bottom - matrix.f) / matrix.d);
      }
    }
    if (!Number.isFinite(inkTop)) {
      const box = svg.getBBox();
      inkTop = box.y;
      inkBottom = box.y + box.height;
    }
    let top = inkTop - 3;
    if (i === 0) {
      const tempo = tunes[0]?.metaText?.tempo?.bpm;
      const meta = document.createElementNS("http://www.w3.org/2000/svg", "text");
      meta.setAttribute("class", "score-metadata");
      meta.setAttribute("x", width / 2);
      meta.setAttribute("y", inkTop - 12);
      meta.setAttribute("text-anchor", "middle");
      meta.setAttribute("font-family", "sans-serif");
      meta.setAttribute("font-size", "14");
      meta.setAttribute("fill", "#43584c");
      meta.textContent = [tonic && "1 = " + tonic, meter, tempo && tempo + " BPM"].filter(Boolean).join("   ·   ");
      svg.append(meta);
      top = meta.getBBox().y - 2;
    }
    // Fit each system after moving the annotation rows. Its bottom padding
    // also separates the fingering from the following system in every export.
    svg.setAttribute("viewBox", [0, top, width, inkBottom + 22 - top].join(" "));
  }
  return tunes;
}
