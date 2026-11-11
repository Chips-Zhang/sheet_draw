const el = id => document.getElementById(id);
const form = el("form");
const optionText = id => el(id).selectedOptions[0].textContent;
async function loadCapabilities() {
  let available = {};
  try {
    const response = await fetch("/capabilities");
    if (response.ok) available = await response.json();
  } catch { /* The default tracking engine works without optional models. */ }
  const songOption = el("source").querySelector('[value="song"]');
  songOption.disabled = !(available.demucs && available.basicPitch);
  if (!songOption.disabled && !el("file").files.length) {
    el("source").value = "song";
    el("source").dispatchEvent(new Event("change"));
  }
  updateSource();
  for (const option of document.querySelectorAll("[data-needs]")) {
    option.disabled = !available[option.dataset.needs];
    if (option.disabled) option.textContent += "（不可用）";
  }
}
function updateSource() {
  const song = el("source").value === "song";
  el("setting-summary").textContent = [song ? "原唱" : "乐器", optionText("key") + " 调", optionText("instrument")].join(" · ");
  el("shift-field").hidden = el("tonic-field").hidden = el("fit-field").hidden = !song;
  for (const id of ["engine", "separation"]) el(id).closest("label").hidden = song;
}
el("source").addEventListener("change", updateSource);
for (const id of ["key","instrument"]) el(id).addEventListener("change",updateSource);
loadCapabilities();

let linkMode = false;
function setInputMode(link) {
  linkMode = link;
  el("file-fields").hidden = link;
  el("link-fields").hidden = !link;
  el("file").required = !link;
  el("media-url").disabled = !link;
  el("media-url").required = link;
  el("file-mode").setAttribute("aria-pressed", String(!link));
  el("link-mode").setAttribute("aria-pressed", String(link));
  if (link) { el("audio").pause(); el("media-url").focus(); }
  clearScore();
  el("status").textContent = "";
}
el("file-mode").onclick = () => setInputMode(false);
el("link-mode").onclick = () => setInputMode(true);
let playbackUrl;
el("file").addEventListener("change", () => {
  if (playbackUrl) URL.revokeObjectURL(playbackUrl);
  const file = el("file").files[0];
  const player = el("audio");
  player.pause();
  player.removeAttribute("src");
  playbackUrl = file ? URL.createObjectURL(file) : null;
  if (playbackUrl) player.src = playbackUrl;
  player.load();
  el("playback").hidden = !file;
});

let currentTune;
let pdfLibrary;

function clearScore() {
  el("score").replaceChildren();
  el("title").textContent = "组合乐谱";
  el("summary").textContent = "";
  el("legend").hidden = true;
  el("export-status").textContent = "";
  el("empty").hidden = el("sample-score").hidden = false;
  el("kind").textContent = "示例节选";
  showSample();
}
form.addEventListener("input", () => { clearScore(); el("status").textContent = ""; });
const exportsMenu = el("export-actions");
function dismissExportMenu() {
  exportsMenu.open = false;
  exportsMenu.querySelector("summary").focus({preventScroll:true});
}
document.addEventListener("pointerdown", event => {
  if (exportsMenu.open && !exportsMenu.contains(event.target)) exportsMenu.open = false;
});
exportsMenu.addEventListener("focusout", event => {
  if (!exportsMenu.contains(event.relatedTarget)) exportsMenu.open = false;
});
exportsMenu.addEventListener("keydown", event => {
  if (event.key === "Escape") { event.preventDefault(); dismissExportMenu(); }
});
el("print").onclick = () => { dismissExportMenu(); window.print(); };

form.onsubmit = async event => {
  event.preventDefault();
  const file = el("file").files[0];
  clearScore();
  if (!linkMode && file.size > 128 * 1024 * 1024) { el("status").textContent = "文件超过 128 MB，请截取较短的录音。"; return; }
  const meter = el("meter").value;
  const query = new URLSearchParams({meter: meter[0], bpm: el("tempo").value});
  if (el("downbeat").value !== "") query.set("downbeat", el("downbeat").value);
  query.set("fit", el("fit").value);
  query.set("source", el("source").value);
  if (el("source").value === "song" && el("tonic").value) query.set("tonic", el("tonic").value);
  if (el("source").value === "song" && el("shift").value !== "") query.set("shift", el("shift").value);
  for (const id of ["instrument", "engine", "key", "subdivision", "separation"]) query.set(id, el(id).value);
  form.setAttribute("aria-busy", "true");
  for (const field of form.elements) field.disabled = true;
  el("status").textContent = linkMode ? "正在解析链接、读取音轨并生成乐谱…" : "正在分析录音…";
  try {
    const endpoint = linkMode ? "/transcribe-link?" : "/transcribe?";
    const response = await fetch(endpoint + query, {method:"POST", headers:{"X-Sheet-Draw":"1"},
      body:linkMode ? el("media-url").value.trim() : file});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "生成失败，请重新选择录音。");
    currentTune = renderCombinedScore("score", data.abc, {}, data.tieEnds)[0];
    el("title").textContent = linkMode ? data.title : (file.name.replace(/\.[^.]+$/, "") || file.name);
    el("summary").textContent = optionText("key") + " 调 " + optionText("instrument") + "口琴演奏";
    el("analysis-info").textContent = optionText("subdivision") + "　·　" + data.count + " 个音符";
    if (el("source").value === "song") el("analysis-info").textContent +=
      "　·　人声旋律　·　整体移调 " + (data.shift > 0 ? "+" : "") + data.shift + " 半音　·　局部换八度 " + data.octaveChanges + " 音　·　无对应孔位 " + data.unplayable + " 音";
    el("export-actions").hidden = el("legend").hidden = false;
    el("empty").hidden = el("sample-score").hidden = true;
    el("kind").textContent = "生成结果";
    el("status").textContent = "已生成乐谱。";
    el("preview").scrollIntoView({block:"start",behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"instant":"smooth"});
  } catch (error) { el("status").textContent = error.message; }
  finally { form.removeAttribute("aria-busy"); for (const field of form.elements) field.disabled = false; el("media-url").disabled = !linkMode; }
};

const svgNamespace = "http://www.w3.org/2000/svg";
function svgNode(name, attributes) {
  const node = document.createElementNS(svgNamespace, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}
function assembleScore() {
  const sample = !el("sample-score").hidden;
  const systems = [...el(sample ? "sample-score" : "score").querySelectorAll("svg")];
  if (!systems.length) throw new Error("请先生成乐谱。");
  const margin = 24;
  // The shared renderer has already fitted the viewBox to the final annotation rows.
  const lines = systems.map(system => {
    const {x:left, y:top, width, height} = system.viewBox.baseVal;
    return {system, left, top, width, height};
  });
  const width = Math.max(760, ...lines.map(line => line.width)) + 2 * margin;
  const sheet = svgNode("svg", {width, "font-family": "sans-serif", color:"#182820", fill:"#182820"});
  const background = svgNode("rect", {width:"100%", height:"100%", fill:"white"});
  sheet.append(background);
  let y = 38;
  function addText(text, size, weight, centered = false) {
    const line = svgNode("text", {x:margin, y, "font-size":size, "font-weight":weight});
    if (centered) { line.setAttribute("text-anchor","middle"); line.setAttribute("x",width/2); }
    sheet.append(line);
    // Preserve even long source filenames without spilling past the page edge.
    const canvas = document.createElement("canvas").getContext("2d");
    canvas.font = weight + " " + size + "px sans-serif";
    let content = "";
    for (const character of text) {
      if (content && canvas.measureText(content + character).width > width - 2 * margin) {
        const part = svgNode("tspan", {x:centered ? width/2 : margin, y});
        part.textContent = content;
        line.append(part);
        y += size + 8;
        content = "";
      }
      content += character;
    }
    const last = svgNode("tspan", {x:centered ? width/2 : margin, y});
    last.textContent = content;
    line.append(last);
    y += size + (centered ? 1 : 12);
  }
  addText(el("title").textContent + (sample ? "（示例节选）" : ""), 26, 650, true);
  addText(el("summary").textContent, 15, 400, true);
  const breaks = [];
  for (const box of lines) {
    const system = box.system;
    const copy = system.cloneNode(true);
    copy.removeAttribute("style");
    copy.setAttribute("viewBox", [box.left,box.top,box.width,box.height].join(" "));
    for (const [key, value] of Object.entries({x:margin, y, width:box.width, height:box.height})) copy.setAttribute(key, value);
    sheet.append(copy);
    y += box.height;
    if (box.height > 0) breaks.push(y);
  }
  y += 30;
  addText(el("legend").textContent, 13, 400);
  const height = Math.ceil(y + margin);
  sheet.setAttribute("height", height);
  sheet.setAttribute("viewBox", "0 0 " + width + " " + height);
  const blob = new Blob([new XMLSerializer().serializeToString(sheet)], {type:"image/svg+xml;charset=utf-8"});
  // Keep the heading with the first staff and the legend with the last staff.
  breaks.pop();
  breaks.push(height);
  return {blob, width, height, breaks, sheet};
}
async function rasterize({blob, width, height}, mime = "image/png", density = 2) {
  // Bound pixel memory for long recordings while retaining every system.
  const scale = Math.min(density, 16000 / height, Math.sqrt(24000000 / (width * height)));
  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil(width * scale);
  canvas.height = Math.ceil(height * scale);
  const image = new Image();
  const url = URL.createObjectURL(blob);
  try {
    image.src = url;
    await image.decode();
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    return await new Promise((resolve, reject) => canvas.toBlob(
      result => result ? resolve(result) : reject(new Error("图像生成失败，请尝试下载 SVG。")), mime, .95));
  } finally { URL.revokeObjectURL(url); }
}
// Load the locally bundled writer only on the first PDF download.
function loadPdfLibrary() {
  if (!pdfLibrary) {
    pdfLibrary = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/pdf-lib.min.js";
      script.onload = () => resolve(window.PDFLib);
      script.onerror = () => { pdfLibrary = null; script.remove(); reject(new Error("PDF 组件加载失败，请重试。")); };
      document.head.append(script);
    });
  }
  return pdfLibrary;
}

async function makePdf(score, title) {
  const {PDFDocument, PageSizes} = await loadPdfLibrary();
  const doc = await PDFDocument.create();
  doc.setTitle(title);
  const [w, h] = PageSizes.A4;
  const margin = 28;
  const capacity = score.width * (h-2*margin) / (w-2*margin);
  let start = 0, previous = 0;
  const ends = [];
  for (const end of score.breaks) {
    if (end-start > capacity && previous > start) { ends.push(previous); start = previous; }
    previous = end;
  }
  ends.push(score.height);
  start = 0;
  for (const end of ends) {
    const height = end-start;
    const part = score.sheet.cloneNode(false);
    part.setAttribute("height",height);
    part.setAttribute("viewBox", "0 " + start + " " + score.width + " " + height);
    for (const node of score.sheet.children) {
      const y = Number(node.getAttribute("y") || 0);
      if (node.tagName === "rect" || (y >= start && y < end))
        part.append(node.cloneNode(true));
    }
    const blob = new Blob([new XMLSerializer().serializeToString(part)], {type:"image/svg+xml"});
    const png = await rasterize({blob, width:score.width, height}, "image/png", 2.6);
    const image = await doc.embedPng(await png.arrayBuffer());
    const factor = Math.min((w-2*margin)/score.width, (h-2*margin)/height);
    doc.addPage([w,h]).drawImage(image, {x:(w-score.width*factor)/2, y:h-margin-height*factor,
                                        width:score.width*factor, height:height*factor});
    start = end;
  }
  return new Blob([await doc.save()], {type:"application/pdf"});
}

for (const format of ["pdf", "svg", "png", "jpeg", "midi"]) {
  el(format).onclick = async () => {
    const button = el(format);
    dismissExportMenu();
    button.disabled = true;
    el("export-status").textContent = "";
    try {
      const stem = el("title").textContent.replace(/[\\/:*?"<>|]/g,"_") +
        (el("sample-score").hidden ? "" : "-示例节选");
      let blob;
      if (format === "midi") {
        const bytes = ABCJS.synth.getMidiFile(currentTune, {midiOutputType:"binary", program:22, chordsOff:true});
        blob = new Blob([bytes],{type:"audio/midi"});
      } else {
        const score = assembleScore();
        if (format === "pdf") blob = await makePdf(score,stem);
        else if (format === "svg") blob = score.blob;
        else blob = await rasterize(score,format === "jpeg" ? "image/jpeg" : "image/png");
      }
      const anchor = document.createElement("a");
      const url = URL.createObjectURL(blob);
      anchor.href = url;
      anchor.download = stem + "-口琴谱." + (format === "midi" ? "mid" : format);
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
    } catch (error) { el("export-status").textContent = error.message; }
    finally { button.disabled = false; }
  };
}

const examples = JSON.parse(el("examples").textContent);
function showSample() {
  const name = el("sample").value;
  const meta = {"蓝莲花":"A 调 Paddy 十孔口琴演奏", "红尘客栈":"C 调十二孔半音阶口琴演奏", "我不难过":"C 调十二孔半音阶口琴演奏"};
  el("title").textContent = name;
  el("summary").textContent = meta[name];
  el("analysis-info").textContent = "";
  currentTune = renderCombinedScore("sample-score", examples[name], {staffwidth:740})[0];
  el("export-status").textContent = "";
}
el("sample").addEventListener("change",showSample);
showSample();
el("help").onclick = () => el("guide").showModal();
el("file").addEventListener("change", () => {
  el("file-name").textContent = el("file").files[0]?.name || "选择或拖入音频";
});
for (const e of ["dragenter","dragleave","drop"]) el("drop").addEventListener(e,() => {
  el("drop").classList.toggle("drag",e==="dragenter");
});

// The input stage moves from a waveform on the left to notation on the right.
function mountStaffScene(canvas) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const stage = canvas.parentElement;
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  let width = 0, height = 0, frame = 0, last = 0, phase = 0, visible = true;
  let aim = 0, tilt = 0;
  function draw() {
    ctx.clearRect(0, 0, width, height);
    const wing = Math.min(width * .38, 590);
    const level = Math.min(310, height * .56);
    const amp = Math.min(1,width/920);
    function leftY(u) {
      return level + amp*(30*Math.sin(u*6+phase*.25) + 22*Math.sin(u*10-phase*.19)) + tilt*6;
    }
    const fade = ctx.createLinearGradient(0,0,wing,0);
    fade.addColorStop(0,"#16917855"); fade.addColorStop(.44,"#158a6599"); fade.addColorStop(1,"#22846100");
    ctx.strokeStyle=fade; ctx.lineWidth=2; ctx.lineCap="round";
    for (let x=1;x<wing;x+=5) {
      const u=x/wing;
      const envelope=(.2+.8*Math.pow(Math.abs(Math.sin(u*5.8+.8)),2))*(1-u*.55);
      const h=amp*(4+52*envelope*(.4+.6*Math.abs(Math.sin(x*.073-phase))));
      const y=leftY(u);ctx.beginPath();ctx.moveTo(x,y-h);ctx.lineTo(x,y+h);ctx.stroke();
    }
    function staffY(u) {
      return level+amp*(55*Math.sin(u*5+phase*.16)-32*Math.sin(u*8-phase*.12))-tilt*6;
    }
    const start=width-wing, spacing=8*amp;
    const ink=ctx.createLinearGradient(start,0,width,0);
    ink.addColorStop(0,"#56836600");ink.addColorStop(.36,"#56836652");ink.addColorStop(1,"#56836685");
    ctx.lineWidth=1;ctx.strokeStyle=ink;
    for(let line=-2;line<=2;line++){
      ctx.beginPath();
      for(let x=0;x<=wing+3;x+=3){
        const y=staffY(x/wing)+line*spacing;
        if(!x)ctx.moveTo(start+x,y);else ctx.lineTo(start+x,y);
      }
      ctx.stroke();
    }
    const notes=[1,-1,0,-3,1,2,0,-2,1];
    notes.forEach((pitch,i)=>{
      const u=(i/notes.length+phase*.014)%1;
      const x=start+wing*u,y=staffY(u)+pitch*spacing*.5;
      ctx.globalAlpha=Math.min(1,u*3)*(width<680?.5:.77);
      ctx.fillStyle="#39816b";ctx.strokeStyle="#39816b";ctx.lineWidth=1.5;
      ctx.beginPath();ctx.ellipse(x,y,5.2*amp,3.7*amp,-.34,0,Math.PI*2);ctx.fill();
      ctx.beginPath();ctx.moveTo(x+4.6*amp,y);ctx.lineTo(x+4.6*amp,y-25*amp);ctx.stroke();
    });
    ctx.globalAlpha=1;
  }
  function tick(now) {
    if (now-last >= 40) {
      phase+=Math.min(now-last,80)/1000; tilt+=(aim-tilt)*.12;
      draw();last=now;
    }
    frame=requestAnimationFrame(tick);
  }
  function sync() {
    cancelAnimationFrame(frame);
    draw();
    if (!motion.matches && visible && !document.hidden) {
      last=performance.now();frame=requestAnimationFrame(tick);
    }
  }
  new ResizeObserver(()=>{
    width=stage.clientWidth;height=stage.clientHeight;
    const ratio=Math.min(window.devicePixelRatio||1,2);
    canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
    ctx.setTransform(ratio,0,0,ratio,0,0);sync();
  }).observe(stage);
  new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;sync();}).observe(stage);
  stage.addEventListener("pointermove",e=>{aim=(e.clientY-stage.getBoundingClientRect().top)/height-.5;},{passive:true});
  stage.addEventListener("pointerleave",()=>{aim=0;});
  motion.addEventListener("change",sync);
  document.addEventListener("visibilitychange",sync);
}
mountStaffScene(el("staff-scene"));

// The select remains the form's source of truth. A shared listbox lives outside
// the studio card, so expanded menus can never be clipped by its animation layer.
function installPickers() {
  const menu = document.createElement("div");
  menu.className = "picker-menu";
  menu.id = "picker-options";
  menu.setAttribute("role", "listbox");
  menu.hidden = true;
  document.body.append(menu);
  let openPicker = null;
  let active = -1;
  let prefix = "", lastKey = 0;

  function close() {
    if (!openPicker) return;
    openPicker.button.setAttribute("aria-expanded", "false");
    openPicker.button.removeAttribute("aria-activedescendant");
    menu.hidden = true;
    openPicker = null;
    prefix = "";
  }

  function place() {
    if (!openPicker) return;
    const rect = openPicker.button.getBoundingClientRect();
    if (!rect.width || rect.bottom < 0 || rect.top > innerHeight) return close();
    const margin = 8, gap = 5;
    const below = innerHeight - rect.bottom - margin - gap;
    const above = rect.top - margin - gap;
    const width = Math.min(Math.max(rect.width, menu.scrollWidth, 165), innerWidth - margin * 2);
    menu.style.width = width + "px";
    const upward = below < Math.min(menu.scrollHeight, 260) && above > below;
    menu.style.maxHeight = Math.min(300, Math.max(32, upward ? above : below)) + "px";
    menu.style.left = Math.max(margin, Math.min(rect.left, innerWidth-width-margin)) + "px";
    menu.style.top = (upward ? rect.top - gap - menu.offsetHeight : rect.bottom + gap) + "px";
  }

  function highlight(i, scroll) {
    if (i < 0 || !menu.children[i] || openPicker.select.options[i].disabled) return;
    menu.children[active]?.classList.remove("active");
    active = i;
    const item = menu.children[i];
    item.classList.add("active");
    openPicker.button.setAttribute("aria-activedescendant", item.id);
    if (scroll) {
      if (item.offsetTop < menu.scrollTop) menu.scrollTop = item.offsetTop;
      if (item.offsetTop + item.offsetHeight > menu.scrollTop + menu.clientHeight)
        menu.scrollTop = item.offsetTop + item.offsetHeight - menu.clientHeight;
    }
  }

  function show(picker) {
    close();
    openPicker = picker;
    active = -1;
    const {select, button, name} = picker;
    menu.setAttribute("aria-label", name);
    menu.replaceChildren(...Array.from(select.options, (option, i) => {
      const row = document.createElement("div");
      row.id = "picker-option-" + i;
      row.setAttribute("role", "option");
      row.setAttribute("aria-selected", String(option.selected));
      row.setAttribute("aria-disabled", String(option.disabled));
      row.textContent = option.textContent;
      return row;
    }));
    menu.style.width = "max-content";
    menu.style.maxHeight = "none";
    menu.hidden = false;
    button.setAttribute("aria-expanded", "true");
    place();
    highlight(select.selectedIndex, true);
  }

  function choose(index) {
    const {select,button} = openPicker;
    if (select.options[index]?.disabled) return;
    close();
    if (select.selectedIndex !== index) {
      select.selectedIndex = index;
      select.dispatchEvent(new Event("input", {bubbles:true}));
      select.dispatchEvent(new Event("change", {bubbles:true}));
    }
    button.focus({preventScroll:true});
  }

  for (const select of document.querySelectorAll(".fields select, #sample")) {
    const label = select.labels[0];
    const name = Array.from(label.childNodes).filter(n => n.nodeType === Node.TEXT_NODE)
      .map(n => n.textContent.trim()).join(" ").trim();
    const button = document.createElement("button");
    button.type = "button";
    button.className = "picker-trigger";
    button.id = select.id + "-picker";
    button.setAttribute("role","combobox");
    button.setAttribute("aria-controls",menu.id);
    button.setAttribute("aria-haspopup","listbox");
    button.setAttribute("aria-expanded","false");
    const value = document.createElement("span");
    button.append(value);
    select.after(button);
    label.htmlFor = button.id;
    select.hidden = true;
    const picker = {select,button,name};
    function sync() {
      value.textContent = select.selectedOptions[0]?.textContent || "";
      button.setAttribute("aria-label", name + "，" + value.textContent);
      button.disabled = select.disabled;
      if (button.disabled && openPicker === picker) close();
    }
    sync();
    select.addEventListener("change",sync);
    new MutationObserver(sync).observe(select,{attributes:true,attributeFilter:["disabled"], subtree:true,childList:true});
    button.addEventListener("click", () => openPicker === picker ? close() : show(picker));
    button.addEventListener("blur",close);
    button.addEventListener("keydown",e => {
      if (e.key === "Tab" || e.key === "Escape") {
        if (openPicker && e.key === "Escape") e.preventDefault();
        return close();
      }
      const enabled = Array.from(select.options, (o,i) => o.disabled ? -1 : i).filter(i => i >= 0);
      if (!enabled.length || e.ctrlKey || e.metaKey) return;
      if (["Enter"," "].includes(e.key)) {
        e.preventDefault();
        if (openPicker === picker && active >= 0) choose(active); else show(picker);
      } else if (["ArrowDown","ArrowUp","Home","End"].includes(e.key)) {
        e.preventDefault(); if (openPicker !== picker) show(picker);
        const n = enabled.indexOf(active);
        const j = e.key === "Home" ? 0 : e.key === "End" ? enabled.length-1 :
          Math.max(0, Math.min(enabled.length-1, n + (e.key === "ArrowDown" ? 1 : -1)));
        highlight(enabled[j],true);
      } else if (e.key.length === 1 && !e.isComposing) {
        if (openPicker !== picker) show(picker);
        const now = performance.now();
        prefix = (now-lastKey < 650 ? prefix : "") + e.key.toLowerCase(); lastKey = now;
        const i = enabled.find(i => select.options[i].textContent.trim().toLowerCase().startsWith(prefix));
        if (i !== undefined) highlight(i,true);
      }
    });
  }
  // Keep keyboard focus on the combobox while choosing or scrolling its menu.
  menu.addEventListener("mousedown",e => e.preventDefault());
  menu.addEventListener("click",e => {
    const row = e.target.closest('[role="option"]');
    if (row && openPicker) choose(Array.prototype.indexOf.call(menu.children,row));
  });
  document.addEventListener("pointerdown",e => {
    if (openPicker && !openPicker.button.contains(e.target) && !menu.contains(e.target)) close();
  },true);
  window.addEventListener("resize",place);
  document.addEventListener("scroll",e => { if (e.target !== menu) place(); },true);
}
installPickers();
