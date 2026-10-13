const SPAN_X = 4 * Math.SQRT2;
const SPAN_Y = 2 * Math.SQRT2;

const state = {
  domain: "sky",
  band: "ultraviolet",
  estimates: true,
  provenance: false,
  grid: true,
  placeId: "centre",
  view: { x: 0, y: 0, scale: 1 },
  fittedScale: 1,
};

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const readout = document.getElementById("readout");
const closeup = document.getElementById("closeup");
const closeupCaption = document.getElementById("closeup-caption");
const closeupFigure = document.getElementById("closeup-figure");
const placesEl = document.getElementById("places");

let meta = null;
const images = new Map();
let provCanvas = null;
let drag = null;
let lastPointer = null;

function mollweideTheta(phi) {
  const polar = Math.abs(Math.sin(phi)) > 0.999999;
  let theta = polar ? Math.sign(Math.sin(phi)) * (Math.PI / 2) : phi;
  if (polar) return theta;
  const target = Math.PI * Math.sin(phi);
  for (let i = 0; i < 8; i += 1) {
    const deriv = 2 + 2 * Math.cos(2 * theta);
    theta -= (2 * theta + Math.sin(2 * theta) - target) / (Math.abs(deriv) < 1e-8 ? 1 : deriv);
  }
  return theta;
}

function project(longitude, latitude, increases) {
  let lon = longitude;
  if (lon > 180) lon -= 360;
  const lam = ((increases === "left" ? -lon : lon) * Math.PI) / 180;
  const theta = mollweideTheta((latitude * Math.PI) / 180);
  return {
    x: ((2 * Math.SQRT2) / Math.PI) * lam * Math.cos(theta),
    y: Math.SQRT2 * Math.sin(theta),
  };
}

function toImage(longitude, latitude) {
  const increases = state.domain === "sky" ? "left" : "right";
  const point = project(longitude, latitude, increases);
  const usable = 1 - 2 * meta.pad;
  return {
    x: (0.5 + (point.x * usable) / SPAN_X) * meta.width - 0.5,
    y: (0.5 - (point.y * usable) / SPAN_Y) * meta.height - 0.5,
  };
}

function fromImage(ix, iy) {
  const increases = state.domain === "sky" ? "left" : "right";
  const usable = 1 - 2 * meta.pad;
  const x = ((ix + 0.5) / meta.width - 0.5) * (SPAN_X / usable);
  const y = (0.5 - (iy + 0.5) / meta.height) * (SPAN_Y / usable);
  const inside = (x / (2 * Math.SQRT2)) ** 2 + (y / Math.SQRT2) ** 2 <= 1;
  const theta = Math.asin(Math.max(-1, Math.min(1, y / Math.SQRT2)));
  const latitude = (Math.asin(Math.max(-1, Math.min(1, (2 * theta + Math.sin(2 * theta)) / Math.PI))) * 180) / Math.PI;
  const cosTheta = Math.cos(theta);
  const plot = Math.abs(cosTheta) > 1e-8
    ? ((Math.PI * x) / (2 * Math.SQRT2 * cosTheta)) * (180 / Math.PI)
    : 0;
  let longitude = increases === "left" ? -plot : plot;
  longitude = ((((longitude + 180) % 360) + 360) % 360) - 180;
  return { longitude, latitude, inside };
}

function loadImage(url) {
  if (images.has(url)) return images.get(url);
  const promise = new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(url));
    image.src = url;
  });
  images.set(url, promise);
  return promise;
}

function currentPlateUrl() {
  const group = meta.assets[state.domain];
  if (state.domain === "sky" && state.band === "ultraviolet" && !state.estimates) {
    return group.ultravioletMeasured;
  }
  return group[state.band];
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function fit() {
  resize();
  const scale = Math.min(canvas.clientWidth / meta.width, canvas.clientHeight / meta.height) * 0.98;
  state.fittedScale = scale;
  state.view.scale = scale;
  state.view.x = (canvas.clientWidth - meta.width * scale) / 2;
  state.view.y = (canvas.clientHeight - meta.height * scale) / 2;
}

function centerOn(longitude, latitude, zoom) {
  fit();
  const point = toImage(longitude, latitude);
  state.view.scale = state.fittedScale * zoom;
  state.view.x = canvas.clientWidth / 2 - point.x * state.view.scale;
  state.view.y = canvas.clientHeight / 2 - point.y * state.view.scale;
  draw();
}

async function draw() {
  resize();
  const plate = await loadImage(currentPlateUrl());
  const { x, y, scale } = state.view;
  ctx.setTransform(Math.min(window.devicePixelRatio || 1, 2), 0, 0, Math.min(window.devicePixelRatio || 1, 2), 0, 0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.imageSmoothingEnabled = scale < 2;
  ctx.drawImage(plate, x, y, meta.width * scale, meta.height * scale);
  if (state.provenance && state.domain === "sky") {
    const overlay = await loadImage(meta.assets.sky.provenance);
    ctx.drawImage(overlay, x, y, meta.width * scale, meta.height * scale);
  }
  if (state.grid) drawGrid();
  drawOval();
}

function imageToScreen(ix, iy) {
  return {
    x: state.view.x + ix * state.view.scale,
    y: state.view.y + iy * state.view.scale,
  };
}

function drawOval() {
  const usable = 1 - 2 * meta.pad;
  const center = imageToScreen(meta.width / 2 - 0.5, meta.height / 2 - 0.5);
  ctx.beginPath();
  ctx.ellipse(
    center.x,
    center.y,
    (usable * meta.width * state.view.scale) / 2,
    (usable * meta.height * state.view.scale) / 2,
    0,
    0,
    Math.PI * 2,
  );
  ctx.strokeStyle = "rgba(244,239,230,0.35)";
  ctx.lineWidth = 1;
  ctx.stroke();
}

function drawGrid() {
  const longitudes = state.domain === "sky"
    ? [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
    : [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150];
  const latitudes = [-60, -30, 0, 30, 60];
  ctx.strokeStyle = "rgba(244,239,230,0.28)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  latitudes.forEach((latitude) => trace(longitudes[0], longitudes[longitudes.length - 1], latitude, true));
  longitudes.forEach((longitude) => trace(-75, 75, longitude, false));
  ctx.stroke();
}

function trace(start, end, fixed, parallel) {
  let previous = null;
  const steps = 80;
  for (let step = 0; step <= steps; step += 1) {
    const t = start + ((end - start) * step) / steps;
    const longitude = parallel ? t : fixed;
    const latitude = parallel ? fixed : t;
    const image = toImage(longitude, latitude);
    const screen = imageToScreen(image.x, image.y);
    if (previous && Math.hypot(screen.x - previous.x, screen.y - previous.y) < 40) {
      ctx.lineTo(screen.x, screen.y);
    } else {
      ctx.moveTo(screen.x, screen.y);
    }
    previous = screen;
  }
}

function formatAngle(value, positive, negative) {
  const hemisphere = value >= 0 ? positive : negative;
  return `${Math.abs(value).toFixed(1)}° ${hemisphere}`;
}

function provenanceLabel(value) {
  if (value >= 200) return "estimated from 12 µm dust";
  if (value >= 120) return "GALEX near-UV only; far-UV estimated on the full plate";
  if (value >= 40) return "GALEX far-UV and near-UV";
  return "no GALEX coverage in this pixel";
}

function pointerToImage(event) {
  const rect = canvas.getBoundingClientRect();
  const mx = event.clientX - rect.left;
  const my = event.clientY - rect.top;
  return {
    ix: (mx - state.view.x) / state.view.scale,
    iy: (my - state.view.y) / state.view.scale,
  };
}

function refreshReadout() {
  if (lastPointer) updateReadout(lastPointer);
}

async function updateReadout(event) {
  lastPointer = event;
  const { ix, iy } = pointerToImage(event);
  if (ix < 0 || iy < 0 || ix >= meta.width || iy >= meta.height) {
    readout.textContent = "Outside the map.";
    return;
  }
  const sky = fromImage(ix, iy);
  if (!sky.inside) {
    readout.textContent = "Outside the oval.";
    return;
  }
  if (state.domain === "earth") {
    readout.textContent = `Earth  ${formatAngle(sky.longitude, "E", "W")}, ${formatAngle(sky.latitude, "N", "S")}`;
    return;
  }
  let source = "";
  if (provCanvas) {
    const pixel = provCanvas.getContext("2d").getImageData(Math.round(ix), Math.round(iy), 1, 1).data;
    source = ` · ${provenanceLabel(pixel[0])}`;
  }
  const longitude = sky.longitude < 0 ? sky.longitude + 360 : sky.longitude;
  readout.textContent = `Galactic  l ${longitude.toFixed(1)}°, b ${sky.latitude.toFixed(1)}°${source}`;
}

function zoomAt(cssX, cssY, factor) {
  const next = Math.min(state.fittedScale * 16, Math.max(state.fittedScale * 0.85, state.view.scale * factor));
  const ix = (cssX - state.view.x) / state.view.scale;
  const iy = (cssY - state.view.y) / state.view.scale;
  state.view.scale = next;
  state.view.x = cssX - ix * next;
  state.view.y = cssY - iy * next;
  draw();
}

function activePlaces() {
  return state.domain === "sky" ? meta.places : meta.earthPlaces;
}

function renderPlaces() {
  placesEl.replaceChildren();
  activePlaces().forEach((place) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = place.name;
    button.setAttribute("aria-pressed", String(place.id === state.placeId));
    button.addEventListener("click", () => selectPlace(place));
    placesEl.appendChild(button);
  });
}

function selectPlace(place) {
  state.placeId = place.id;
  renderPlaces();
  centerOn(place.longitude, place.latitude, place.zoom || 3);
  updateCloseup();
}

function updateCloseup() {
  const skyPlace = meta.places.find((place) => place.id === state.placeId) || meta.places[0];
  if (state.domain !== "sky") {
    closeupFigure.hidden = true;
    return;
  }
  closeupFigure.hidden = false;
  const sources = meta.assets.sky.closeups[skyPlace.id];
  closeup.src = sources[state.band];
  closeup.alt = `${skyPlace.name} in ${state.band} light`;
  closeupCaption.textContent = `${skyPlace.name}. ${skyPlace.blurb || ""}`;
}

function setPressed(selector, attr, value) {
  document.querySelectorAll(selector).forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset[attr] === value));
  });
}

function bindControls() {
  document.querySelectorAll("[data-domain]").forEach((button) => {
    button.addEventListener("click", () => {
      state.domain = button.dataset.domain;
      setPressed("[data-domain]", "domain", state.domain);
      if (!activePlaces().some((place) => place.id === state.placeId)) {
        state.placeId = activePlaces()[0].id;
      }
      renderPlaces();
      updateCloseup();
      fit();
      draw();
      refreshReadout();
    });
  });
  document.querySelectorAll("[data-band]").forEach((button) => {
    button.addEventListener("click", () => {
      state.band = button.dataset.band;
      setPressed("[data-band]", "band", state.band);
      updateCloseup();
      draw();
    });
  });
  document.getElementById("estimates").addEventListener("click", (event) => {
    state.estimates = !state.estimates;
    event.currentTarget.setAttribute("aria-pressed", String(state.estimates));
    draw();
  });
  document.getElementById("provenance").addEventListener("click", (event) => {
    state.provenance = !state.provenance;
    event.currentTarget.setAttribute("aria-pressed", String(state.provenance));
    draw();
  });
  document.getElementById("grid").addEventListener("click", (event) => {
    state.grid = !state.grid;
    event.currentTarget.setAttribute("aria-pressed", String(state.grid));
    draw();
  });
  document.getElementById("zoom-in").addEventListener("click", () => {
    zoomAt(canvas.clientWidth / 2, canvas.clientHeight / 2, 1.3);
  });
  document.getElementById("zoom-out").addEventListener("click", () => {
    zoomAt(canvas.clientWidth / 2, canvas.clientHeight / 2, 1 / 1.3);
  });
  document.getElementById("reset").addEventListener("click", () => {
    fit();
    draw();
  });

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    zoomAt(event.clientX - rect.left, event.clientY - rect.top, event.deltaY < 0 ? 1.12 : 1 / 1.12);
  }, { passive: false });

  canvas.addEventListener("pointerdown", (event) => {
    canvas.setPointerCapture(event.pointerId);
    drag = { x: event.clientX, y: event.clientY, vx: state.view.x, vy: state.view.y };
  });
  canvas.addEventListener("pointermove", (event) => {
    if (drag) {
      state.view.x = drag.vx + (event.clientX - drag.x);
      state.view.y = drag.vy + (event.clientY - drag.y);
      draw();
    } else {
      updateReadout(event);
    }
  });
  canvas.addEventListener("pointerup", () => {
    drag = null;
  });
  canvas.addEventListener("pointerleave", () => {
    if (!drag) readout.textContent = "Move across the map for coordinates.";
  });

  window.addEventListener("keydown", (event) => {
    if (event.target.matches("input, textarea")) return;
    const bands = { Digit1: "ultraviolet", Digit2: "visible", Digit3: "infrared" };
    if (bands[event.code]) {
      state.band = bands[event.code];
      setPressed("[data-band]", "band", state.band);
      updateCloseup();
      draw();
    } else if (event.key === "s") {
      state.domain = "sky";
      setPressed("[data-domain]", "domain", state.domain);
      if (!activePlaces().some((place) => place.id === state.placeId)) {
        state.placeId = activePlaces()[0].id;
      }
      renderPlaces();
      updateCloseup();
      draw();
      refreshReadout();
    } else if (event.key === "e") {
      state.domain = "earth";
      setPressed("[data-domain]", "domain", state.domain);
      if (!activePlaces().some((place) => place.id === state.placeId)) {
        state.placeId = activePlaces()[0].id;
      }
      renderPlaces();
      updateCloseup();
      draw();
      refreshReadout();
    } else if (event.key === "0") {
      fit();
      draw();
    }
  });
  window.addEventListener("resize", () => {
    fit();
    draw();
  });
}

async function main() {
  const response = await fetch("maps/meta.json");
  meta = await response.json();
  const code = await loadImage(meta.assets.sky.provenanceCode);
  provCanvas = document.createElement("canvas");
  provCanvas.width = code.width;
  provCanvas.height = code.height;
  provCanvas.getContext("2d").drawImage(code, 0, 0);
  bindControls();
  renderPlaces();
  updateCloseup();
  fit();
  await draw();
}

main().catch((error) => {
  readout.textContent = `The map failed to load (${error.message}). Serve this folder over HTTP.`;
});
