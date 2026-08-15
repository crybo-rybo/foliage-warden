import {
  ZONE_TYPES,
  buildScene,
  normalizeCanvasPoint,
  validateId,
  validatePolygon,
} from "./geometry.mjs";

const elements = {
  calibrationId: document.querySelector("#calibration-id"),
  canvas: document.querySelector("#scene"),
  copy: document.querySelector("#copy"),
  configInput: document.querySelector("#config-input"),
  download: document.querySelector("#download"),
  empty: document.querySelector("#empty-message"),
  finish: document.querySelector("#finish"),
  hardwareTarget: document.querySelector("#hardware-target"),
  hardwareTargetRow: document.querySelector("#hardware-target-row"),
  imageInput: document.querySelector("#image-input"),
  output: document.querySelector("#json-output"),
  remove: document.querySelector("#remove"),
  shapeId: document.querySelector("#shape-id"),
  shapeList: document.querySelector("#shape-list"),
  shapeType: document.querySelector("#shape-type"),
  status: document.querySelector("#status"),
  undo: document.querySelector("#undo"),
  zoneLink: document.querySelector("#zone-link"),
  zoneLinkRow: document.querySelector("#zone-link-row"),
};

const context = elements.canvas.getContext("2d");
const colors = {
  approach: "#69d58b",
  foliage: "#43b7a5",
  soil: "#d69a59",
  no_fire: "#ff6868",
  aim_preset: "#ffd85e",
};
const state = {
  aimPresets: [],
  baseConfig: { schema_version: 1 },
  currentPoints: [],
  image: null,
  imageMetadata: { name: "unselected", width: 1280, height: 720 },
  zones: [],
};

function setStatus(message, error = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("error", error);
}

function allIds() {
  return new Set([...state.zones, ...state.aimPresets].map(({ id }) => id));
}

function denormalize(point) {
  return { x: point.x * elements.canvas.width, y: point.y * elements.canvas.height };
}

function drawPolygon(points, color, fill = true) {
  if (points.length === 0) return;
  context.beginPath();
  const first = denormalize(points[0]);
  context.moveTo(first.x, first.y);
  for (const point of points.slice(1)) {
    const mapped = denormalize(point);
    context.lineTo(mapped.x, mapped.y);
  }
  if (points.length > 2) context.closePath();
  context.strokeStyle = color;
  context.lineWidth = Math.max(2, elements.canvas.width / 500);
  context.stroke();
  if (fill && points.length > 2) {
    context.fillStyle = `${color}2d`;
    context.fill();
  }
  for (const point of points) {
    const mapped = denormalize(point);
    context.beginPath();
    context.arc(mapped.x, mapped.y, Math.max(3, elements.canvas.width / 320), 0, Math.PI * 2);
    context.fillStyle = color;
    context.fill();
  }
}

function drawAim(preset) {
  const point = denormalize(preset.point);
  const radius = Math.max(9, elements.canvas.width / 80);
  context.strokeStyle = colors.aim_preset;
  context.lineWidth = Math.max(2, elements.canvas.width / 500);
  context.beginPath();
  context.moveTo(point.x - radius, point.y);
  context.lineTo(point.x + radius, point.y);
  context.moveTo(point.x, point.y - radius);
  context.lineTo(point.x, point.y + radius);
  context.stroke();
}

function draw() {
  context.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
  if (state.image) context.drawImage(state.image, 0, 0, elements.canvas.width, elements.canvas.height);
  for (const zone of state.zones) drawPolygon(zone.points, colors[zone.type]);
  for (const preset of state.aimPresets) drawAim(preset);
  drawPolygon(state.currentPoints, colors[elements.shapeType.value] ?? "#ffffff", false);
}

function exportDocument() {
  const calibrationId = elements.calibrationId.value.trim();
  const scene = buildScene({
    calibrationId,
    zones: state.zones,
    aimPresets: state.aimPresets,
  });
  return { ...state.baseConfig, schema_version: state.baseConfig.schema_version ?? 1, scene };
}

function render() {
  const isAim = elements.shapeType.value === "aim_preset";
  elements.finish.disabled = isAim || state.currentPoints.length < 3;
  elements.undo.disabled = state.currentPoints.length === 0;
  elements.remove.disabled = state.zones.length + state.aimPresets.length === 0;
  elements.zoneLinkRow.hidden = !isAim;
  elements.hardwareTargetRow.hidden = !isAim;
  elements.zoneLink.replaceChildren(
    ...state.zones
      .filter((zone) => zone.type === "approach")
      .map((zone) => new Option(`${zone.id} (${zone.type})`, zone.id)),
  );
  elements.shapeList.replaceChildren(
    ...[...state.zones, ...state.aimPresets].map((shape) => {
      const item = document.createElement("li");
      item.textContent = `${shape.id} — ${shape.type ?? "aim preset"}`;
      return item;
    }),
  );
  elements.output.value = `${JSON.stringify(exportDocument(), null, 2)}\n`;
  draw();
}

function finishPolygon() {
  try {
    const id = elements.shapeId.value.trim();
    validateId(id, allIds());
    validatePolygon(state.currentPoints);
    const type = elements.shapeType.value;
    if (!ZONE_TYPES.has(type)) throw new Error("select a polygon shape type");
    state.zones.push({ id, type, points: structuredClone(state.currentPoints) });
    state.currentPoints = [];
    setStatus(`Added ${type} zone ${id}.`);
    render();
  } catch (error) {
    setStatus(error.message, true);
  }
}

elements.canvas.addEventListener("click", (event) => {
  if (event.detail > 1) return;
  if (!state.image) return setStatus("Choose a reference image first.", true);
  const point = normalizeCanvasPoint(event.clientX, event.clientY, elements.canvas.getBoundingClientRect());
  if (elements.shapeType.value !== "aim_preset") {
    state.currentPoints.push(point);
    setStatus(`${state.currentPoints.length} polygon point(s).`);
    return render();
  }
  try {
    const id = elements.shapeId.value.trim();
    validateId(id, allIds());
    if (!elements.zoneLink.value) throw new Error("add and select a related zone first");
    state.aimPresets.push({
      id,
      zone_id: elements.zoneLink.value,
      point,
      hardware_target: elements.hardwareTarget.value.trim() || "disabled",
    });
    setStatus(`Added safe aim preset ${id}.`);
    render();
  } catch (error) {
    setStatus(error.message, true);
  }
});

elements.canvas.addEventListener("dblclick", (event) => {
  event.preventDefault();
  if (elements.shapeType.value !== "aim_preset" && state.currentPoints.length >= 3) finishPolygon();
});

elements.finish.addEventListener("click", finishPolygon);
elements.undo.addEventListener("click", () => {
  state.currentPoints.pop();
  render();
});
elements.remove.addEventListener("click", () => {
  if (state.aimPresets.length > 0) state.aimPresets.pop();
  else state.zones.pop();
  setStatus("Removed the most recently added shape.");
  render();
});
elements.shapeType.addEventListener("change", () => {
  state.currentPoints = [];
  render();
});
elements.calibrationId.addEventListener("input", render);

elements.imageInput.addEventListener("change", () => {
  const [file] = elements.imageInput.files;
  if (!file) return;
  const image = new Image();
  image.addEventListener("load", () => {
    state.image = image;
    state.imageMetadata = { name: file.name, width: image.naturalWidth, height: image.naturalHeight };
    elements.canvas.width = image.naturalWidth;
    elements.canvas.height = image.naturalHeight;
    elements.empty.hidden = true;
    setStatus("Image loaded locally. Add calibration shapes.");
    render();
  });
  image.src = URL.createObjectURL(file);
});

elements.configInput.addEventListener("change", async () => {
  const [file] = elements.configInput.files;
  if (!file) return;
  try {
    const parsed = JSON.parse(await file.text());
    const scene = parsed.scene ?? parsed;
    state.baseConfig = parsed.scene ? parsed : { schema_version: parsed.schema_version ?? 1 };
    state.zones = structuredClone(scene.zones ?? []);
    state.aimPresets = structuredClone(scene.aim_presets ?? []);
    elements.calibrationId.value = scene.calibration_id ?? "living_room_v1";
    state.currentPoints = [];
    setStatus(`Imported ${state.zones.length} zones and ${state.aimPresets.length} aim presets.`);
    render();
  } catch (error) {
    setStatus(`Cannot import config: ${error.message}`, true);
  }
});

elements.copy.addEventListener("click", async () => {
  await navigator.clipboard.writeText(elements.output.value);
  setStatus("Scene JSON copied.");
});
elements.download.addEventListener("click", () => {
  const blob = new Blob([elements.output.value], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "scene-calibration.json";
  link.click();
  URL.revokeObjectURL(link.href);
});

render();
