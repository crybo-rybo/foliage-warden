export const ZONE_TYPES = new Set(["approach", "foliage", "soil", "no_fire"]);

export function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

export function normalizeCanvasPoint(clientX, clientY, rect) {
  if (rect.width <= 0 || rect.height <= 0) {
    throw new Error("canvas has no drawable area");
  }
  return {
    x: clamp01((clientX - rect.left) / rect.width),
    y: clamp01((clientY - rect.top) / rect.height),
  };
}

export function polygonArea(points) {
  if (points.length < 3) return 0;
  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    twiceArea += current.x * next.y - next.x * current.y;
  }
  return Math.abs(twiceArea) / 2;
}

export function validateId(id, existingIds) {
  if (!/^[a-z][a-z0-9_-]{1,63}$/.test(id)) {
    throw new Error("IDs must be 2-64 lowercase letters, digits, underscores, or dashes");
  }
  if (existingIds.has(id)) throw new Error(`ID already exists: ${id}`);
}

export function validatePolygon(points) {
  if (points.length < 3) throw new Error("a polygon needs at least three points");
  if (polygonArea(points) < 0.0001) throw new Error("polygon area is too small");
}

export function buildScene({ calibrationId, zones, aimPresets }) {
  return {
    calibration_id: calibrationId,
    coordinate_space: "NORMALIZED_IMAGE",
    zones: zones.map((zone) => ({
      id: zone.id,
      type: zone.type,
      points: zone.points.map(({ x, y }) => ({ x, y })),
    })),
    aim_presets: aimPresets.map((preset) => ({
      id: preset.id,
      zone_id: preset.zone_id,
      point: { x: preset.point.x, y: preset.point.y },
      hardware_target: preset.hardware_target,
    })),
  };
}
