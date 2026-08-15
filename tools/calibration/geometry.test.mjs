import assert from "node:assert/strict";
import test from "node:test";

import {
  buildScene,
  normalizeCanvasPoint,
  polygonArea,
  validateId,
  validatePolygon,
} from "./geometry.mjs";

test("canvas coordinates are normalized and clamped", () => {
  const rect = { left: 10, top: 20, width: 200, height: 100 };
  assert.deepEqual(normalizeCanvasPoint(110, 70, rect), { x: 0.5, y: 0.5 });
  assert.deepEqual(normalizeCanvasPoint(-10, 500, rect), { x: 0, y: 1 });
});

test("polygon area is independent of winding", () => {
  const clockwise = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 },
  ];
  assert.equal(polygonArea(clockwise), 1);
  assert.equal(polygonArea(clockwise.toReversed()), 1);
});

test("degenerate polygons and duplicate IDs are rejected", () => {
  assert.throws(() => validatePolygon([{ x: 0, y: 0 }, { x: 1, y: 1 }]));
  assert.throws(() => validateId("Pot 1", new Set()));
  assert.throws(() => validateId("pot_1", new Set(["pot_1"])));
});

test("scene export uses normalized canonical fields", () => {
  const scene = buildScene({
    calibrationId: "living_room_v1",
    zones: [
      {
        id: "pot_1",
        type: "approach",
        points: [
          { x: 0.2, y: 0.3 },
          { x: 0.4, y: 0.3 },
          { x: 0.3, y: 0.5 },
        ],
      },
    ],
    aimPresets: [
      {
        id: "pot_1_front",
        zone_id: "pot_1",
        point: { x: 0.25, y: 0.4 },
        hardware_target: "disabled",
      },
    ],
  });
  assert.equal(scene.calibration_id, "living_room_v1");
  assert.equal(scene.coordinate_space, "NORMALIZED_IMAGE");
  assert.equal(scene.zones[0].points[0].x, 0.2);
  assert.equal(scene.aim_presets[0].hardware_target, "disabled");
});
