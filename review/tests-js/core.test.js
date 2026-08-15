"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const core = require("../src/foliage_warden_review/web/core.js");

test("suggestEventId produces a stable valid identifier", () => {
  const draft = {
    session_id: "Session 01",
    media_id: "Clip/A",
    behavior: "EATING",
    start_ms: 120,
    end_ms: 980,
  };
  assert.equal(core.suggestEventId(draft), "session-01-clip-a-eating-120-980");
  assert.equal(core.suggestEventId(draft), core.suggestEventId(draft));
});

test("client validation fails closed on privacy and interval errors", () => {
  const errors = core.validateDraft(
    {
      event_id: "event-1",
      session_id: "session-1",
      group_id: "day-1",
      media_id: "clip-1",
      behavior: "EATING",
      start_ms: 500,
      end_ms: 400,
      rationale: "visible repeated mouth motion",
      zone_id: null,
      person_present: true,
      privacy_restricted: false,
    },
    1000
  );
  assert.ok(errors.some((value) => value.includes("greater than")));
  assert.ok(errors.some((value) => value.includes("privacy restricted")));
});

test("annotation sorting is deterministic and does not mutate input", () => {
  const values = [
    { session_id: "s", media_id: "b", start_ms: 1, end_ms: 2, event_id: "z" },
    { session_id: "s", media_id: "a", start_ms: 8, end_ms: 9, event_id: "a" },
    { session_id: "s", media_id: "a", start_ms: 2, end_ms: 3, event_id: "b" },
  ];
  assert.deepEqual(core.sortAnnotations(values).map((value) => value.event_id), ["b", "a", "z"]);
  assert.equal(values[0].event_id, "z");
});
