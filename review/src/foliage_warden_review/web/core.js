(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.FoliageReviewCore = api;
})(typeof globalThis === "object" ? globalThis : this, function () {
  "use strict";

  const behaviors = new Set(["PASSING", "SNIFFING", "EATING", "DIGGING", "OTHER", "UNKNOWN"]);
  const identifierPattern = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

  function identifierPart(value) {
    const normalized = String(value || "")
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, "-")
      .replace(/^[^a-z0-9]+/, "")
      .replace(/-+$/g, "")
      .slice(0, 40);
    return normalized || "event";
  }

  function suggestEventId(draft) {
    return [
      identifierPart(draft.session_id),
      identifierPart(draft.media_id),
      identifierPart(draft.behavior),
      String(draft.start_ms),
      String(draft.end_ms),
    ].join("-").slice(0, 128);
  }

  function validateDraft(draft, durationMs) {
    const errors = [];
    for (const field of ["event_id", "session_id", "group_id", "media_id"]) {
      if (!identifierPattern.test(draft[field] || "")) errors.push(`${field} is not a valid identifier`);
    }
    if (!behaviors.has(draft.behavior)) errors.push("behavior is invalid");
    if (!Number.isInteger(draft.start_ms) || draft.start_ms < 0) errors.push("start_ms must be an integer ≥ 0");
    if (!Number.isInteger(draft.end_ms) || draft.end_ms <= draft.start_ms) errors.push("end_ms must be greater than start_ms");
    if (Number.isInteger(durationMs) && draft.end_ms > durationMs) errors.push("end_ms exceeds the manifest duration");
    if (!String(draft.rationale || "").trim()) errors.push("rationale is required");
    if (String(draft.rationale || "").length > 2000) errors.push("rationale is longer than 2000 characters");
    if (draft.zone_id !== null && !identifierPattern.test(draft.zone_id || "")) errors.push("zone_id is not a valid identifier");
    if (draft.person_present && !draft.privacy_restricted) errors.push("person-present media must be privacy restricted");
    return errors;
  }

  function sortAnnotations(values) {
    return [...values].sort((left, right) => {
      for (const field of ["session_id", "media_id"]) {
        const comparison = left[field].localeCompare(right[field]);
        if (comparison) return comparison;
      }
      return left.start_ms - right.start_ms || left.end_ms - right.end_ms || left.event_id.localeCompare(right.event_id);
    });
  }

  return { behaviors, identifierPart, sortAnnotations, suggestEventId, validateDraft };
});
