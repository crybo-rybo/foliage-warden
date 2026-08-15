(function () {
  "use strict";

  const core = globalThis.FoliageReviewCore;
  const elements = {
    annotationCount: document.querySelector("#annotation-count"),
    annotationRows: document.querySelector("#annotation-rows"),
    currentTime: document.querySelector("#current-time"),
    endMs: document.querySelector("#end-ms"),
    eventId: document.querySelector("#event-id"),
    form: document.querySelector("#annotation-form"),
    formError: document.querySelector("#form-error"),
    markEnd: document.querySelector("#mark-end"),
    markStart: document.querySelector("#mark-start"),
    mediaContext: document.querySelector("#media-context"),
    mediaSelect: document.querySelector("#media-select"),
    mediaShell: document.querySelector("#media-shell"),
    newAnnotation: document.querySelector("#new-annotation"),
    personPresent: document.querySelector("#person-present"),
    privacyRestricted: document.querySelector("#privacy-restricted"),
    rationale: document.querySelector("#rationale"),
    saveStatus: document.querySelector("#save-status"),
    stagedSafe: document.querySelector("#staged-safe"),
    startMs: document.querySelector("#start-ms"),
    zoneId: document.querySelector("#zone-id"),
  };

  let manifest = null;
  let store = { annotations: [], revision: 0 };
  let currentMediaElement = null;
  let editingEventId = null;

  function allMedia() {
    return manifest.sessions.flatMap((session) =>
      session.media.map((media) => ({ ...media, group_id: session.group_id, session_id: session.session_id }))
    );
  }

  function selectedMedia() {
    return allMedia().find((media) => `${media.session_id}\u0000${media.media_id}` === elements.mediaSelect.value);
  }

  async function getJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`Request failed (${response.status})`);
    return response.json();
  }

  async function postJson(url, body) {
    const response = await fetch(url, {
      body: JSON.stringify(body),
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      method: "POST",
    });
    const payload = await response.json();
    if (!response.ok) {
      if (response.status === 409) store = await getJson("/api/annotations");
      throw new Error(payload.error || `Request failed (${response.status})`);
    }
    return payload;
  }

  function behaviorValue() {
    return elements.form.querySelector('input[name="behavior"]:checked').value;
  }

  function currentMilliseconds() {
    return currentMediaElement instanceof HTMLVideoElement
      ? Math.round(currentMediaElement.currentTime * 1000)
      : 0;
  }

  function updateTime() {
    elements.currentTime.textContent = `${currentMilliseconds()} ms`;
  }

  function mountMedia() {
    const media = selectedMedia();
    elements.mediaShell.replaceChildren();
    const element = document.createElement(media.kind === "video" ? "video" : "img");
    element.src = media.media_url;
    element.alt = media.description || media.display_name;
    if (media.kind === "video") {
      element.controls = true;
      element.preload = "metadata";
      element.addEventListener("timeupdate", updateTime);
      element.addEventListener("seeked", updateTime);
    }
    elements.mediaShell.append(element);
    currentMediaElement = element;
    elements.mediaContext.textContent = `${media.session_id} · ${media.group_id} · ${media.duration_ms} ms`;
    clearForm();
  }

  function draftFromForm() {
    const media = selectedMedia();
    return {
      behavior: behaviorValue(),
      end_ms: Number(elements.endMs.value),
      event_id: elements.eventId.value.trim(),
      group_id: media.group_id,
      media_id: media.media_id,
      person_present: elements.personPresent.checked,
      privacy_restricted: elements.privacyRestricted.checked,
      rationale: elements.rationale.value.trim(),
      session_id: media.session_id,
      staged_safe: elements.stagedSafe.checked,
      start_ms: Number(elements.startMs.value),
      zone_id: elements.zoneId.value.trim() || null,
    };
  }

  function updateSuggestedId() {
    if (editingEventId) return;
    const media = selectedMedia();
    if (!media || elements.startMs.value === "" || elements.endMs.value === "") return;
    elements.eventId.value = core.suggestEventId({
      behavior: behaviorValue(),
      end_ms: Number(elements.endMs.value),
      media_id: media.media_id,
      session_id: media.session_id,
      start_ms: Number(elements.startMs.value),
    });
  }

  function clearForm() {
    editingEventId = null;
    elements.form.reset();
    const media = selectedMedia();
    elements.startMs.value = "0";
    elements.endMs.value = String(Math.min(1000, media.duration_ms));
    elements.zoneId.value = media.zone_id || "";
    elements.formError.hidden = true;
    elements.saveStatus.textContent = "";
    updateSuggestedId();
  }

  function editAnnotation(annotation) {
    const key = `${annotation.session_id}\u0000${annotation.media_id}`;
    if (elements.mediaSelect.value !== key) {
      elements.mediaSelect.value = key;
      mountMedia();
    }
    editingEventId = annotation.event_id;
    elements.form.querySelector(`input[name="behavior"][value="${annotation.behavior}"]`).checked = true;
    elements.startMs.value = annotation.start_ms;
    elements.endMs.value = annotation.end_ms;
    elements.zoneId.value = annotation.zone_id || "";
    elements.rationale.value = annotation.rationale;
    elements.stagedSafe.checked = annotation.staged_safe;
    elements.personPresent.checked = annotation.person_present;
    elements.privacyRestricted.checked = annotation.privacy_restricted;
    elements.eventId.value = annotation.event_id;
    elements.form.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function archiveAnnotation(eventId) {
    if (!globalThis.confirm(`Archive ${eventId}? Its prior value remains in local history.`)) return;
    try {
      store = await postJson("/api/archive", { event_id: eventId, expected_revision: store.revision });
      renderAnnotations();
      if (editingEventId === eventId) clearForm();
    } catch (error) {
      elements.saveStatus.textContent = error.message;
      renderAnnotations();
    }
  }

  function flagText(annotation) {
    const flags = [];
    if (annotation.staged_safe) flags.push("staged-safe");
    if (annotation.person_present) flags.push("person");
    if (annotation.privacy_restricted) flags.push("restricted");
    return flags.join(" · ") || "—";
  }

  function renderAnnotations() {
    const values = core.sortAnnotations(store.annotations);
    elements.annotationCount.textContent = values.length;
    elements.annotationRows.replaceChildren();
    for (const annotation of values) {
      const row = document.createElement("tr");
      const eventCell = document.createElement("td");
      const eventStrong = document.createElement("strong");
      eventStrong.textContent = annotation.event_id;
      const eventMeta = document.createElement("small");
      eventMeta.textContent = `${annotation.session_id} / ${annotation.media_id}`;
      eventCell.append(eventStrong, eventMeta);
      const labelCell = document.createElement("td");
      const chip = document.createElement("span");
      chip.className = `label label-${annotation.behavior.toLowerCase()}`;
      chip.textContent = annotation.behavior;
      labelCell.append(chip);
      const intervalCell = document.createElement("td");
      intervalCell.textContent = `${annotation.start_ms}–${annotation.end_ms} ms`;
      const flagsCell = document.createElement("td");
      flagsCell.textContent = flagText(annotation);
      const actionCell = document.createElement("td");
      const edit = document.createElement("button");
      edit.className = "table-button";
      edit.textContent = "Edit";
      edit.type = "button";
      edit.addEventListener("click", () => editAnnotation(annotation));
      const archive = document.createElement("button");
      archive.className = "table-button danger";
      archive.textContent = "Archive";
      archive.type = "button";
      archive.addEventListener("click", () => archiveAnnotation(annotation.event_id));
      actionCell.append(edit, archive);
      row.append(eventCell, labelCell, intervalCell, flagsCell, actionCell);
      elements.annotationRows.append(row);
    }
  }

  async function saveAnnotation(event) {
    event.preventDefault();
    elements.formError.hidden = true;
    elements.saveStatus.textContent = "";
    const draft = draftFromForm();
    const errors = core.validateDraft(draft, selectedMedia().duration_ms);
    if (editingEventId && draft.event_id !== editingEventId) {
      errors.push("event_id cannot change while editing; archive and create a new event instead");
    }
    if (errors.length) {
      elements.formError.textContent = errors.join(" · ");
      elements.formError.hidden = false;
      return;
    }
    try {
      store = await postJson("/api/annotations", {
        annotation: draft,
        expected_revision: store.revision,
      });
      renderAnnotations();
      clearForm();
      elements.saveStatus.textContent = "Saved atomically.";
    } catch (error) {
      elements.formError.textContent = error.message;
      elements.formError.hidden = false;
      renderAnnotations();
    }
  }

  async function initialize() {
    try {
      [manifest, store] = await Promise.all([getJson("/api/manifest"), getJson("/api/annotations")]);
      for (const media of allMedia()) {
        const option = document.createElement("option");
        option.value = `${media.session_id}\u0000${media.media_id}`;
        option.textContent = `${media.session_id} · ${media.display_name}`;
        elements.mediaSelect.append(option);
      }
      mountMedia();
      renderAnnotations();
    } catch (error) {
      elements.mediaShell.textContent = `Unable to initialize: ${error.message}`;
    }
  }

  elements.mediaSelect.addEventListener("change", mountMedia);
  elements.markStart.addEventListener("click", () => {
    elements.startMs.value = currentMilliseconds();
    updateSuggestedId();
  });
  elements.markEnd.addEventListener("click", () => {
    elements.endMs.value = currentMilliseconds();
    updateSuggestedId();
  });
  elements.form.addEventListener("submit", saveAnnotation);
  elements.newAnnotation.addEventListener("click", clearForm);
  elements.personPresent.addEventListener("change", () => {
    if (elements.personPresent.checked) elements.privacyRestricted.checked = true;
  });
  for (const input of elements.form.querySelectorAll('input[name="behavior"], #start-ms, #end-ms')) {
    input.addEventListener("change", updateSuggestedId);
  }

  initialize();
})();
