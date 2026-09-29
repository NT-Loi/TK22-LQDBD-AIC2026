import { submitResultAPI } from "./api.js?v=10";
import {
  getDresState,
  getSubmissionMode,
  getSubmissionRoutes,
  handleSessionExpired,
  refreshCurrentTask,
  requireDresReady,
} from "./dres-session.js?v=21";
import { addTrakeCandidate, loadTrakeSequence } from "./trake.js?v=21";
import {
  bindCopyButton,
  clearDresResult,
  renderDresError,
  renderDresReceipt,
} from "./dres-result.js?v=21";

let candidate = null;
let submitting = false;
let submitted = false;

let pendingItemToSubmit = null;

export function resumePendingSubmission() {
  if (pendingItemToSubmit && requireDresReady()) {
    const item = pendingItemToSubmit;
    pendingItemToSubmit = null;
    openSubmissionModal(item);
  }
}

function normalizeCandidate(item) {
  let videoId = String(item.videoId || item.video_id || "").trim();
  if (/^N\d+_/i.test(videoId)) {
    videoId = videoId.replace(/^(N\d+)_([A-Za-z0-9]+)$/i, "$1-$2");
  } else if (/^L\d+-/i.test(videoId)) {
    videoId = videoId.replace(/^(L\d+)-([A-Za-z0-9]+)$/i, "$1_$2");
  }
  const fps = Number(item.fps) || 25;
  const frameId = Number(item.frameId ?? item.keyframe_index ?? item.keyframe_idx ?? item.frame_idx);
  const hasExplicitTime = item.timeMs !== undefined && item.timeMs !== null;
  const explicitTime = Number(item.timeMs);
  const timeMs = hasExplicitTime && Number.isFinite(explicitTime)
    ? Math.round(explicitTime)
    : Math.round((frameId / fps) * 1000);
  if (!videoId || !Number.isInteger(frameId) || frameId < 0 || timeMs < 0) {
    throw new Error("Không xác định được video/frame cần nộp.");
  }
  return {
    videoId,
    frameId,
    fps,
    timeMs,
    source: item.source || "result",
    previewUrl: typeof item.previewUrl === "string" ? item.previewUrl : null,
  };
}

function dresPreview(mode, answer = "") {
  if (!candidate) return {};
  if (mode === "qa") {
    return {
      answerSets: [{ answers: [{ text: `QA-${answer.trim()}-${candidate.videoId}-${candidate.timeMs}` }] }],
    };
  }
  return {
    answerSets: [{
      answers: [{
        mediaItemName: candidate.videoId.replace(/\.(mp4|mov|avi|mkv|webm)$/i, ""),
        start: candidate.timeMs,
        end: candidate.timeMs,
      }],
    }],
  };
}

function renderModal() {
  const mode = getSubmissionMode();
  const answerRow = document.getElementById("qa-answer-row");
  const answerInput = document.getElementById("qa-answer-input");
  const details = document.getElementById("submission-details");
  const preview = document.getElementById("submission-payload-preview");
  const confirmButton = document.getElementById("confirm-submission-btn");
  const modeLabel = document.getElementById("submission-mode-label");
  const context = document.getElementById("submission-context");
  const routePreview = document.getElementById("submission-route-preview");
  const answer = answerInput?.value || "";
  const dres = getDresState();
  const evaluation = dres.evaluations.find(
    (item) => String(item.id) === String(dres.selectedEvaluationId),
  );

  answerRow?.classList.toggle("hidden", mode !== "qa");
  if (modeLabel) modeLabel.textContent = mode.toUpperCase();
  if (context) {
    context.textContent = `${dres.user?.username || "DRES"} • ${evaluation?.name || dres.selectedEvaluationId || "Chưa chọn evaluation"}${dres.currentTask?.name ? ` • ${dres.currentTask.name}` : ""}`;
  }
  if (details && candidate) {
    details.textContent = `${candidate.videoId} • frame ${candidate.frameId} • ${candidate.timeMs} ms • ${candidate.source === "player" ? "vị trí video đang dừng" : "keyframe kết quả"}`;
  }
  if (routePreview) {
    const routes = getSubmissionRoutes();
    routePreview.textContent = `POST ${routes.application}\n→ POST ${routes.dres}`;
  }
  if (preview) preview.textContent = JSON.stringify(dresPreview(mode, answer), null, 2);
  if (confirmButton) {
    confirmButton.disabled = submitting || (mode === "qa" && !answer.trim());
    confirmButton.textContent = submitted
      ? `Nộp lại ${mode.toUpperCase()}`
      : (submitting ? "Đang nộp..." : `Xác nhận nộp ${mode.toUpperCase()}`);
  }
}

function openSubmissionModal(item) {
  candidate = normalizeCandidate(item);
  pendingItemToSubmit = item;

  if (getSubmissionMode() === "trake") {
    addTrakeCandidate(candidate);
    return;
  }
  const result = document.getElementById("submission-result");
  submitted = false;
  if (result) {
    clearDresResult(result);
  }
  const answerInput = document.getElementById("qa-answer-input");
  if (answerInput) answerInput.value = "";
  document.getElementById("submission-modal")?.classList.remove("hidden");
  renderModal();

  if (!requireDresReady()) {
    return;
  }
  refreshCurrentTask(false);
}

export function handleSubmissionCandidate(event, item) {
  event?.stopPropagation();
  const vid = String(item?.videoId || item?.video_id || "").trim();
  if (vid.startsWith("N") && getSubmissionMode() === "trake") {
    alert("⚠️ Task TRAKE không sử dụng video giao thông (prefix N) theo quy định của BTC!");
    return;
  }
  try {
    openSubmissionModal(item);
  } catch (error) {
    alert(error.message);
  }
}

export function handleSequenceSubmission(event, frames) {
  event?.stopPropagation();
  const firstVid = String(frames?.[0]?.videoId || frames?.[0]?.video_id || "").trim();
  if (firstVid.startsWith("N") && getSubmissionMode() === "trake") {
    alert("⚠️ Task TRAKE không sử dụng video giao thông (prefix N) theo quy định của BTC!");
    return;
  }
  if (getSubmissionMode() === "trake") {
    try {
      loadTrakeSequence(frames);
    } catch (error) {
      alert(error.message);
    }
    return;
  }
  if (frames?.length) handleSubmissionCandidate(event, frames[0]);
}

export function handlePlayerSubmission({ videoId, currentTime, fps, previewUrl = null }) {
  const frameId = Math.round(currentTime * fps);
  openSubmissionModal({
    videoId,
    frameId,
    fps,
    timeMs: Math.round(currentTime * 1000),
    source: "player",
    previewUrl,
  });
}

async function confirmSubmission() {
  if (submitting || !candidate || !requireDresReady()) return;
  const mode = getSubmissionMode();
  if (mode === "trake") {
    addTrakeCandidate(candidate);
    document.getElementById("submission-modal")?.classList.add("hidden");
    return;
  }
  const answer = document.getElementById("qa-answer-input")?.value.trim() || "";
  if (mode === "qa" && !answer) return;

  submitting = true;
  renderModal();
  const resultBox = document.getElementById("submission-result");
  try {
    await refreshCurrentTask(true);
    if (!requireDresReady()) return;
    const dres = getDresState();
    const result = await submitResultAPI({
      evaluationId: dres.selectedEvaluationId,
      mode,
      videoId: candidate.videoId,
      timeMs: candidate.timeMs,
      answer: mode === "qa" ? answer : null,
    });
    renderDresReceipt(resultBox, result);
    submitted = true;
  } catch (error) {
    if (error.status === 401) handleSessionExpired();
    submitted = false;
    renderDresError(resultBox, error);
  } finally {
    submitting = false;
    renderModal();
  }
}

function updateSubmitButtonLabels() {
  document.querySelectorAll(".card-submit-btn").forEach((button) => {
    const isSequence = button.dataset.sequence === "true";
    button.textContent = getSubmissionButtonLabel({ sequence: isSequence });
  });
  const playerButton = document.getElementById("dynamic-submit-btn");
  if (playerButton) playerButton.textContent = getSubmissionButtonLabel({ player: true });
}

export function getSubmissionButtonLabel({ sequence = false, player = false } = {}) {
  if (getSubmissionMode() !== "trake") return "Submit";
  if (sequence) return "Nạp chuỗi TRAKE";
  if (player) return "Thêm mốc TRAKE";
  return "Thêm vào TRAKE";
}

export function initSubmissionUI() {
  document.getElementById("close-submission-btn")?.addEventListener("click", () => {
    document.getElementById("submission-modal")?.classList.add("hidden");
  });
  document.getElementById("cancel-submission-btn")?.addEventListener("click", () => {
    document.getElementById("submission-modal")?.classList.add("hidden");
  });
  document.getElementById("confirm-submission-btn")?.addEventListener("click", confirmSubmission);
  document.getElementById("qa-answer-input")?.addEventListener("input", () => {
    submitted = false;
    clearDresResult(document.getElementById("submission-result"));
    renderModal();
  });
  bindCopyButton(
    document.getElementById("copy-submission-payload-btn"),
    () => document.getElementById("submission-payload-preview")?.textContent || "",
  );
  document.addEventListener("dres:mode-change", () => {
    submitted = false;
    clearDresResult(document.getElementById("submission-result"));
    updateSubmitButtonLabels();
    renderModal();
  });
  document.addEventListener("dres:evaluation-selected", () => {
    resumePendingSubmission();
    renderModal();
  });
  document.addEventListener("dres:state-change", () => {
    renderModal();
  });
  updateSubmitButtonLabels();
}
