import { submitResultAPI } from "./api.js?v=10";
import {
  getDresState,
  getSubmissionMode,
  getSubmissionRoutes,
  handleSessionExpired,
  refreshCurrentTask,
  requireDresReady,
} from "./dres-session.js?v=12";
import {
  bindCopyButton,
  clearDresResult,
  renderDresError,
  renderDresReceipt,
} from "./dres-result.js?v=12";

let trakeVideoId = null;
let trakeFrames = [];
let submitting = false;
let submitted = false;

function resetReceipt() {
  submitted = false;
  clearDresResult(document.getElementById("trake-result"));
}

function normalizeCandidate(candidate) {
  const videoId = String(candidate.videoId || candidate.video_id || "").trim();
  const frameId = Number(candidate.frameId ?? candidate.keyframe_index);
  const fps = Number(candidate.fps) || 25;
  if (!videoId || !Number.isInteger(frameId) || frameId < 0) {
    throw new Error("Mốc TRAKE không hợp lệ.");
  }
  return { videoId, frameId, fps, timeMs: Math.round((frameId / fps) * 1000) };
}

function isStrictlyIncreasing() {
  return trakeFrames.every((item, index) => index === 0 || trakeFrames[index - 1].frameId < item.frameId);
}

function statusMessage() {
  if (!trakeFrames.length) return "Chưa có semantic keyframe nào.";
  if (!isStrictlyIncreasing()) return "Frame TRAKE phải tăng nghiêm ngặt theo thời gian.";
  return `${trakeFrames.length} mốc • sẵn sàng kiểm tra và nộp`;
}

function previewText() {
  if (!trakeVideoId || !trakeFrames.length) return "TR-<VIDEO_ID>-<FRAME_IDS>";
  return `TR-${trakeVideoId}-${trakeFrames.map((item) => item.frameId).join(",")}`;
}

function render() {
  const list = document.getElementById("trake-frame-list");
  const videoLabel = document.getElementById("trake-video-id");
  const status = document.getElementById("trake-status");
  const preview = document.getElementById("trake-preview");
  const submitButton = document.getElementById("trake-submit-btn");
  const count = document.getElementById("trake-count");
  const routePreview = document.getElementById("trake-route-preview");
  if (!list) return;

  list.innerHTML = "";
  trakeFrames.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "trake-frame-row";
    const image = document.createElement("img");
    image.src = `/keyframes/${encodeURIComponent(item.videoId)}/keyframe_${item.frameId}.webp`;
    image.alt = `Event ${index + 1}, frame ${item.frameId}`;
    const info = document.createElement("div");
    info.className = "trake-frame-info";
    info.innerHTML = `<strong>Event ${index + 1}</strong><span>Frame ${item.frameId} • ${item.timeMs} ms</span>`;
    const actions = document.createElement("div");
    actions.className = "trake-frame-actions";
    actions.innerHTML = `
      <button type="button" data-action="open" title="Mở video">▶</button>
      <button type="button" data-action="replace" title="Thay frame">↻</button>
      <button type="button" data-action="up" title="Đưa lên" ${index === 0 ? "disabled" : ""}>↑</button>
      <button type="button" data-action="down" title="Đưa xuống" ${index === trakeFrames.length - 1 ? "disabled" : ""}>↓</button>
      <button type="button" data-action="remove" title="Xóa">×</button>`;
    actions.addEventListener("click", (event) => {
      const action = event.target.closest("button")?.dataset.action;
      if (!action) return;
      if (action === "open") {
        document.dispatchEvent(new CustomEvent("trake:open-frame", { detail: item }));
      } else if (action === "remove") {
        trakeFrames.splice(index, 1);
        resetReceipt();
      } else if (action === "replace") {
        const rawFrame = prompt("Frame ID thay thế:", String(item.frameId));
        if (rawFrame === null) return;
        const frameId = Number(rawFrame);
        if (!Number.isInteger(frameId) || frameId < 0) {
          alert("Frame ID phải là số nguyên không âm.");
          return;
        }
        if (trakeFrames.some((frame, frameIndex) => frameIndex !== index && frame.frameId === frameId)) {
          alert(`Frame ${frameId} đã có trong TRAKE.`);
          return;
        }
        trakeFrames[index] = {
          ...item,
          frameId,
          timeMs: Math.round((frameId / item.fps) * 1000),
        };
        resetReceipt();
      } else if (action === "up" && index > 0) {
        [trakeFrames[index - 1], trakeFrames[index]] = [trakeFrames[index], trakeFrames[index - 1]];
        resetReceipt();
      } else if (action === "down" && index < trakeFrames.length - 1) {
        [trakeFrames[index + 1], trakeFrames[index]] = [trakeFrames[index], trakeFrames[index + 1]];
        resetReceipt();
      }
      if (!trakeFrames.length) trakeVideoId = null;
      render();
    });
    row.append(image, info, actions);
    list.appendChild(row);
  });

  if (videoLabel) videoLabel.textContent = trakeVideoId || "Chưa chọn video";
  if (status) {
    status.textContent = statusMessage();
    status.classList.toggle("error", trakeFrames.length > 0 && !isStrictlyIncreasing());
  }
  if (preview) preview.textContent = previewText();
  if (routePreview) {
    const routes = getSubmissionRoutes();
    routePreview.textContent = `POST ${routes.application}\n→ POST ${routes.dres}`;
  }
  if (count) count.textContent = String(trakeFrames.length);
  if (submitButton) {
    submitButton.disabled = submitting || !trakeFrames.length || !isStrictlyIncreasing();
    submitButton.textContent = submitting
      ? "Đang nộp..."
      : (submitted ? "Nộp lại TRAKE" : "Nộp TRAKE");
  }
}

export function openTrakeWorkspace() {
  document.getElementById("trake-workspace")?.classList.remove("hidden");
  render();
}

export function addTrakeCandidate(candidate) {
  let item;
  try {
    item = normalizeCandidate(candidate);
  } catch (error) {
    alert(error.message);
    return;
  }
  if (trakeVideoId && trakeVideoId !== item.videoId) {
    const reset = confirm(`TRAKE hiện tại thuộc ${trakeVideoId}. Xóa chuỗi cũ và chuyển sang ${item.videoId}?`);
    if (!reset) return;
    trakeFrames = [];
    resetReceipt();
  }
  if (trakeFrames.some((frame) => frame.frameId === item.frameId)) {
    alert(`Frame ${item.frameId} đã có trong TRAKE.`);
    return;
  }
  trakeVideoId = item.videoId;
  trakeFrames.push(item);
  resetReceipt();
  openTrakeWorkspace();
}

export function loadTrakeSequence(frames) {
  if (!Array.isArray(frames) || !frames.length) {
    alert("Chuỗi kết quả không có frame để nạp vào TRAKE.");
    return;
  }
  const normalized = frames.map(normalizeCandidate);
  const videoId = normalized[0].videoId;
  if (normalized.some((item) => item.videoId !== videoId)) {
    alert("TRAKE chỉ chấp nhận các frame trong cùng một video.");
    return;
  }
  trakeVideoId = videoId;
  trakeFrames = normalized.filter(
    (item, index, items) => items.findIndex((other) => other.frameId === item.frameId) === index,
  );
  resetReceipt();
  openTrakeWorkspace();
}

async function submitTrake() {
  if (submitting || !requireDresReady()) return;
  if (!trakeFrames.length || !isStrictlyIncreasing()) {
    alert("Frame TRAKE phải tăng nghiêm ngặt theo thời gian.");
    return;
  }
  const routes = getSubmissionRoutes();
  if (!confirm(`Nộp TRAKE?\n${previewText()}\n\nĐích DRES:\nPOST ${routes.dres}\n\nNộp sai bị trừ điểm.`)) return;

  submitting = true;
  render();
  const status = document.getElementById("trake-status");
  if (status) status.textContent = "Đang nộp TRAKE...";
  try {
    await refreshCurrentTask(true);
    if (!requireDresReady()) return;
    if (getSubmissionMode() !== "trake") {
      throw new Error("Task DRES hiện tại không còn là TRAKE. Vui lòng kiểm tra lại trước khi nộp.");
    }
    const dres = getDresState();
    const result = await submitResultAPI({
      evaluationId: dres.selectedEvaluationId,
      mode: "trake",
      videoId: trakeVideoId,
      frameIds: trakeFrames.map((item) => item.frameId),
    });
    submitted = true;
    renderDresReceipt(document.getElementById("trake-result"), result);
  } catch (error) {
    if (error.status === 401) handleSessionExpired();
    submitted = false;
    renderDresError(document.getElementById("trake-result"), error);
  } finally {
    submitting = false;
    render();
  }
}

export function initTrakeWorkspace() {
  document.getElementById("open-trake-btn")?.addEventListener("click", openTrakeWorkspace);
  document.getElementById("close-trake-btn")?.addEventListener("click", () => {
    document.getElementById("trake-workspace")?.classList.add("hidden");
  });
  document.getElementById("clear-trake-btn")?.addEventListener("click", () => {
    if (trakeFrames.length && !confirm("Xóa toàn bộ chuỗi TRAKE hiện tại?")) return;
    trakeFrames = [];
    trakeVideoId = null;
    resetReceipt();
    render();
  });
  bindCopyButton(
    document.getElementById("copy-trake-payload-btn"),
    previewText,
  );
  document.getElementById("trake-submit-btn")?.addEventListener("click", submitTrake);
  document.addEventListener("dres:mode-change", (event) => {
    if (event.detail === "trake") openTrakeWorkspace();
  });
  render();
}
