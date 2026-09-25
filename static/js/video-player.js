import { elements } from "./elements.js";
import {
  getSubmissionButtonLabel,
  handlePlayerSubmission,
} from "./submission.js?v=12";
import { getSubmissionMode } from "./dres-session.js?v=12";

let currentOpenVideoId = null;

// Chỉ chờ khi video vừa được tua tới frame mới; giới hạn thời gian chờ.
function waitForSeek(video) {
  if (!video.seeking) return Promise.resolve();

  return new Promise((resolve) => {
    const finish = () => {
      clearTimeout(timer);
      video.removeEventListener("seeked", finish);
      resolve();
    };
    const timer = setTimeout(finish, 300);
    video.addEventListener("seeked", finish, { once: true });
  });
}

// Chụp chính khung hình đã giải mã trong player, không tải/giải mã video lần hai.
function captureSmallPreview(video) {
  if (video.seeking || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
    return null;
  }

  const scale = Math.min(1, 320 / video.videoWidth);
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));

  try {
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.7);
  } catch {
    // Không để lỗi tạo ảnh cản việc thêm mốc TRAKE.
    return null;
  }
}


let modalSidebarSortMode = "time";

function renderModalSidebar(items, isTemporal, specificKeyframe, fps) {
  if (!elements.modalShotList) return;
  elements.modalShotList.innerHTML = "";

  if (!items || items.length === 0) {
    elements.modalShotList.innerHTML = "<div style='padding:10px; color:#8b949e;'>No frames list available.</div>";
    return;
  }

  // Header container
  const headerDiv = document.createElement("div");
  headerDiv.style.cssText = "padding: 8px 10px; background: #21262d; border-bottom: 1px solid #30363d; display: flex; align-items: center; justify-content: space-between;";

  const titleSpan = document.createElement("span");
  titleSpan.style.cssText = "color: #c9d1d9; font-size: 13px; font-weight: 600;";
  titleSpan.textContent = isTemporal ? "Sequence Events" : `Matched Keyframes (${items.length})`;
  headerDiv.appendChild(titleSpan);

  // Sort button
  const sortBtn = document.createElement("button");
  sortBtn.id = "modal-sort-btn";
  sortBtn.style.cssText = "background: #30363d; color: #58a6ff; border: 1px solid #58a6ff; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600; cursor: pointer; transition: background 0.2s;";
  sortBtn.textContent = modalSidebarSortMode === "score" ? "Sort: Score ⬇" : "Sort: Time ⬆";

  sortBtn.addEventListener("click", () => {
    modalSidebarSortMode = modalSidebarSortMode === "score" ? "time" : "score";
    renderModalSidebar(items, isTemporal, specificKeyframe, fps);
  });

  headerDiv.appendChild(sortBtn);
  elements.modalShotList.appendChild(headerDiv);

  // Sort items according to sort mode
  const sortedItems = [...items];
  if (modalSidebarSortMode === "score") {
    sortedItems.sort((a, b) => (b.score || 0) - (a.score || 0));
  } else {
    sortedItems.sort((a, b) => a.keyframe_index - b.keyframe_index);
  }

  sortedItems.forEach((item) => {
    const itemDiv = document.createElement("div");
    itemDiv.className = "sidebar-keyframe-item";
    if (item.keyframe_index === specificKeyframe) {
      itemDiv.classList.add("active");
      itemDiv.style.border = "2px solid #58a6ff";
    }

    const titleText = isTemporal ? `Event ${item.query_index + 1}` : `Frame: ${item.keyframe_index}`;
    const scoreVal = typeof item.score === "number" ? item.score.toFixed(3) : "N/A";

    itemDiv.innerHTML = `
        <img src="/keyframes/${item.video_id}/keyframe_${item.keyframe_index}.webp" loading="lazy">
        <div class="sidebar-info">
            <strong>${titleText}</strong>
            <span>Score: ${scoreVal}</span>
            ${item.ocr_text ? `<div style="font-size:10px; color:#58a6ff; margin-top:2px; max-height:2.4em; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; word-break:break-word;" title="${item.ocr_text}">🔍 ${item.ocr_text}</div>` : ""}
            ${item.audio_text ? `<div style="font-size:10px; color:#e3b341; margin-top:2px; max-height:2.4em; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; word-break:break-word;" title="${item.audio_text}">🎙️ ${item.audio_text}</div>` : ""}
        </div>
    `;
    itemDiv.addEventListener("click", () => {
      elements.modalVideoPlayer.currentTime = item.keyframe_index / fps;
      elements.modalVideoPlayer.play();
    });
    elements.modalShotList.appendChild(itemDiv);
  });
}

export function initVideoModal() {
  elements.closeModalBtn.addEventListener("click", closeModal);
  elements.modalOverlay.addEventListener("click", (e) => {
    if (e.target === elements.modalOverlay) {
      closeModal();
    }
  });
  document.addEventListener("trake:open-frame", (event) => {
    const item = event.detail;
    if (item?.videoId && Number.isInteger(item.frameId)) {
      openDirectVideo(item.videoId, item.frameId);
    }
  });
}

function highlightActiveCard(videoId, shotData, specificKeyframe) {
  document.querySelectorAll(".active-viewed-card").forEach((el) => {
    el.classList.remove("active-viewed-card");
  });
  let selector = "";
  if (shotData) {
    // Group Shot Mode
    const shotKey = `${videoId}|${shotData.shotStart}|${shotData.shotEnd}`;
    selector = `.shot-group-card[data-shot-key="${shotKey}"]`;
  } else {
    // Flat Mode
    if (specificKeyframe !== undefined && specificKeyframe !== null) {
      selector = `.result-item[data-video-id="${videoId}"][data-keyframe-index="${specificKeyframe}"]`;
    } else {
      selector = `.result-item[data-video-id="${videoId}"]`;
    }
  }
  const activeCard = document.querySelector(selector);
  if (activeCard) {
    activeCard.classList.add("active-viewed-card");
    activeCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

export function openModal(
  videoId,
  startTime,
  fps,
  shotData = null,
  specificKeyframe = null,
  sequenceData = null,
) {
  closeModal();
  currentOpenVideoId = videoId;
  highlightActiveCard(videoId, shotData, specificKeyframe);

  elements.modalVideoTitle.textContent = `Playing: ${videoId} (FPS: ${fps})`;
  elements.modalOverlay.classList.remove("hidden");

  // --- 1. SETUP PLAYER ---
  const videoUrl = `/video/${videoId}`;
  let mainHls = null;
  elements.modalVideoPlayer.src = videoUrl;
  elements.modalVideoPlayer.addEventListener(
    "loadedmetadata",
    function () {
      elements.modalVideoPlayer.currentTime = startTime;
      elements.modalVideoPlayer.play().catch((e) => console.warn(e));
    },
    { once: true },
  );

  // --- 2. SETUP TIMELINE UI (Full features) ---
  const videoWrapper = elements.modalVideoPlayer.parentElement;
  if (getComputedStyle(videoWrapper).position === "static") {
    videoWrapper.style.position = "relative";
  }

  // Clear old timeline
  const oldTimeline = videoWrapper.querySelector(".video-timeline");
  if (oldTimeline) oldTimeline.remove();
  // Clear old preview video if any
  const oldPreview = videoWrapper.querySelector("video[data-type='preview']");
  if (oldPreview) oldPreview.remove();

  const timelineBar = document.createElement("div");
  timelineBar.className = "video-timeline";
  videoWrapper.appendChild(timelineBar);

  const progressFill = document.createElement("div");
  Object.assign(progressFill.style, {
    position: "absolute",
    left: "0",
    top: "0",
    bottom: "0",
    width: "0%",
    background: "#1db954",
    pointerEvents: "none",
    zIndex: "10",
  });
  timelineBar.appendChild(progressFill);

  // --- 3. RED SHOT BAR (KHÔI PHỤC) ---
  if (shotData) {
    const addRedBar = () => {
      const duration = elements.modalVideoPlayer.duration;
      if (!duration || duration === Infinity) return;

      const startSec = shotData.shotStart / fps;
      const endSec = shotData.shotEnd / fps;
      const leftPct = (startSec / duration) * 100;
      const widthPct = ((endSec - startSec) / duration) * 100;

      const redBar = document.createElement("div");
      redBar.className = "shot-highlight-bar";
      Object.assign(redBar.style, {
        position: "absolute",
        top: "0",
        bottom: "0",
        left: `${leftPct}%`,
        width: `${widthPct}%`,
        backgroundColor: "rgba(220, 53, 69, 0.8)",
        pointerEvents: "none",
        zIndex: "5",
      });
      timelineBar.appendChild(redBar);
    };

    if (elements.modalVideoPlayer.readyState >= 1) {
      addRedBar();
    } else {
      elements.modalVideoPlayer.addEventListener("loadedmetadata", addRedBar, {
        once: true,
      });
    }
  }

  // --- 4. TIMELINE PREVIEW (KHÔI PHỤC) ---
  const timelinePreview = document.createElement("div");
  timelinePreview.className = "timeline-preview";
  timelinePreview.innerHTML = `<img src="" alt="Preview" style="display:none;"><div class="time-label">0:00</div>`;
  timelineBar.appendChild(timelinePreview);

  // Hidden video for preview generation
  const previewVideo = document.createElement("video");
  previewVideo.muted = true;
  previewVideo.setAttribute("data-type", "preview");
  previewVideo.style.display = "none";
  videoWrapper.appendChild(previewVideo);

  let previewHls = null;
  previewVideo.src = videoUrl;

  // Hover logic
  const previewImg = timelinePreview.querySelector("img");
  const timeLabel = timelinePreview.querySelector(".time-label");
  const previewCanvas = document.createElement("canvas");
  const previewCtx = previewCanvas.getContext("2d");
  let hoverTargetTime = null;
  let hoverScheduled = false;

  const runHoverPreview = () => {
    hoverScheduled = false;
    if (hoverTargetTime === null || !elements.modalVideoPlayer.duration) return;

    // Nếu HLS, cần seek
    previewVideo.currentTime = hoverTargetTime;

    // Ở đây ta dùng sự kiện seeked để capture frame
    const onSeeked = () => {
      if (Math.abs(previewVideo.currentTime - hoverTargetTime) > 1.0) return; // Quá xa thì bỏ

      const vw = previewVideo.videoWidth || 320;
      const vh = previewVideo.videoHeight || 180;
      if (vw && vh) {
        previewCanvas.width = vw;
        previewCanvas.height = vh;
        try {
          previewCtx.drawImage(previewVideo, 0, 0, vw, vh);
          previewImg.src = previewCanvas.toDataURL("image/jpeg", 0.5);
          previewImg.style.display = "block";
        } catch (e) { }
      }
    };
    previewVideo.addEventListener("seeked", onSeeked, { once: true });
  };

  timelineBar.addEventListener("mousemove", (e) => {
    if (!elements.modalVideoPlayer.duration) return;
    const rect = timelineBar.getBoundingClientRect();
    const percent = Math.max(
      0,
      Math.min(1, (e.clientX - rect.left) / rect.width),
    );
    const hoverTime = percent * elements.modalVideoPlayer.duration;

    timelinePreview.style.display = "block";
    // Position
    let left = percent * rect.width - 80; // 80 = half width of 160px
    left = Math.max(0, Math.min(left, rect.width - 160));
    timelinePreview.style.left = `${percent * 100}%`;
    timelinePreview.style.transform = `translateX(-50%)`;

    // Time Label
    const m = Math.floor(hoverTime / 60);
    const s = Math.floor(hoverTime % 60);
    timeLabel.textContent = `${m}:${s.toString().padStart(2, "0")}`;

    hoverTargetTime = hoverTime;
    if (!hoverScheduled) {
      hoverScheduled = true;
      setTimeout(runHoverPreview, 100); // Debounce
    }
  });

  timelineBar.addEventListener("mouseleave", () => {
    timelinePreview.style.display = "none";
    hoverTargetTime = null;
  });

  // --- 5. SIDEBAR PLAYLIST (WITH SORT BY SCORE/TIME TOGGLE) ---
  if (sequenceData && sequenceData.length > 0) {
    const isTemporal = sequenceData[0] && sequenceData[0].query_index !== undefined;
    renderModalSidebar(sequenceData, isTemporal, specificKeyframe, fps);
  } else if (shotData && shotData.items) {
    renderModalSidebar(shotData.items, false, specificKeyframe, fps);
  } else if (elements.modalShotList) {
    elements.modalShotList.innerHTML =
      "<div style='padding:10px; color:#8b949e;'>No frames list available.</div>";
  }

  // --- 6. CONTROLS (NEW) ---
  const oldControls =
    elements.modalPlayerSection.querySelector(".frame-controls");
  if (oldControls) oldControls.remove();
  const frameControls = document.createElement("div");
  frameControls.className = "frame-controls";
  frameControls.innerHTML = `
        <button id="dynamic-submit-btn" class="modal-submit-btn">Submit</button>
        <div class="frame-navigation">
            <button class="frame-btn" id="prev-frame-btn">-</button>
            <div class="frame-input-group">
                <input type="number" id="current-frame-input" class="frame-input" value="0">
                <span id="total-frames-span">/ 0</span>
            </div>
            <button class="frame-btn" id="next-frame-btn">+</button>
        </div>
        <div style="width: 80px;"></div>`;
  elements.modalPlayerSection.appendChild(frameControls);

  // --- 7. EVENT LISTENERS ---
  const frameRate = fps;
  const dynSubmitBtn = frameControls.querySelector("#dynamic-submit-btn");
  dynSubmitBtn.textContent = getSubmissionButtonLabel({ player: true });
  const frameInput = frameControls.querySelector("#current-frame-input");
  const totalFramesSpan = frameControls.querySelector("#total-frames-span");
  const prevBtn = frameControls.querySelector("#prev-frame-btn");
  const nextBtn = frameControls.querySelector("#next-frame-btn");

  // Progress Update
  const updateProgress = () => {
    if (!elements.modalVideoPlayer.duration) return;
    const p =
      (elements.modalVideoPlayer.currentTime /
        elements.modalVideoPlayer.duration) *
      100;
    progressFill.style.width = `${p}%`;
  };
  const updateFrameInfo = () => {
    if (!elements.modalVideoPlayer.duration) return;
    const cf = Math.round(elements.modalVideoPlayer.currentTime * frameRate);
    if (document.activeElement !== frameInput) frameInput.value = cf;
    totalFramesSpan.textContent = `/ ${Math.floor(elements.modalVideoPlayer.duration * frameRate)}`;
  };
  elements.modalVideoPlayer.addEventListener("timeupdate", updateProgress);
  elements.modalVideoPlayer.addEventListener("timeupdate", updateFrameInfo);
  elements.modalVideoPlayer.addEventListener("loadedmetadata", updateFrameInfo);

  // Timeline Click
  timelineBar.addEventListener("click", (e) => {
    const rect = timelineBar.getBoundingClientRect();
    const percent = Math.max(
      0,
      Math.min(1, (e.clientX - rect.left) / rect.width),
    );
    elements.modalVideoPlayer.currentTime =
      percent * elements.modalVideoPlayer.duration;
  });

  // Submit Logic (Pause Video)
  dynSubmitBtn.addEventListener("click", async () => {
    if (dynSubmitBtn.disabled) return;
    dynSubmitBtn.disabled = true;

    try {
      const video = elements.modalVideoPlayer;
      video.pause(); // Giữ hành vi hiện tại khi chọn frame.
      const selectedTime = video.currentTime;
      let previewUrl = null;

      if (getSubmissionMode() === "trake") {
        await waitForSeek(video);
        // Không gắn ảnh của một frame khác nếu người dùng tua tiếp khi đang chờ.
        if (Math.abs(video.currentTime - selectedTime) <= 1 / frameRate) {
          previewUrl = captureSmallPreview(video);
        }
      }

      handlePlayerSubmission({ videoId, currentTime: selectedTime, fps: frameRate, previewUrl });
    } finally {
      dynSubmitBtn.disabled = false;
    }
  });

  // Nav Logic
  const stepFrame = (dir) => {
    elements.modalVideoPlayer.pause();
    const cf = Math.round(elements.modalVideoPlayer.currentTime * frameRate);
    elements.modalVideoPlayer.currentTime = Math.max(
      0,
      (cf + dir) / frameRate + 0.0001,
    );
  };
  prevBtn.addEventListener("click", () => stepFrame(-1));
  nextBtn.addEventListener("click", () => stepFrame(1));
  frameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const val = parseInt(frameInput.value, 10);
      if (!isNaN(val) && val >= 0) {
        elements.modalVideoPlayer.currentTime = val / frameRate;
        elements.modalVideoPlayer.pause();
      }
      frameInput.blur();
    }
  });

  // Global Space Key
  const handleKey = (e) => {
    if (elements.modalOverlay.classList.contains("hidden")) return;
    if (e.target.tagName === "INPUT") return;
    if (e.code === "Space") {
      e.preventDefault();
      elements.modalVideoPlayer.paused
        ? elements.modalVideoPlayer.play()
        : elements.modalVideoPlayer.pause();
    }
    if (e.key === "ArrowLeft") stepFrame(-1);
    if (e.key === "ArrowRight") stepFrame(1);
    if (e.key === "Escape") closeModal();
  };
  document.addEventListener("keydown", handleKey);

  // Store cleanup
  elements.modalOverlay.dataset.handlersAttached = "true";
  elements.modalOverlay._cleanupHandlers = {
    handleKey,
    timelineBar,
    frameControls,
    mainHls,
    previewHls,
    previewVideo,
    updateProgress,
    updateFrameInfo,
  };
}

export function closeModal() {
  if (elements.modalOverlay.classList.contains("hidden")) return;
  const h = elements.modalOverlay._cleanupHandlers;
  if (h) {
    document.removeEventListener("keydown", h.handleKey);
    if (h.timelineBar) h.timelineBar.remove();
    if (h.frameControls) h.frameControls.remove();
    elements.modalVideoPlayer.removeEventListener(
      "timeupdate",
      h.updateProgress,
    );
    elements.modalVideoPlayer.removeEventListener(
      "timeupdate",
      h.updateFrameInfo,
    );
    elements.modalVideoPlayer.removeEventListener(
      "loadedmetadata",
      h.updateFrameInfo,
    );
    if (h.mainHls) h.mainHls.destroy();
    if (h.previewHls) h.previewHls.destroy();
    if (h.previewVideo) h.previewVideo.remove();
  }
  delete elements.modalOverlay._cleanupHandlers;
  elements.modalOverlay.classList.add("hidden");
  elements.modalVideoPlayer.pause();
  elements.modalVideoPlayer.removeAttribute("src");
  elements.modalVideoPlayer.load();
  if (elements.modalShotList) elements.modalShotList.innerHTML = "";
}

/**
 * Open modal for a caption search result.
 * Fetches all keyframes in the shot range and renders a tabbed sidebar:
 *  - "Keyframes" tab: all keyframes in the shot
 *  - "Caption" tab: caption aspects ranked by their score
 */
export async function openCaptionModal(captionItem, fps) {
  const videoId = captionItem.video_id;
  const repFrame = captionItem.representative_frame || captionItem.keyframe_index;
  const startTime = Math.max(0, repFrame / fps - 0.5);

  // Open the standard modal (video player, timeline, controls)
  openModal(videoId, startTime, fps, null, repFrame, null);

  // Now override the sidebar with tabbed content
  if (!elements.modalShotList) return;

  elements.modalShotList.innerHTML = '<div style="padding:10px; color:#8b949e;">Loading shot keyframes...</div>';

  // Fetch all keyframes in the shot range
  let shotKeyframes = [];
  try {
    const startF = captionItem.shot_start_frame || 0;
    const endF = captionItem.shot_end_frame || 999999999;
    const res = await fetch(`/api/shot_keyframes/${encodeURIComponent(videoId)}?start_frame=${startF}&end_frame=${endF}`);
    const data = await res.json();
    shotKeyframes = data.keyframes || [];
  } catch (e) {
    console.error("Failed to fetch shot keyframes:", e);
  }

  // Build tabbed sidebar
  renderCaptionSidebar(captionItem, shotKeyframes, fps, repFrame);
}

let captionSidebarActiveTab = "keyframes";

function renderCaptionSidebar(captionItem, shotKeyframes, fps, specificKeyframe) {
  if (!elements.modalShotList) return;
  elements.modalShotList.innerHTML = "";

  // --- Tab bar ---
  const tabBar = document.createElement("div");
  tabBar.style.cssText = "display: flex; border-bottom: 2px solid #30363d; background: #161b22;";

  const tabs = [
    { id: "keyframes", label: `🖼️ Keyframes (${shotKeyframes.length})` },
    { id: "caption", label: "📝 Caption" }
  ];

  tabs.forEach(tab => {
    const tabBtn = document.createElement("button");
    tabBtn.textContent = tab.label;
    tabBtn.dataset.tab = tab.id;
    const isActive = captionSidebarActiveTab === tab.id;
    tabBtn.style.cssText = `flex: 1; padding: 8px 4px; border: none; cursor: pointer; font-size: 12px; font-weight: 600; transition: all 0.2s; ${isActive
      ? "background: #21262d; color: #58a6ff; border-bottom: 2px solid #58a6ff;"
      : "background: #161b22; color: #8b949e; border-bottom: 2px solid transparent;"
      }`;
    tabBtn.addEventListener("click", () => {
      captionSidebarActiveTab = tab.id;
      renderCaptionSidebar(captionItem, shotKeyframes, fps, specificKeyframe);
    });
    tabBar.appendChild(tabBtn);
  });
  elements.modalShotList.appendChild(tabBar);

  // --- Tab content ---
  const contentDiv = document.createElement("div");
  contentDiv.style.cssText = "overflow-y: auto; flex: 1;";

  if (captionSidebarActiveTab === "keyframes") {
    renderKeyframesTab(contentDiv, shotKeyframes, fps, specificKeyframe);
  } else {
    renderCaptionTab(contentDiv, captionItem);
  }

  elements.modalShotList.appendChild(contentDiv);
}

function renderKeyframesTab(container, keyframes, fps, specificKeyframe) {
  if (!keyframes || keyframes.length === 0) {
    container.innerHTML = '<div style="padding:10px; color:#8b949e;">No keyframes found in this shot.</div>';
    return;
  }

  keyframes.forEach(kf => {
    const itemDiv = document.createElement("div");
    itemDiv.className = "sidebar-keyframe-item";
    if (kf.keyframe_index === specificKeyframe) {
      itemDiv.classList.add("active");
      itemDiv.style.border = "2px solid #58a6ff";
    }

    itemDiv.innerHTML = `
      <img src="/keyframes/${kf.video_id}/keyframe_${kf.keyframe_index}.webp" loading="lazy">
      <div class="sidebar-info">
        <strong>Frame: ${kf.keyframe_index}</strong>
      </div>
    `;
    itemDiv.addEventListener("click", () => {
      elements.modalVideoPlayer.currentTime = kf.keyframe_index / fps;
      elements.modalVideoPlayer.play();
    });
    container.appendChild(itemDiv);
  });
}

function renderCaptionTab(container, captionItem) {
  const aspects = captionItem.aspects || {};
  const bestAspect = captionItem.best_aspect || "";

  // Sort aspects: best aspect first, then alphabetically
  const sortedKeys = Object.keys(aspects).sort((a, b) => {
    if (a === bestAspect) return -1;
    if (b === bestAspect) return 1;
    return a.localeCompare(b);
  });

  if (sortedKeys.length === 0) {
    // Fallback to full caption text
    const fullCaption = captionItem.caption_text || "No caption available.";
    container.innerHTML = `<div style="padding: 10px; color: #c9d1d9; font-size: 12px; line-height: 1.5; white-space: pre-wrap;">${fullCaption}</div>`;
    return;
  }

  sortedKeys.forEach(key => {
    const aspectDiv = document.createElement("div");
    const isBest = key === bestAspect;
    aspectDiv.style.cssText = `padding: 8px 10px; border-bottom: 1px solid #21262d; ${isBest ? 'background: rgba(88, 166, 255, 0.08);' : ''}`;

    const label = key.replace(/_/g, ' ');
    const text = aspects[key] || "";

    aspectDiv.innerHTML = `
      <div style="font-size: 11px; font-weight: 700; color: ${isBest ? '#58a6ff' : '#8b949e'}; margin-bottom: 4px; text-transform: capitalize;">
        ${isBest ? '⭐ ' : ''}${label}
      </div>
      <div style="font-size: 12px; color: #c9d1d9; line-height: 1.4; white-space: pre-wrap; word-break: break-word;">
        ${text}
      </div>
    `;
    container.appendChild(aspectDiv);
  });
}


export async function openDirectVideo(videoIdRaw, keyframeIdxRaw) {
  if (!videoIdRaw) return;

  let videoId = videoIdRaw.trim().toUpperCase();
  let keyframeIdx = parseInt(keyframeIdxRaw, 10);

  // Smart parsing if user inputs e.g. "L24_V044 / 186" or "L24_V044/186" into the video input
  if (videoId.includes("/") || videoId.includes(" ")) {
    const parts = videoId.split(/[\/\s]+/).filter(Boolean);
    if (parts.length >= 1) videoId = parts[0];
    if (parts.length >= 2 && isNaN(keyframeIdx)) {
      keyframeIdx = parseInt(parts[1], 10);
    }
  } else if (videoId.includes("_") && isNaN(keyframeIdx)) {
    const subParts = videoId.split("_");
    if (subParts.length >= 3) {
      const lastPart = subParts[subParts.length - 1];
      if (!isNaN(parseInt(lastPart, 10))) {
        keyframeIdx = parseInt(lastPart, 10);
        videoId = subParts.slice(0, -1).join("_");
      }
    }
  }

  if (isNaN(keyframeIdx) || keyframeIdx < 0) {
    keyframeIdx = 0;
  }

  try {
    const newUrl = new URL(window.location.href);
    newUrl.searchParams.set("video", videoId);
    newUrl.searchParams.set("keyframe", keyframeIdx);
    window.history.pushState({}, "", newUrl);
  } catch (e) {
    console.warn("Failed to update URL params", e);
  }

  try {
    const res = await fetch(`/api/video_keyframes/${encodeURIComponent(videoId)}`);
    const data = await res.json();
    const fps = data.fps || 25.0;
    const keyframes = data.keyframes || [];

    const startTime = Math.max(0, keyframeIdx / fps - 0.5);

    openModal(videoId, startTime, fps, null, keyframeIdx);

    if (keyframes.length > 0) {
      renderModalSidebar(keyframes, false, keyframeIdx, fps);
    }
  } catch (err) {
    console.error("Error launching direct video jump:", err);
    const fps = 25.0;
    const startTime = Math.max(0, keyframeIdx / fps - 0.5);
    openModal(videoId, startTime, fps, null, keyframeIdx);
  }
}
