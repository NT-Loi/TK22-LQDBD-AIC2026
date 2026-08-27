import { openDirectVideo } from "./video-player.js";

let allCatalogVideos = [];
let selectedVideoId = null;
let currentKeyframes = [];

export async function initCatalogTab() {
  const tabBtnSearch = document.getElementById("tab-btn-search");
  const tabBtnCatalog = document.getElementById("tab-btn-catalog");
  const viewSearch = document.getElementById("view-search");
  const viewCatalog = document.getElementById("view-catalog");

  if (!tabBtnSearch || !tabBtnCatalog || !viewSearch || !viewCatalog) return;

  // Tab switching logic
  const switchTab = (tabName) => {
    if (tabName === "catalog") {
      tabBtnSearch.classList.remove("active");
      tabBtnCatalog.classList.add("active");
      viewSearch.classList.add("hidden");
      viewCatalog.classList.remove("hidden");

      if (allCatalogVideos.length === 0) {
        loadCatalogVideos();
      }
    } else {
      tabBtnCatalog.classList.remove("active");
      tabBtnSearch.classList.add("active");
      viewCatalog.classList.add("hidden");
      viewSearch.classList.remove("hidden");
    }
  };

  tabBtnSearch.addEventListener("click", () => switchTab("search"));
  tabBtnCatalog.addEventListener("click", () => switchTab("catalog"));

  // Check URL query param or hash
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get("tab") === "catalog" || window.location.hash === "#catalog") {
    switchTab("catalog");
  }

  // Search video catalog input
  const catalogSearchInput = document.getElementById("catalog-video-search");
  const catalogJumpBtn = document.getElementById("catalog-jump-btn");

  if (catalogSearchInput) {
    catalogSearchInput.addEventListener("input", () => {
      filterCatalogVideoList();
    });
  }

  const handleCatalogJump = () => {
    const rawVal = catalogSearchInput ? catalogSearchInput.value.trim() : "";

    if (!rawVal) {
      filterCatalogVideoList();
      return;
    }

    let vId = rawVal.toUpperCase();

    // 1. Filter video library list on left
    if (catalogSearchInput) catalogSearchInput.value = vId;
    filterCatalogVideoList();

    // 2. Find matching videos (exact match first, then partial match)
    const exactMatch = allCatalogVideos.find((v) => v.video_id.toUpperCase() === vId);
    const filteredMatches = allCatalogVideos.filter((v) => v.video_id.toUpperCase().includes(vId));

    if (exactMatch) {
      selectCatalogVideo(exactMatch);
    } else if (filteredMatches.length > 0) {
      // Select the first matching video in the filtered library
      selectCatalogVideo(filteredMatches[0]);
    } else {
      resetGalleryPane();
    }
  };

  if (catalogJumpBtn) {
    catalogJumpBtn.addEventListener("click", handleCatalogJump);
  }
  if (catalogSearchInput) {
    catalogSearchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleCatalogJump();
      }
    });
  }

  // Keyframe filter input in gallery header
  const galleryFrameFilter = document.getElementById("gallery-frame-filter");
  if (galleryFrameFilter) {
    galleryFrameFilter.addEventListener("input", filterGalleryKeyframes);
  }

  // Play video button
  const galleryPlayBtn = document.getElementById("gallery-play-btn");
  if (galleryPlayBtn) {
    galleryPlayBtn.addEventListener("click", () => {
      if (selectedVideoId) {
        openDirectVideo(selectedVideoId, 0);
      }
    });
  }
}

function resetGalleryPane() {
  selectedVideoId = null;
  currentKeyframes = [];
  const nameEl = document.getElementById("gallery-video-name");
  const metaEl = document.getElementById("gallery-video-info");
  const playBtn = document.getElementById("gallery-play-btn");
  const gridContainer = document.getElementById("catalog-keyframe-grid");

  if (nameEl) nameEl.textContent = "Select a video to browse keyframes";
  if (metaEl) metaEl.textContent = "";
  if (playBtn) playBtn.classList.add("hidden");
  if (gridContainer) {
    gridContainer.innerHTML = '<p class="placeholder" style="padding:40px; text-align:center; color:#8b949e; grid-column:1/-1;">Click any video from the library on the left to view all its keyframes.</p>';
  }
}

async function loadCatalogVideos() {
  const container = document.getElementById("catalog-video-list");
  const countBadge = document.getElementById("catalog-total-videos");
  if (container) {
    container.innerHTML = '<div style="padding:15px; color:#8b949e;">Loading video catalog...</div>';
  }

  try {
    const res = await fetch("/api/videos");
    const data = await res.json();
    allCatalogVideos = data.videos || [];

    if (countBadge) {
      countBadge.textContent = `Total Videos: ${allCatalogVideos.length}`;
    }

    renderCatalogVideoList(allCatalogVideos);

    // Auto-select video ONLY if present in URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    const paramVideo = urlParams.get("video");
    const paramKeyframe = urlParams.get("keyframe");

    if (paramVideo) {
      const match = allCatalogVideos.find((v) => v.video_id.toUpperCase() === paramVideo.toUpperCase());
      if (match) {
        const searchInput = document.getElementById("catalog-video-search");
        if (searchInput) searchInput.value = match.video_id;

        selectCatalogVideo(match, paramKeyframe !== null ? parseInt(paramKeyframe, 10) : null);
      } else {
        resetGalleryPane();
      }
    } else {
      // First time switching to tab: ONLY show video list on left, do NOT show keyframes on right until tapped
      resetGalleryPane();
    }
  } catch (err) {
    console.error("Failed to load catalog videos:", err);
    if (container) {
      container.innerHTML = '<div style="padding:15px; color:#f85149;">Failed to load video library.</div>';
    }
  }
}

function renderCatalogVideoList(videos) {
  const container = document.getElementById("catalog-video-list");
  if (!container) return;
  container.innerHTML = "";

  if (!videos || videos.length === 0) {
    container.innerHTML = '<div style="padding:15px; color:#8b949e;">No matching videos found.</div>';
    return;
  }

  videos.forEach((v) => {
    const card = document.createElement("div");
    card.className = `catalog-video-card ${v.video_id === selectedVideoId ? "active" : ""}`;
    card.dataset.videoId = v.video_id;

    const fpsFormatted = typeof v.fps === "number" ? (Number.isInteger(v.fps) ? v.fps : v.fps.toFixed(2)) : v.fps;
    const thumbUrl = v.first_keyframe ? `/keyframes/${v.video_id}/${v.first_keyframe}` : "/static/placeholder.png";

    card.innerHTML = `
      <img src="${thumbUrl}" class="catalog-video-thumb" loading="lazy" onerror="this.onerror=null;this.src='/static/placeholder.png';">
      <div class="catalog-video-info">
        <div class="catalog-video-id">${v.video_id}</div>
        <div class="catalog-video-meta-tags">
          <span>FPS: ${fpsFormatted}</span>
          <span>Frames: ${v.keyframe_count}</span>
        </div>
      </div>
    `;

    card.addEventListener("click", () => {
      document.querySelectorAll(".catalog-video-card.active").forEach((el) => el.classList.remove("active"));
      card.classList.add("active");
      selectCatalogVideo(v);
    });

    container.appendChild(card);
  });
}

function filterCatalogVideoList() {
  const query = document.getElementById("catalog-video-search")?.value.trim().toUpperCase() || "";
  if (!query) {
    renderCatalogVideoList(allCatalogVideos);
    return;
  }
  const filtered = allCatalogVideos.filter((v) => v.video_id.toUpperCase().includes(query));
  renderCatalogVideoList(filtered);
}

async function selectCatalogVideo(videoObj, targetFrameIdx = null) {
  selectedVideoId = videoObj.video_id;

  // Highlight active video card in left list
  document.querySelectorAll(".catalog-video-card").forEach((card) => {
    if (card.dataset.videoId === videoObj.video_id) {
      card.classList.add("active");
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } else {
      card.classList.remove("active");
    }
  });

  const nameEl = document.getElementById("gallery-video-name");
  const metaEl = document.getElementById("gallery-video-info");
  const playBtn = document.getElementById("gallery-play-btn");
  const gridContainer = document.getElementById("catalog-keyframe-grid");

  if (nameEl) nameEl.textContent = `Video: ${videoObj.video_id}`;
  if (metaEl) {
    const fpsFormatted = typeof videoObj.fps === "number" ? (Number.isInteger(videoObj.fps) ? videoObj.fps : videoObj.fps.toFixed(2)) : videoObj.fps;
    metaEl.textContent = `${videoObj.keyframe_count} keyframes | FPS: ${fpsFormatted}`;
  }
  if (playBtn) playBtn.classList.remove("hidden");

  if (gridContainer) {
    gridContainer.innerHTML = '<div style="padding:20px; color:#8b949e; grid-column:1/-1;">Loading keyframes...</div>';
  }

  try {
    const res = await fetch(`/api/video_keyframes/${encodeURIComponent(videoObj.video_id)}`);
    const data = await res.json();
    currentKeyframes = data.keyframes || [];
    renderGalleryKeyframes(currentKeyframes, videoObj.fps, targetFrameIdx);
  } catch (err) {
    console.error("Error loading keyframes for gallery:", err);
    if (gridContainer) {
      gridContainer.innerHTML = '<div style="padding:20px; color:#f85149; grid-column:1/-1;">Failed to load keyframes.</div>';
    }
  }
}

function findNearestKeyframe(keyframes, targetFrameIdx) {
  if (!keyframes || keyframes.length === 0) return null;
  if (targetFrameIdx === null || targetFrameIdx === undefined || isNaN(targetFrameIdx)) return null;

  let nearest = keyframes[0];
  let minDiff = Math.abs(keyframes[0].keyframe_index - targetFrameIdx);

  for (let i = 1; i < keyframes.length; i++) {
    const diff = Math.abs(keyframes[i].keyframe_index - targetFrameIdx);
    if (diff < minDiff) {
      minDiff = diff;
      nearest = keyframes[i];
    }
  }
  return nearest;
}

function renderGalleryKeyframes(keyframes, fps, targetFrameIdx = null) {
  const container = document.getElementById("catalog-keyframe-grid");
  if (!container) return;
  container.innerHTML = "";

  if (!keyframes || keyframes.length === 0) {
    container.innerHTML = '<div style="padding:20px; color:#8b949e; grid-column:1/-1;">No keyframes available for this video.</div>';
    return;
  }

  // Find nearest keyframe if a frame # is requested
  const nearestKf = targetFrameIdx !== null && targetFrameIdx !== undefined && !isNaN(targetFrameIdx)
    ? findNearestKeyframe(keyframes, targetFrameIdx)
    : null;

  let targetCardEl = null;

  keyframes.forEach((kf) => {
    const card = document.createElement("div");
    card.className = "catalog-keyframe-card";
    card.style.cssText = "min-height: 120px; display: flex; flex-direction: column; background: #161b22; border: 1px solid #30363d; border-radius: 6px; overflow: hidden; cursor: pointer; position: relative;";

    if (nearestKf && kf.keyframe_index === nearestKf.keyframe_index) {
      card.classList.add("active-target");
      card.style.border = "2px solid #58a6ff";
      card.style.boxShadow = "0 0 16px rgba(88, 166, 255, 0.75)";
      targetCardEl = card;
    }

    const sec = kf.keyframe_index / (fps || 25.0);
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    const timeStr = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;

    card.innerHTML = `
      <div class="catalog-keyframe-img-wrap" style="width: 100%; aspect-ratio: 16 / 9; min-height: 90px; background: #21262d; position: relative; overflow: hidden;">
        <img src="/keyframes/${kf.video_id}/keyframe_${kf.keyframe_index}.webp" class="catalog-keyframe-img" style="width: 100%; height: 100%; object-fit: cover; display: block;" loading="lazy" onerror="this.onerror=null;this.src='/static/placeholder.png';">
      </div>
      <div class="catalog-keyframe-footer" style="padding: 6px 10px; display: flex; align-items: center; justify-content: space-between; background: #161b22; border-top: 1px solid #21262d; font-size: 11px;">
        <span class="catalog-keyframe-idx" style="font-weight: 600; color: #f0f6fc;">#${kf.keyframe_index}</span>
        <span class="catalog-keyframe-time" style="color: #8b949e;">⏱️ ${timeStr}</span>
      </div>
    `;

    card.addEventListener("click", () => {
      openDirectVideo(kf.video_id, kf.keyframe_index);
    });

    container.appendChild(card);
  });

  if (targetCardEl) {
    setTimeout(() => {
      targetCardEl.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 150);
  }
}

function filterGalleryKeyframes() {
  const filterVal = document.getElementById("gallery-frame-filter")?.value.trim() || "";
  const fps = selectedVideoId ? (allCatalogVideos.find((v) => v.video_id === selectedVideoId)?.fps || 25) : 25;
  if (!filterVal) {
    renderGalleryKeyframes(currentKeyframes, fps);
    return;
  }

  const filtered = currentKeyframes.filter((kf) => String(kf.keyframe_index).includes(filterVal));
  renderGalleryKeyframes(filtered, fps);
}
