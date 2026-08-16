import { elements } from "./elements.js";
import { openModal } from "./video-player.js";
import { submitResultAPI } from "./api.js";

// Helper: Shuffle array
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

function groupResultsByShot(flatResults) {
  const shotMap = new Map();

  flatResults.forEach((item) => {
    const hasShotInfo = item.shot_start_frame !== undefined && item.shot_end_frame !== undefined && item.shot_start_frame !== item.shot_end_frame;
    const shotKey = hasShotInfo
      ? `${item.video_id}_${item.shot_start_frame}_${item.shot_end_frame}`
      : `${item.video_id}_single_${item.keyframe_index}`;

    if (!shotMap.has(shotKey)) {
      shotMap.set(shotKey, []);
    }
    shotMap.get(shotKey).push(item);
  });

  const grouped = [];
  shotMap.forEach((items) => {
    items.sort((a, b) => a.keyframe_index - b.keyframe_index);
    const avgScore = items.reduce((sum, i) => sum + (i.score || 0), 0) / items.length;
    const anchor = items[0];

    grouped.push({
      type: "shot",
      video_id: anchor.video_id,
      score: avgScore,
      frames: items,
      display_frames: items,
      keyframe_index: anchor.keyframe_index,
      shot_start_frame: anchor.shot_start_frame || 0,
      shot_end_frame: anchor.shot_end_frame || 0,
      fps: anchor.fps || 25.0
    });
  });

  grouped.sort((a, b) => b.score - a.score);
  return grouped;
}

function flattenGroupedResults(groupedResults) {
  const flatMap = new Map();

  groupedResults.forEach((group) => {
    const frames = group.frames || [group];
    frames.forEach((frame) => {
      const key = `${frame.video_id}_${frame.keyframe_index}`;
      if (!flatMap.has(key)) {
        flatMap.set(key, frame);
      }
    });
  });

  const flat = Array.from(flatMap.values());
  flat.sort((a, b) => (b.score || 0) - (a.score || 0));
  return flat;
}

export function displayResults(results, groupShots = false) {
  elements.resultsContainer.innerHTML = "";

  if (!results || results.length === 0) {
    elements.resultsContainer.innerHTML =
      '<p style="padding:10px;">No results found.</p>';
    return;
  }

  const isGrouped = Boolean(results[0] && results[0].frames);

  if (groupShots) {
    const groupedData = isGrouped ? results : groupResultsByShot(results);
    displaySequenceResults(groupedData);
  } else {
    const flatData = isGrouped ? flattenGroupedResults(results) : results;
    displayFlatResults(flatData);
  }
}

// --- HIỂN THỊ DẠNG DANH SÁCH THƯỜNG ---
function displayFlatResults(results) {
  const scoreLabels = {
    score: "Score",
  };

  results.forEach((item) => {
    const resultElement = document.createElement("div");
    resultElement.classList.add("result-item");
    resultElement.dataset.videoId = item.video_id;
    resultElement.dataset.keyframeIndex = item.keyframe_index;

    const imageUrl = `/keyframes/${item.video_id}/keyframe_${item.keyframe_index}.webp`;

    // 1. Hover Preview Container
    const previewContainer = document.createElement("div");
    previewContainer.className = "hover-preview";
    const previewVideo = document.createElement("video");
    previewVideo.muted = true;
    previewVideo.playsInline = true;
    // Style inline để đảm bảo object-fit
    Object.assign(previewVideo.style, {
      width: "100%",
      height: "100%",
      objectFit: "cover",
    });
    previewContainer.appendChild(previewVideo);

    // 2. Nội dung Card
    const infoHTML = `
            <img src="${imageUrl}" class="result-item-image" onerror="this.onerror=null;this.src='/static/placeholder.png';">
            <div class="result-info">
                <h3>${item.video_id} / ${item.keyframe_index}</h3>
                <div class="result-scores">
                    <span>FPS: ${item.fps}</span>
                    ${["score"]
                      .map((score) => {
                        const val = item[score] ? item[score].toFixed(3) : null;
                        return val
                          ? `<span>${scoreLabels[score]}: ${val}</span>`
                          : "";
                      })
                      .join("")}
                </div>
                ${item.temporal_sequence ? `<div style="font-size:11px; color:blue; margin-top:2px; font-weight:bold;">🔗 Sequence: ${item.temporal_sequence.length} events</div>` : ""}
                <button class="card-submit-btn" type="button">Submit</button>
            </div>`;

    resultElement.innerHTML = infoHTML;

    // Chèn Preview vào đầu (để nó đè lên ảnh nhờ CSS absolute)
    resultElement.insertBefore(previewContainer, resultElement.firstChild);

    // Setup Hover logic
    setupHoverPreview(resultElement, previewVideo, item);

    // Submit Handler
    const submitBtn = resultElement.querySelector(".card-submit-btn");
    submitBtn.addEventListener("click", (e) => handleSubmit(e, item));

    // Open Modal
    resultElement.addEventListener("click", () => {
      const fps = parseFloat(item.fps) || 25;
      let startTime = item.keyframe_index / fps;
      startTime = Math.max(0, startTime - 0.5);

      // Truyền sequenceData vào tham số cuối
      openModal(
        item.video_id,
        startTime,
        fps,
        null,
        item.keyframe_index,
        item.temporal_sequence,
      );
    });

    elements.resultsContainer.appendChild(resultElement);
  });
}



// --- HELPER FUNCTIONS ---
function setupHoverPreview(element, videoEl, item) {
  let hls = null;
  let hoverTimeout;

  const cleanup = () => {
    if (hls) {
      hls.destroy();
      hls = null;
    }
    videoEl.pause();
    videoEl.removeAttribute("src");
    videoEl.load();
    // Reset lại trạng thái ẩn video khi chuột rời đi
    videoEl.classList.remove("is-playing");
    videoEl.style.opacity = 0;
  };

  // Hàm xử lý khi video đã thực sự có hình
  const onVideoReady = () => {
    videoEl.classList.add("is-playing");
    videoEl.style.opacity = 1;
  };

  element.addEventListener("mouseenter", () => {
    hoverTimeout = setTimeout(() => {
      const videoId = item.video_id;
      const fps = item.fps || 25;
      const startTime = Math.max(0, item.keyframe_index / fps - 1.0); // Preview trước 1s
      const videoUrl = `/video/${videoId}.mp4`;

      // Lắng nghe sự kiện timeupdate hoặc playing để hiện video
      // timeupdate > 0 nghĩa là frame đã chạy, đảm bảo không bị màn hình đen
      videoEl.removeEventListener("timeupdate", onVideoReady); // Xóa listener cũ tránh duplicate
      videoEl.addEventListener("timeupdate", function checkFrame() {
        if (videoEl.currentTime > 0) {
          onVideoReady();
          videoEl.removeEventListener("timeupdate", checkFrame);
        }
      });

      videoEl.src = videoUrl;
      videoEl.currentTime = startTime;
      videoEl.play().catch((e) => {});
    }, 200); // Delay nhẹ tránh spam
  });

  element.addEventListener("mouseleave", () => {
    clearTimeout(hoverTimeout);
    cleanup();
  });
}

function displaySequenceResults(results) {
  results.forEach((seq) => {
    const card = document.createElement("div");
    card.classList.add("shot-group-card"); // Tái sử dụng class này cho layout grid
    
    // Use display_frames (best frame per shot) for card thumbnails, fallback to frames
    let displayFrames = seq.display_frames || seq.frames;
    const gridClass = `items-${displayFrames.length > 4 ? 4 : displayFrames.length}`;
    
    let gridHTML = `<div class="shot-thumbnails-grid ${gridClass}">`;
    displayFrames.slice(0, 4).forEach((itm) => {
      gridHTML += `<div style="position:relative; width:100%; height:100%;">
        <img src="/keyframes/${itm.video_id}/keyframe_${itm.keyframe_index}.webp" loading="lazy" style="width:100%; height:100%; object-fit:cover;">
        <span style="position:absolute; bottom:2px; right:2px; background:rgba(0,0,0,0.7); color:white; font-size:10px; padding:2px; border-radius:2px;">${itm.keyframe_index}</span>
      </div>`;
    });
    gridHTML += `</div>`;

    // Info
    const infoHTML = `
            <div class="shot-info">
                <h3>${seq.video_id}</h3>
                <div class="shot-stats">
                    Matches: ${seq.frames.length}<br>
                    Avg Score: ${(seq.sequence_score || seq.score).toFixed(3)}
                </div>
                <button class="card-submit-btn" style="margin-top: 5px; width:100%; font-size:12px; padding:4px;">Submit Anchor</button>
            </div>
        `;

    card.innerHTML = gridHTML + infoHTML;

    // Submit Handler for the anchor (the first display frame)
    const submitBtn = card.querySelector(".card-submit-btn");
    if (submitBtn) {
        submitBtn.addEventListener("click", (e) => handleSubmit(e, displayFrames[0]));
    }

    // Click to Open Modal — pass ALL frames so modal shows full detail
    card.addEventListener("click", (e) => {
      if (e.target.tagName.toLowerCase() === 'button') return;
      const anchor = displayFrames[0];
      const fps = parseFloat(anchor.fps) || 25;
      let startTime = anchor.keyframe_index / fps;
      startTime = Math.max(0, startTime - 0.5);

      openModal(anchor.video_id, startTime, fps, null, anchor.keyframe_index, seq.frames);
    });

    elements.resultsContainer.appendChild(card);
  });
}
async function handleSubmit(e, item) {
  e.stopPropagation();
  const sessionId = localStorage.getItem("sessionId");
  const evaluationId = localStorage.getItem("evaluationId");
  if (!sessionId || !evaluationId) {
    alert("Please LOGIN first!");
    return;
  }
  const confirmSubmit = confirm(
    `Submit frame ${item.keyframe_index} of ${item.video_id}?`,
  );
  if (!confirmSubmit) return;

  const fps = parseFloat(item.fps) || 25.0;
  const timeMs = Math.round((item.keyframe_index / fps) * 1000);

  try {
    const res = await submitResultAPI(
      sessionId,
      evaluationId,
      item.video_id,
      timeMs,
    );
    alert(`Success!`);
  } catch (err) {
    alert(`Submit Failed: ${err.message}`);
  }
}
