import { elements } from "./elements.js";
import { openModal, openCaptionModal } from "./video-player.js?v=10";
import {
  getSubmissionButtonLabel,
  handleSequenceSubmission,
  handleSubmissionCandidate,
} from "./submission.js?v=12";
import { getSubmissionMode } from "./dres-session.js?v=12";

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
    const ocrText = items.find((i) => i.ocr_text && i.ocr_text.trim())?.ocr_text || "";

    grouped.push({
      type: "shot",
      video_id: anchor.video_id,
      score: avgScore,
      frames: items,
      display_frames: items,
      keyframe_index: anchor.keyframe_index,
      shot_start_frame: anchor.shot_start_frame || 0,
      shot_end_frame: anchor.shot_end_frame || 0,
      fps: anchor.fps || 25.0,
      ocr_text: ocrText
    });
  });

  grouped.sort((a, b) => b.score - a.score);
  return grouped;
}

function groupResultsByVideo(flatResults) {
  const videoMap = new Map();

  flatResults.forEach((item) => {
    const videoKey = item.video_id;
    if (!videoMap.has(videoKey)) {
      videoMap.set(videoKey, []);
    }
    videoMap.get(videoKey).push(item);
  });

  const grouped = [];
  videoMap.forEach((items, video_id) => {
    items.sort((a, b) => a.keyframe_index - b.keyframe_index);
    const avgScore = items.reduce((sum, i) => sum + (i.score || 0), 0) / items.length;
    const anchor = items[0];
    const ocrText = items.find((i) => i.ocr_text && i.ocr_text.trim())?.ocr_text || "";

    grouped.push({
      type: "video",
      video_id: video_id,
      score: avgScore,
      frames: items,
      display_frames: items,
      keyframe_index: anchor.keyframe_index,
      shot_start_frame: items[0].keyframe_index,
      shot_end_frame: items[items.length - 1].keyframe_index,
      fps: anchor.fps || 25.0,
      ocr_text: ocrText
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

function groupSequencesByVideo(sequenceResults) {
  const videoMap = new Map();

  sequenceResults.forEach((seq) => {
    const vid = seq.video_id;
    if (!videoMap.has(vid)) {
      videoMap.set(vid, []);
    }
    videoMap.get(vid).push(seq);
  });

  const grouped = [];
  videoMap.forEach((seqList, vid) => {
    const totalScore = seqList.reduce((sum, s) => sum + (s.sequence_score || s.score || 0), 0);
    const videoAvgScore = totalScore / seqList.length;

    const bestSeq = seqList.reduce((max, s) => 
      ((s.best_sequence_score || s.sequence_score || s.score || 0) > 
       (max.best_sequence_score || max.sequence_score || max.score || 0)) ? s : max, seqList[0]);

    grouped.push({
      type: "video",
      video_id: vid,
      sequence_score: videoAvgScore,
      best_sequence_score: bestSeq.best_sequence_score || bestSeq.sequence_score || bestSeq.score,
      frames: bestSeq.frames,
      display_frames: bestSeq.display_frames || bestSeq.frames,
      all_sequences_count: seqList.length,
      keyframe_index: bestSeq.keyframe_index,
      score: videoAvgScore,
    });
  });

  grouped.sort((a, b) => b.sequence_score - a.sequence_score);
  return grouped;
}

export function displayResults(results, groupMode = "none") {
  elements.resultsContainer.innerHTML = "";

  if (!results || results.length === 0) {
    elements.resultsContainer.innerHTML =
      '<p style="padding:10px;">No results found.</p>';
    return;
  }

  // Display translation banner if English query was translated to Vietnamese
  if (results[0] && results[0].translated_query) {
    const banner = document.createElement("div");
    banner.style.cssText = "grid-column: 1 / -1; width: 100%; box-sizing: border-box; background: rgba(56, 139, 253, 0.12); border: 1px solid #1f6feb; border-radius: 6px; padding: 10px 14px; margin-bottom: 6px; color: #c9d1d9; font-size: 13px; display: flex; align-items: center; gap: 8px;";
    banner.innerHTML = `<span style="font-size:16px;">🌐</span><span><strong>Đã tự động dịch query sang Tiếng Việt:</strong> <em style="color:#58a6ff; font-weight:600;">"${results[0].translated_query}"</em></span>`;
    elements.resultsContainer.appendChild(banner);
  }

  // Check if these are caption-only results
  const isCaptionResult = results[0] && results[0].result_type === "caption";
  const isGrouped = Boolean(results[0] && results[0].frames);
  const isTemporal = Boolean(results[0] && results[0].sequence_score !== undefined);

  if (isCaptionResult) {
    if (groupMode === "video") {
      const groupedData = groupResultsByVideo(results);
      displaySequenceResults(groupedData);
    } else {
      displayCaptionResults(results);
    }
  } else if (groupMode === "video") {
    const groupedData = (isGrouped && isTemporal)
      ? groupSequencesByVideo(results)
      : groupResultsByVideo(isGrouped ? flattenGroupedResults(results) : results);
    displaySequenceResults(groupedData);
  } else if (groupMode === "shot" || groupMode === true) {
    const flatData = isGrouped ? flattenGroupedResults(results) : results;
    const groupedData = groupResultsByShot(flatData);
    displaySequenceResults(groupedData);
  } else if (isGrouped && isTemporal) {
    displaySequenceResults(results);
  } else if (isGrouped) {
    // When results were shot-grouped but user turned groupMode to "none", un-group and show flat keyframes
    const flatData = flattenGroupedResults(results);
    displayFlatResults(flatData);
  } else {
    displayFlatResults(results);
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

    const fpsStr = typeof item.fps === 'number' ? (Number.isInteger(item.fps) ? item.fps : item.fps.toFixed(2)) : item.fps;

    // Fused search extra scores: show badges only for modalities that matched
    let fusedBadges = [];
    if (item.result_type === "fused") {
      if (typeof item.keyframe_score === 'number' && item.keyframe_score > 0) {
        fusedBadges.push(`<span style="color:#a371f7;">KF: ${item.keyframe_score.toFixed(3)}</span>`);
      }
      if (typeof item.caption_score === 'number' && item.caption_score > 0) {
        fusedBadges.push(`<span style="color:#f0883e;">Cap: ${item.caption_score.toFixed(3)}</span>`);
      }
    }
    const fusedScoreHTML = fusedBadges.join("");

    // Best aspect caption for fused results
    const captionSnippet = item.best_aspect_caption
      ? `<div style="font-size:11px; color:#a371f7; margin-top:3px; max-height:2.6em; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; word-break:break-word;" title="${item.best_aspect_caption}">📝 <strong>${item.best_aspect || 'Caption'}:</strong> ${item.best_aspect_caption}</div>`
      : "";

    // 2. Nội dung Card
    const infoHTML = `
            <img src="${imageUrl}" class="result-item-image" onerror="this.onerror=null;this.src='/static/placeholder.png';">
            <div class="result-info">
                <h3>${item.video_id} / ${item.keyframe_index}</h3>
                <div class="result-scores">
                    <span>FPS: ${fpsStr}</span>
                    ${["score"]
                      .map((score) => {
                        const val = item[score] ? item[score].toFixed(3) : null;
                        return val
                          ? `<span>${"Score"}: ${val}</span>`
                          : "";
                      })
                      .join("")}
                    ${fusedScoreHTML}
                    ${item.frames ? `<span style="color:#58a6ff; font-weight:bold;">🖼️ Keyframes: ${item.frames.length}</span>` : ""}
                </div>
                ${captionSnippet}
                ${item.ocr_text ? `<div style="font-size:11px; color:#58a6ff; margin-top:3px; max-height:2.6em; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; word-break:break-word;" title="${item.ocr_text}">🔍 <strong>OCR:</strong> ${item.ocr_text}</div>` : ""}
                ${item.audio_text ? `<div style="font-size:11px; color:#e3b341; margin-top:3px; max-height:2.6em; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; word-break:break-word;" title="${item.audio_text}">🎙️ <strong>Audio:</strong> ${item.audio_text}</div>` : ""}
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
    const sequenceFrames = Array.isArray(item.temporal_sequence)
      ? item.temporal_sequence
      : (Array.isArray(item.frames) ? item.frames : null);
    const isTrafficVideo = (item.video_id || "").startsWith("N");
    if (isTrafficVideo) {
      submitBtn.title = "Video giao thông (VFR): Click để mở video player và nộp thời gian thực tế";
    }

    if (sequenceFrames?.length) {
      submitBtn.dataset.sequence = "true";
      submitBtn.textContent = getSubmissionButtonLabel({ sequence: true });
      submitBtn.addEventListener("click", (e) => {
        if (isTrafficVideo && getSubmissionMode() === "trake") {
          e.stopPropagation();
          alert("⚠️ Task TRAKE không sử dụng video giao thông (prefix N) theo quy định của BTC!");
          return;
        }
        handleSequenceSubmission(e, sequenceFrames);
      });
    } else {
      submitBtn.textContent = getSubmissionButtonLabel();
      submitBtn.addEventListener("click", (e) => {
        if (isTrafficVideo) {
          e.stopPropagation();
          if (getSubmissionMode() === "trake") {
            alert("⚠️ Task TRAKE không sử dụng video giao thông (prefix N) theo quy định của BTC!");
            return;
          }
          const fps = parseFloat(item.fps) || 25;
          let startTime = item.timeMs ? (item.timeMs / 1000) : (item.keyframe_index / fps);
          startTime = Math.max(0, startTime - 0.5);
          openModal(
            item.video_id,
            startTime,
            fps,
            null,
            item.keyframe_index,
            item.temporal_sequence || item.frames,
          );
          alert(`⚠️ Video giao thông ${item.video_id} (VFR):\nĐã mở Trình phát video tại khung hình này. Vui lòng kiểm tra và bấm "Nộp vị trí đang dừng" trên video player để lấy thời gian thực tế chính xác nhất.`);
          return;
        }
        handleSubmissionCandidate(e, item);
      });
    }

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
        item.temporal_sequence || item.frames,
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
      const videoUrl = `/video/${videoId}`;

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
    let displayFrames = seq.display_frames || seq.frames || [];
    const maxThumbs = 9;
    const thumbsToRender = displayFrames.length > maxThumbs ? displayFrames.slice(0, maxThumbs) : displayFrames;
    const numFrames = thumbsToRender.length;
    let gridStyle = "";
    if (numFrames > 4) {
      const cols = Math.ceil(Math.sqrt(numFrames));
      const rows = Math.ceil(numFrames / cols);
      gridStyle = `style="grid-template-columns: repeat(${cols}, 1fr); grid-template-rows: repeat(${rows}, 1fr);"`;
    }
    const gridClass = `items-${numFrames > 4 ? 4 : numFrames}`;
    
    let gridHTML = `<div class="shot-thumbnails-grid ${gridClass}" ${gridStyle}>`;
    thumbsToRender.forEach((itm) => {
      gridHTML += `<div style="position:relative; width:100%; height:100%;">
        <img src="/keyframes/${itm.video_id}/keyframe_${itm.keyframe_index}.webp" loading="lazy" style="width:100%; height:100%; object-fit:cover;">
        <span style="position:absolute; bottom:2px; right:2px; background:rgba(0,0,0,0.7); color:white; font-size:10px; padding:2px; border-radius:2px;">${itm.keyframe_index}</span>
      </div>`;
    });
    gridHTML += `</div>`;

    // Info
    const totalKeyframes = seq.frames ? seq.frames.length : numFrames;
    const isVideoGroupedCard = Boolean(seq.all_sequences_count && seq.all_sequences_count > 1);
    const scoreLabel = isVideoGroupedCard ? "Video Avg Score" : "Score";
    const bestScoreHTML = (isVideoGroupedCard && seq.best_sequence_score)
      ? `<br><span style="font-size:11px; color:#238636; font-weight:bold;">Best Pair Score: ${seq.best_sequence_score.toFixed(3)}</span>`
      : "";
    const pairsCountHTML = isVideoGroupedCard
      ? `<br><span style="font-size:11px; color:#8b949e;">Pairs in Video: ${seq.all_sequences_count}</span>`
      : "";

    const infoHTML = `
            <div class="shot-info">
                <h3>${seq.video_id}</h3>
                <div class="shot-stats">
                    <strong style="color:#58a6ff;">🖼️ Keyframes:</strong> ${totalKeyframes}<br>
                    <strong>${scoreLabel}:</strong> ${(seq.sequence_score || seq.score).toFixed(3)}${bestScoreHTML}${pairsCountHTML}
                </div>
                <button class="card-submit-btn" data-sequence="true">Submit</button>
            </div>
        `;

    card.innerHTML = gridHTML + infoHTML;

    // KIS/QA uses the anchor; TRAKE loads the whole semantic sequence.
    const submitBtn = card.querySelector(".card-submit-btn");
    if (submitBtn) {
        submitBtn.textContent = getSubmissionButtonLabel({ sequence: true });
        submitBtn.addEventListener("click", (e) => handleSequenceSubmission(e, seq.frames || displayFrames));
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

// --- CAPTION SEARCH RESULTS ---
function displayCaptionResults(results) {
  results.forEach((item) => {
    const card = document.createElement("div");
    card.classList.add("result-item");
    card.dataset.videoId = item.video_id;
    card.dataset.keyframeIndex = item.keyframe_index;

    const imageUrl = `/keyframes/${item.video_id}/keyframe_${item.keyframe_index}.webp`;
    const fpsStr = typeof item.fps === 'number' ? (Number.isInteger(item.fps) ? item.fps : item.fps.toFixed(2)) : (item.fps || 25);

    // Best aspect caption (only show the highest scoring aspect)
    const bestAspectLabel = item.best_aspect ? item.best_aspect.replace(/_/g, ' ') : 'Caption';
    const captionText = item.best_aspect_caption || item.caption_text || '';
    const captionSnippet = captionText
      ? `<div style="font-size:11px; color:#a371f7; margin-top:3px; max-height:3.9em; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; word-break:break-word;" title="${captionText.replace(/"/g, '&quot;')}">📝 <strong>${bestAspectLabel}:</strong> ${captionText}</div>`
      : '';

    const shotRange = (item.shot_start_frame !== undefined && item.shot_end_frame !== undefined)
      ? `<span style="color:#8b949e; font-size:11px;">Shot: ${item.shot_start_frame}–${item.shot_end_frame}</span>`
      : '';

    card.innerHTML = `
      <img src="${imageUrl}" class="result-item-image" onerror="this.onerror=null;this.src='/static/placeholder.png';">
      <div class="result-info">
        <h3>${item.video_id} / ${item.keyframe_index}</h3>
        <div class="result-scores">
          <span title="Dense: ${(item.dense_score || 0).toFixed(3)} | BM25: ${(item.bm25_score || 0).toFixed(3)}">Score: ${item.score.toFixed(3)}</span>
          ${shotRange}
        </div>
        ${captionSnippet}
        <button class="card-submit-btn" type="button">Submit</button>
      </div>`;

    // Submit
    const submitBtn = card.querySelector(".card-submit-btn");
    const isTrafficVideo = (item.video_id || "").startsWith("N");
    if (isTrafficVideo) {
      submitBtn.title = "Video giao thông (VFR): Click để mở video player và nộp thời gian thực tế";
    }
    submitBtn.textContent = getSubmissionButtonLabel();
    submitBtn.addEventListener("click", (e) => {
      if (isTrafficVideo) {
        e.stopPropagation();
        if (getSubmissionMode() === "trake") {
          alert("⚠️ Task TRAKE không sử dụng video giao thông (prefix N) theo quy định của BTC!");
          return;
        }
        const fps = parseFloat(item.fps) || 25;
        openCaptionModal(item, fps);
        alert(`⚠️ Video giao thông ${item.video_id} (VFR):\nĐã mở Trình phát video. Vui lòng kiểm tra và bấm "Nộp vị trí đang dừng" trên video player để lấy thời gian thực tế chính xác nhất.`);
        return;
      }
      handleSubmissionCandidate(e, item);
    });

    // Click -> open modal with ALL keyframes in this shot
    card.addEventListener("click", (e) => {
      if (e.target.tagName.toLowerCase() === 'button') return;
      const fps = parseFloat(item.fps) || 25;
      openCaptionModal(item, fps);
    });

    elements.resultsContainer.appendChild(card);
  });
}
