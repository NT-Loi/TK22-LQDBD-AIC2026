import { elements } from "./elements.js";
import { initFilters, getObjectQueries } from "./filters.js";
import { searchAPI, loginAPI } from "./api.js";
import { displayResults } from "./results.js";
import { initVideoModal } from "./video-player.js";

let currentResults = [];
let isGroupShots = false;
let isGroupVideo = false;

function getGroupMode() {
  if (isGroupVideo) return "video";
  if (isGroupShots) return "shot";
  return "none";
}

function updateGroupButtonsUI() {
  if (elements.toggleGroupShotsBtn) {
    elements.toggleGroupShotsBtn.textContent = isGroupShots ? "Group Shots: ON" : "Group Shots: OFF";
    elements.toggleGroupShotsBtn.classList.toggle("active", isGroupShots);
  }
  if (elements.toggleGroupVideoBtn) {
    elements.toggleGroupVideoBtn.textContent = isGroupVideo ? "Group Video: ON" : "Group Video: OFF";
    elements.toggleGroupVideoBtn.classList.toggle("active", isGroupVideo);
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  initFilters();
  initVideoModal();
  initDynamicInputs(); // Khởi tạo logic thêm bớt input

  // --- POPULATE MODEL(S) ---
  try {
    const res = await fetch("/api/models");
    if (res.ok) {
      const data = await res.json();
      const modelsSelect = document.getElementById("models-select");
      if (modelsSelect) {
        modelsSelect.innerHTML = '<option value="all" selected>All</option>';
        data.models.forEach(model => {
          if (model !== "score" && model !== "all") {
            const option = document.createElement("option");
            option.value = model;
            option.textContent = `${model}`;
            modelsSelect.appendChild(option);
          }
        });

        // Add event listener for mutual exclusivity
        let previousSelection = ["all"];
        modelsSelect.addEventListener("change", (e) => {
          const options = Array.from(modelsSelect.options);
          let currentSelection = options.filter(o => o.selected).map(o => o.value);
          
          if (currentSelection.includes("all") && !previousSelection.includes("all")) {
            // User just clicked 'All'
            options.forEach(o => o.selected = (o.value === "all"));
            currentSelection = ["all"];
          } else if (currentSelection.includes("all") && currentSelection.length > 1) {
            // User clicked something else while 'All' was selected
            options.find(o => o.value === "all").selected = false;
            currentSelection = currentSelection.filter(v => v !== "all");
          } else if (currentSelection.length === 0) {
            // Nothing selected, default to All
            options.find(o => o.value === "all").selected = true;
            currentSelection = ["all"];
          }
          previousSelection = currentSelection;
        });
      }
    }
  } catch (error) {
    console.error("Failed to load models for models-select:", error);
  }

  // --- TOGGLE GROUP SHOTS & VIDEO ---
  if (elements.toggleGroupShotsBtn) {
    elements.toggleGroupShotsBtn.addEventListener("click", () => {
      isGroupShots = !isGroupShots;
      if (isGroupShots) isGroupVideo = false;
      updateGroupButtonsUI();
      displayResults(currentResults, getGroupMode());
    });
  }

  if (elements.toggleGroupVideoBtn) {
    elements.toggleGroupVideoBtn.addEventListener("click", () => {
      isGroupVideo = !isGroupVideo;
      if (isGroupVideo) isGroupShots = false;
      updateGroupButtonsUI();
      displayResults(currentResults, getGroupMode());
    });
  }

  // --- LOGIN LOGIC (Giữ nguyên) ---
  if (elements.loginBtn) {
    elements.loginBtn.addEventListener("click", async () => {
      elements.loginBtn.textContent = "Logging in...";
      try {
        const data = await loginAPI();
        if (!data.evaluations || data.evaluations.length === 0) {
          alert("No active evaluations found.");
          elements.loginBtn.textContent = "Login";
          return;
        }
        showEvaluationModal(data.evaluations, data.sessionId);
      } catch (error) {
        alert(`Login Failed: ${error.message}`);
        elements.loginBtn.textContent = "Login failed";
      }
    });
  }

  // --- SEARCH SUBMIT ---
  const handleSearch = async (e) => {
    if (e) e.preventDefault();

    // 1. Thu thập các câu query text
    const textQueries = [];
    const rows = elements.queryInputsContainer.querySelectorAll(".query-row");
    let anchorIndex = 0;

    rows.forEach((row, index) => {
      const input = row.querySelector(".main-query-input");
      const text = input ? input.value.trim() : "";

      if (text) {
        const ocrInput = row.querySelector(".event-ocr-input");
        const ocrLevel = row.querySelector(".event-ocr-level");
        const audioInput = row.querySelector(".event-audio-input");
        const audioLevel = row.querySelector(".event-audio-level");

        const ocrTxt = ocrInput ? ocrInput.value.trim() : "";
        const audioTxt = audioInput ? audioInput.value.trim() : "";

        const eventObj = { text: text };
        if (ocrTxt) {
          eventObj.ocr = [{ text: ocrTxt, level: ocrLevel ? ocrLevel.value : "frame" }];
        }
        if (audioTxt) {
          eventObj.audio = [{ text: audioTxt, level: audioLevel ? audioLevel.value : "frame" }];
        }

        textQueries.push(eventObj);
      }
    });

    // Build payload
    const modelsSelect = document.getElementById("models-select");
    let selectedModels = ["all"];
    if (modelsSelect) {
      selectedModels = Array.from(modelsSelect.options).filter(o => o.selected).map(o => o.value);
    }

    // 2. Thu thập các audio query kèm level của từng dòng
    const audioQueries = [];
    const audioRows = elements.audioInputsContainer ? elements.audioInputsContainer.querySelectorAll(".audio-row") : [];
    audioRows.forEach((row) => {
      const input = row.querySelector(".audio-query-input");
      const levelSelect = row.querySelector(".audio-level-select");
      const text = input ? input.value.trim() : "";
      const level = levelSelect ? levelSelect.value : "frame";
      if (text) {
        audioQueries.push({ text: text, level: level });
      }
    });

    // 3. Thu thập các OCR query kèm level của từng dòng
    const ocrQueries = [];
    const ocrRows = elements.ocrInputsContainer ? elements.ocrInputsContainer.querySelectorAll(".ocr-row") : [];
    ocrRows.forEach((row) => {
      const input = row.querySelector(".ocr-query-input");
      const levelSelect = row.querySelector(".ocr-level-select");
      const text = input ? input.value.trim() : "";
      const level = levelSelect ? levelSelect.value : "frame";
      if (text) {
        ocrQueries.push({ text: text, level: level });
      }
    });

    const thresholdInput = document.getElementById("score-threshold")?.value;
    const parsedThreshold = parseFloat(thresholdInput);
    const scoreThreshold = isNaN(parsedThreshold) ? 0.0 : parsedThreshold;

    const queryData = {
      text_queries: textQueries, // List of objects/strings
      anchor_index: anchorIndex, // Index of query used for sorting
      models: selectedModels,
      objects: getObjectQueries(),
      audio: audioQueries, // List of objects: [{text: "...", level: "frame"|"video"}]
      ocr_query: ocrQueries, // List of objects: [{text: "...", level: "frame"|"video"}]
      group_by_shot: isGroupShots,
      group_by_video: isGroupVideo,
      score_threshold: scoreThreshold,
      limit: parseInt(document.getElementById("result-limit")?.value) || 100
    };

    if (
      queryData.text_queries.length === 0 &&
      queryData.objects.length === 0 &&
      queryData.ocr_query.length === 0 &&
      queryData.audio.length === 0
    ) {
      alert("Please enter at least one text query, OCR filter, Audio filter, or Object filter.");
      return;
    }

    elements.resultsContainer.innerHTML = "<p>Searching sequence...</p>";
    const results = await searchAPI(queryData);
    currentResults = results;

    const countEl = document.getElementById("results-count");
    if (countEl) {
      countEl.textContent = `Results: ${currentResults.length}`;
    }

    displayResults(currentResults, getGroupMode());
  };

  if (elements.searchForm) {
    elements.searchForm.addEventListener("submit", handleSearch);
  }

  const searchBtn = document.getElementById("search-btn");
  if (searchBtn) {
    searchBtn.addEventListener("click", handleSearch);
  }

  // --- SCROLL TOP ---
  const scrollTopBtn = document.getElementById("scroll-top-btn");
  if (scrollTopBtn) {
    scrollTopBtn.addEventListener("click", (e) => {
      e.preventDefault();
      const resultsArea = document.querySelector(".results-area");
      if (resultsArea) resultsArea.scrollTo({ top: 0, behavior: "instant" });
      scrollTopBtn.blur();
    });
  }
});

// --- LOGIC INPUT ĐỘNG ---
function initDynamicInputs() {
  if (!elements.addQueryBtn) return;

  // Toggle event filter panel
  if (elements.queryInputsContainer) {
    elements.queryInputsContainer.addEventListener("click", (e) => {
      const toggleBtn = e.target.closest(".toggle-event-filter-btn");
      if (toggleBtn) {
        const row = toggleBtn.closest(".query-row");
        const panel = row ? row.querySelector(".event-filter-panel") : null;
        if (panel) {
          panel.style.display = panel.style.display === "none" ? "block" : "none";
        }
      }
      if (e.target.classList.contains("remove-query-btn")) {
        e.target.closest(".query-row").remove();
        reindexRows();
      }
    });
  }

  elements.addQueryBtn.addEventListener("click", () => {
    const rows = elements.queryInputsContainer.querySelectorAll(".query-row");
    const newIndex = rows.length;
    const div = document.createElement("div");
    div.className = "search-row query-row";
    div.dataset.index = newIndex;
    div.style.flexDirection = "column";
    div.style.alignItems = "stretch";
    div.style.marginTop = "5px";

    div.innerHTML = `
      <div style="display: flex; gap: 5px;">
        <input type="text" name="description_${newIndex}" class="main-query-input" placeholder="Next Event (approx. 1 min later)..." autocomplete="off" style="flex:1;">
        <button type="button" class="toggle-event-filter-btn" style="background: #238636; color: white; border: none; border-radius: 4px; padding: 0 8px; font-size: 12px; cursor: pointer;" title="Event Sub-filters">⚙️ Filters</button>
        <button type="button" class="remove-query-btn" style="background:#dc3545; color:white; border:none; border-radius:4px; cursor:pointer; padding:0 8px;">X</button>
      </div>
      <div class="event-filter-panel" style="display: none; margin-top: 6px; padding: 6px; background: rgba(255,255,255,0.05); border-radius: 4px; border: 1px solid rgba(255,255,255,0.1);">
        <div style="margin-bottom: 4px;">
          <label style="font-size: 11px; color: #8b949e; display: block; margin-bottom: 2px;">📝 Event ${newIndex + 1} OCR Filter:</label>
          <div style="display: flex; gap: 4px;">
            <input type="text" class="event-ocr-input sidebar-input" placeholder="OCR for Event ${newIndex + 1}..." style="font-size: 11px; padding: 4px;">
            <select class="event-ocr-level sidebar-select-sm" style="width: 65px; font-size: 11px;">
              <option value="frame" selected>Frame</option>
              <option value="video">Video</option>
            </select>
          </div>
        </div>
        <div>
          <label style="font-size: 11px; color: #8b949e; display: block; margin-bottom: 2px;">🎙️ Event ${newIndex + 1} Audio Filter:</label>
          <div style="display: flex; gap: 4px;">
            <input type="text" class="event-audio-input sidebar-input" placeholder="Audio for Event ${newIndex + 1}..." style="font-size: 11px; padding: 4px;">
            <select class="event-audio-level sidebar-select-sm" style="width: 65px; font-size: 11px;">
              <option value="frame" selected>Frame</option>
              <option value="video">Video</option>
            </select>
          </div>
        </div>
      </div>
    `;

    elements.queryInputsContainer.appendChild(div);
    updateRemoveButtons();
  });

  elements.queryInputsContainer.addEventListener("click", (e) => {
    if (e.target.classList.contains("remove-query-btn")) {
      e.target.parentElement.remove();
      reindexRows();
    }
  });

  // Dynamic Audio Inputs
  if (elements.addAudioBtn && elements.audioInputsContainer) {
    elements.addAudioBtn.addEventListener("click", () => {
      const rows = elements.audioInputsContainer.querySelectorAll(".audio-row");
      const newIndex = rows.length;
      const div = document.createElement("div");
      div.className = "search-row audio-row";
      div.dataset.index = newIndex;
      div.style.marginTop = "5px";

      div.innerHTML = `
        <input type="text" name="audio_${newIndex}" class="sidebar-input audio-query-input" placeholder="Spoken words in video..." autocomplete="off">
        <select name="audio_level_${newIndex}" class="sidebar-select-sm audio-level-select" style="width: 72px; flex-shrink: 0;">
          <option value="frame" selected>Frame</option>
          <option value="video">Video</option>
        </select>
        <button type="button" class="remove-audio-btn" style="background:#dc3545; color:white; border:none; border-radius:4px; cursor:pointer; padding:0 8px;">X</button>
      `;

      elements.audioInputsContainer.appendChild(div);
      updateRemoveAudioButtons();
    });

    elements.audioInputsContainer.addEventListener("click", (e) => {
      if (e.target.classList.contains("remove-audio-btn")) {
        e.target.parentElement.remove();
        updateRemoveAudioButtons();
      }
    });
  }

  // Dynamic OCR Inputs
  if (elements.addOcrBtn && elements.ocrInputsContainer) {
    elements.addOcrBtn.addEventListener("click", () => {
      const rows = elements.ocrInputsContainer.querySelectorAll(".ocr-row");
      const newIndex = rows.length;
      const div = document.createElement("div");
      div.className = "search-row ocr-row";
      div.dataset.index = newIndex;
      div.style.marginTop = "5px";

      div.innerHTML = `
        <input type="text" name="ocr_${newIndex}" class="sidebar-input ocr-query-input" placeholder="Text visible on screen..." autocomplete="off">
        <select name="ocr_level_${newIndex}" class="sidebar-select-sm ocr-level-select" style="width: 72px; flex-shrink: 0;">
          <option value="frame" selected>Frame</option>
          <option value="video">Video</option>
        </select>
        <button type="button" class="remove-ocr-btn" style="background:#dc3545; color:white; border:none; border-radius:4px; cursor:pointer; padding:0 8px;">X</button>
      `;

      elements.ocrInputsContainer.appendChild(div);
      updateRemoveOcrButtons();
    });

    elements.ocrInputsContainer.addEventListener("click", (e) => {
      if (e.target.classList.contains("remove-ocr-btn")) {
        e.target.parentElement.remove();
        updateRemoveOcrButtons();
      }
    });
  }
}

function updateRemoveOcrButtons() {
  if (!elements.ocrInputsContainer) return;
  const rows = elements.ocrInputsContainer.querySelectorAll(".ocr-row");
  rows.forEach((row) => {
    const btn = row.querySelector(".remove-ocr-btn");
    if (btn) btn.style.display = rows.length > 1 ? "block" : "none";
  });
}

function updateRemoveButtons() {
  // Chỉ hiện nút X nếu có > 1 dòng
  const rows = elements.queryInputsContainer.querySelectorAll(".query-row");
  rows.forEach((row) => {
    const btn = row.querySelector(".remove-query-btn");
    if (btn) btn.style.display = rows.length > 1 ? "block" : "none";
  });
}

function updateRemoveAudioButtons() {
  if (!elements.audioInputsContainer) return;
  const rows = elements.audioInputsContainer.querySelectorAll(".audio-row");
  rows.forEach((row) => {
    const btn = row.querySelector(".remove-audio-btn");
    if (btn) btn.style.display = rows.length > 1 ? "block" : "none";
  });
}

function reindexRows() {
  const rows = elements.queryInputsContainer.querySelectorAll(".query-row");
  rows.forEach((row, index) => {
    row.dataset.index = index;

    const input = row.querySelector(".main-query-input");
    input.name = `description_${index}`;
    input.placeholder = index === 0 ? "Event 1..." : "Next Event...";
  });
}

// ... (Giữ nguyên phần showEvaluationModal) ...
function showEvaluationModal(evaluations, sessionId) {
  elements.evalListContainer.innerHTML = "";
  evaluations.forEach((ev) => {
    const btn = document.createElement("button");
    btn.textContent = `${ev.name} (${ev.status})`;
    btn.style.width = "100%";
    btn.style.margin = "5px 0";
    btn.onclick = () => {
      localStorage.setItem("sessionId", sessionId);
      localStorage.setItem("evaluationId", ev.id);
      elements.evalModal.classList.add("hidden");
      elements.loginBtn.textContent = `Logged: ${ev.name}`;
      elements.loginBtn.style.background = "#28a745";
    };
    elements.evalListContainer.appendChild(btn);
  });
  elements.evalModal.classList.remove("hidden");

  if (elements.cancelEvalBtn) {
    elements.cancelEvalBtn.onclick = () => {
      elements.evalModal.classList.add("hidden");
      elements.loginBtn.textContent = "Login";
    };
  }
}
