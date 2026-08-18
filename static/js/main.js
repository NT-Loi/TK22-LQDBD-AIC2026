import { elements } from "./elements.js";
import { initFilters, getObjectQueries } from "./filters.js";
import { searchAPI, loginAPI } from "./api.js";
import { displayResults } from "./results.js";
import { initVideoModal } from "./video-player.js";

let currentResults = [];
let isGroupShots = false;

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

  // --- TOGGLE GROUP SHOTS ---
  if (elements.toggleGroupShotsBtn) {
    elements.toggleGroupShotsBtn.addEventListener("click", () => {
      isGroupShots = !isGroupShots;
      elements.toggleGroupShotsBtn.textContent = isGroupShots
        ? "Group Shots: ON"
        : "Group Shots: OFF";
      elements.toggleGroupShotsBtn.classList.toggle("active", isGroupShots);
      displayResults(currentResults, isGroupShots);
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
  elements.searchForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    // 1. Thu thập các câu query text
    const textQueries = [];
    const rows = elements.queryInputsContainer.querySelectorAll(".query-row");
    let anchorIndex = 0;

    rows.forEach((row, index) => {
      const input = row.querySelector(".main-query-input");
      const text = input.value.trim();

      if (text) {
        textQueries.push(text);
      }
    });

    // Nếu người dùng chọn Rank dòng trống hoặc dòng k có text, fallback về dòng đầu tiên có text
    // Nhưng đơn giản nhất là gửi anchorIndex theo thứ tự đã filter
    // Tuy nhiên, để chính xác, ta cần map đúng index của list textQueries.
    // Logic dưới đây giả định người dùng nhập liên tiếp.

    // Build payload
    const modelsSelect = document.getElementById("models-select");
    let selectedModels = ["all"];
    if (modelsSelect) {
      selectedModels = Array.from(modelsSelect.options).filter(o => o.selected).map(o => o.value);
    }

    const thresholdInput = document.getElementById("score-threshold")?.value;
    const parsedThreshold = parseFloat(thresholdInput);
    const scoreThreshold = isNaN(parsedThreshold) ? 0.0 : parsedThreshold;

    const queryData = {
      text_queries: textQueries, // List of strings
      anchor_index: anchorIndex, // Index of query used for sorting
      models: selectedModels,
      objects: getObjectQueries(),
      audio: document.getElementById("audio-query-input")?.value.trim() || "",
      ocr_query: document.getElementById("ocr-query-input")?.value.trim() || "",
      group_by_shot: isGroupShots,
      score_threshold: scoreThreshold,
      limit: parseInt(document.getElementById("result-limit")?.value) || 100
    };

    if (
      queryData.text_queries.length === 0 &&
      queryData.objects.length === 0 &&
      !queryData.ocr_query &&
      !queryData.audio
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

    displayResults(currentResults, isGroupShots);
  });

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

  elements.addQueryBtn.addEventListener("click", () => {
    const rows = elements.queryInputsContainer.querySelectorAll(".query-row");
    const newIndex = rows.length;
    const div = document.createElement("div");
    div.className = "search-row query-row";
    div.dataset.index = newIndex;
    div.style.marginTop = "5px";

    div.innerHTML = `
            <input type="text" name="description_${newIndex}" class="main-query-input" placeholder="Next Event (approx. 1 min later)..." autocomplete="off">
            <button type="button" class="remove-query-btn" style="background:#dc3545; color:white; border:none; border-radius:4px; cursor:pointer; padding:0 8px;">X</button>
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
}

function updateRemoveButtons() {
  // Chỉ hiện nút X nếu có > 1 dòng
  const rows = elements.queryInputsContainer.querySelectorAll(".query-row");
  rows.forEach((row) => {
    const btn = row.querySelector(".remove-query-btn");
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
