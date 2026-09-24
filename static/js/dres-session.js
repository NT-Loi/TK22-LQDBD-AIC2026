import {
  APIError,
  getCurrentTaskAPI,
  getDresSessionAPI,
  loginAPI,
  loginDefaultAPI,
  logoutAPI,
  selectEvaluationAPI,
} from "./api.js?v=10";

const state = {
  connected: false,
  user: null,
  evaluations: [],
  selectedEvaluationId: null,
  currentTask: null,
  taskFetchedAt: 0,
  defaultUsername: "team_197",
  dresBaseUrl: "https://eventretrieval.one",
};

export function getDresState() {
  return { ...state };
}

export function getSubmissionMode() {
  return document.getElementById("submission-mode-select")?.value || "kis";
}

export function getSubmissionRoutes() {
  const evaluationId = state.selectedEvaluationId || "<evaluationId>";
  const dresBaseUrl = state.dresBaseUrl.replace(/\/+$/, "");
  return {
    application: `${window.location.origin}/api/dres/submit`,
    dres: `${dresBaseUrl}/api/v2/submit/${encodeURIComponent(evaluationId)}?session=<SESSION_ID_ẨN>`,
  };
}

function emitState() {
  document.dispatchEvent(new CustomEvent("dres:state-change", { detail: getDresState() }));
}

function setStatus(message, kind = "info") {
  const status = document.getElementById("dres-login-status");
  if (!status) return;
  status.textContent = message;
  status.className = `dres-status ${kind}`;
}

function updateConnectionUI() {
  const button = document.getElementById("login-btn");
  const summary = document.getElementById("dres-account-summary");
  const connectedActions = document.getElementById("dres-connected-actions");
  const loginActions = document.getElementById("dres-login-actions");
  const evaluation = state.evaluations.find(
    (item) => String(item.id) === state.selectedEvaluationId,
  );

  if (button) {
    button.classList.toggle("connected", state.connected);
    button.textContent = state.connected
      ? `${state.user?.username || "DRES"}${evaluation ? ` • ${evaluation.name}` : ""}`
      : "Đăng nhập DRES";
    button.title = state.connected ? "Quản lý kết nối DRES" : "Đăng nhập DRES";
  }
  if (summary) {
    summary.textContent = state.connected
      ? `${state.user?.username || "DRES"} • ${evaluation?.name || "Chưa chọn evaluation"}`
      : "Chưa kết nối";
  }
  connectedActions?.classList.toggle("hidden", !state.connected);
  loginActions?.classList.toggle("hidden", state.connected);
  emitState();
}

function applySession(data) {
  state.connected = Boolean(data?.connected);
  state.user = data?.user || null;
  state.evaluations = data?.evaluations || [];
  state.selectedEvaluationId = data?.selectedEvaluationId || null;
  state.defaultUsername = data?.defaultUsername || "team_197";
  state.dresBaseUrl = data?.dresBaseUrl || state.dresBaseUrl;
  state.currentTask = null;
  state.taskFetchedAt = 0;
  const label = document.getElementById("dres-default-username");
  if (label) label.textContent = state.defaultUsername;
  updateConnectionUI();
}

export function openLoginModal(message = "") {
  document.getElementById("dres-login-modal")?.classList.remove("hidden");
  setStatus(message, message ? "error" : "info");
}

function closeLoginModal() {
  document.getElementById("dres-login-modal")?.classList.add("hidden");
  setStatus("");
}

function showEvaluationModal() {
  const modal = document.getElementById("evaluation-modal");
  const list = document.getElementById("evaluation-list");
  if (!modal || !list) return;

  document.getElementById("dres-login-modal")?.classList.add("hidden");

  list.innerHTML = "";
  if (!state.evaluations.length) {
    const empty = document.createElement("p");
    empty.className = "dres-empty";
    empty.textContent = "Không có evaluation ACTIVE cho tài khoản này.";
    list.appendChild(empty);
  }

  state.evaluations.forEach((evaluation) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "evaluation-choice";
    button.textContent = `${evaluation.name} (${evaluation.status})`;
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await selectEvaluationAPI(String(evaluation.id));
        state.selectedEvaluationId = String(evaluation.id);
        modal.classList.add("hidden");
        closeLoginModal();
        updateConnectionUI();
        await refreshCurrentTask(true);
      } catch (error) {
        button.disabled = false;
        alert(`Không thể chọn evaluation: ${error.message}`);
      }
    });
    list.appendChild(button);
  });
  modal.classList.remove("hidden");
}

function inferMode(task) {
  const taskText = [task?.taskType, task?.taskGroup, task?.name]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  if (taskText.includes("trake")) return "trake";
  if (taskText.includes("q&a") || taskText.includes("question") || /(^|\W)qa(\W|$)/.test(taskText)) {
    return "qa";
  }
  if (taskText.includes("kis")) return "kis";
  return null;
}

function applyTaskMode(task) {
  const mode = inferMode(task);
  const select = document.getElementById("submission-mode-select");
  const taskLabel = document.getElementById("dres-task-label");
  if (taskLabel) {
    taskLabel.textContent = task
      ? `${task.name || "Task"} • ${task.taskType || "Không rõ loại"}`
      : "Chưa có task hoạt động";
  }
  if (mode && select) {
    select.value = mode;
    select.dispatchEvent(new Event("change"));
  }
}

export async function refreshCurrentTask(force = false) {
  if (!state.connected || !state.selectedEvaluationId) return null;
  if (!force && Date.now() - state.taskFetchedAt < 5000) return state.currentTask;
  try {
    const data = await getCurrentTaskAPI(state.selectedEvaluationId);
    state.currentTask = data?.task || null;
    state.taskFetchedAt = Date.now();
    applyTaskMode(state.currentTask);
    emitState();
    return state.currentTask;
  } catch (error) {
    if (error instanceof APIError && error.status === 401) {
      handleSessionExpired();
    } else if (!(error instanceof APIError && error.status === 404)) {
      console.warn("Không thể lấy current task:", error);
    }
    state.currentTask = null;
    state.taskFetchedAt = Date.now();
    applyTaskMode(null);
    return null;
  }
}

export function requireDresReady() {
  if (!state.connected) {
    openLoginModal("Vui lòng đăng nhập DRES trước khi nộp bài.");
    return false;
  }
  if (!state.selectedEvaluationId) {
    showEvaluationModal();
    return false;
  }
  return true;
}

export function handleSessionExpired() {
  applySession({ connected: false, defaultUsername: state.defaultUsername });
  openLoginModal("Phiên DRES đã hết hạn. Vui lòng đăng nhập lại.");
}

async function runLogin(action, button) {
  button.disabled = true;
  const originalText = button.textContent;
  button.textContent = "Đang đăng nhập...";
  setStatus("Đang kết nối tới DRES...");
  try {
    const data = await action();
    applySession(data);
    setStatus("Đăng nhập thành công.", "success");
    if (state.evaluations.length) showEvaluationModal();
    else setStatus("Đăng nhập thành công nhưng không có evaluation ACTIVE.", "error");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

export async function initDresSession() {
  const loginButton = document.getElementById("login-btn");
  const closeButton = document.getElementById("close-dres-login-btn");
  const defaultButton = document.getElementById("default-login-btn");
  const customForm = document.getElementById("custom-login-form");
  const customToggle = document.getElementById("show-custom-login-btn");
  const logoutButton = document.getElementById("dres-logout-btn");
  const switchAccountButton = document.getElementById("switch-account-btn");
  const changeEvaluationButton = document.getElementById("change-evaluation-btn");
  const refreshTaskButton = document.getElementById("refresh-task-btn");
  const modeSelect = document.getElementById("submission-mode-select");

  loginButton?.addEventListener("click", () => openLoginModal());
  closeButton?.addEventListener("click", closeLoginModal);
  document.getElementById("cancel-eval-btn")?.addEventListener("click", () => {
    document.getElementById("evaluation-modal")?.classList.add("hidden");
  });
  customToggle?.addEventListener("click", () => {
    document.getElementById("custom-login-fields")?.classList.toggle("hidden");
  });
  defaultButton?.addEventListener("click", () => runLogin(loginDefaultAPI, defaultButton));
  customForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    const username = document.getElementById("dres-username")?.value || "";
    const password = document.getElementById("dres-password")?.value || "";
    const submitButton = customForm.querySelector("button[type='submit']");
    runLogin(() => loginAPI(username, password), submitButton).finally(() => {
      const passwordInput = document.getElementById("dres-password");
      if (passwordInput) passwordInput.value = "";
    });
  });
  logoutButton?.addEventListener("click", async () => {
    try {
      await logoutAPI();
    } finally {
      applySession({ connected: false, defaultUsername: state.defaultUsername });
      closeLoginModal();
    }
  });
  switchAccountButton?.addEventListener("click", async () => {
    switchAccountButton.disabled = true;
    try {
      await logoutAPI();
      applySession({
        connected: false,
        defaultUsername: state.defaultUsername,
        dresBaseUrl: state.dresBaseUrl,
      });
      document.getElementById("custom-login-fields")?.classList.remove("hidden");
      setStatus("Đã kết thúc phiên cũ. Hãy đăng nhập tài khoản mới.");
    } catch (error) {
      setStatus(`Không thể đổi tài khoản: ${error.message}`, "error");
    } finally {
      switchAccountButton.disabled = false;
    }
  });
  changeEvaluationButton?.addEventListener("click", showEvaluationModal);
  refreshTaskButton?.addEventListener("click", () => refreshCurrentTask(true));
  modeSelect?.addEventListener("change", () => {
    document.dispatchEvent(new CustomEvent("dres:mode-change", { detail: modeSelect.value }));
  });
  document.getElementById("open-dres-btn")?.addEventListener("click", () => {
    window.open(state.dresBaseUrl, "_blank", "noopener,noreferrer");
  });
  document.addEventListener("dres:login-required", () => openLoginModal("Vui lòng đăng nhập DRES trước."));
  document.addEventListener("dres:session-expired", handleSessionExpired);

  try {
    applySession(await getDresSessionAPI());
    if (state.selectedEvaluationId) await refreshCurrentTask(true);
  } catch (error) {
    console.warn("Không thể khôi phục phiên DRES:", error);
    applySession({ connected: false });
  }
}
