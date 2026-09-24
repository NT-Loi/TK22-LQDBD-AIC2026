import { elements } from "./elements.js";

export class APIError extends Error {
  constructor(message, status = 0, data = null) {
    super(message);
    this.name = "APIError";
    this.status = status;
    this.data = data;
  }
}

async function requestJSON(url, options = {}) {
  let response;
  try {
    response = await fetch(url, {
      credentials: "same-origin",
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
  } catch (error) {
    throw new APIError("Không thể kết nối tới máy chủ ứng dụng", 0, error);
  }

  const rawText = await response.text();
  let data = null;
  if (rawText) {
    try {
      data = JSON.parse(rawText);
    } catch {
      data = rawText;
    }
  }

  if (!response.ok) {
    const message =
      (data && typeof data === "object" && (data.detail || data.error || data.message)) ||
      (typeof data === "string" && data) ||
      `HTTP ${response.status}`;
    throw new APIError(String(message), response.status, data);
  }
  return data;
}

export async function searchAPI(queryData) {
  elements.resultsContainer.innerHTML = "<p>Searching...</p>";
  try {
    return await requestJSON("/search", {
      method: "POST",
      body: JSON.stringify(queryData),
    });
  } catch (error) {
    console.error("Search failed:", error);
    elements.resultsContainer.innerHTML = `<p style="color: red;">An error occurred: ${error.message}</p>`;
    return [];
  }
}

export function getDresSessionAPI() {
  return requestJSON("/api/dres/session");
}

export function loginDefaultAPI() {
  return requestJSON("/api/dres/login/default", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function loginAPI(username = null, password = null) {
  if (username === null && password === null) {
    return loginDefaultAPI();
  }
  return requestJSON("/api/dres/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function logoutAPI() {
  return requestJSON("/api/dres/logout", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function getEvaluationsAPI() {
  return requestJSON("/api/dres/evaluations");
}

export function selectEvaluationAPI(evaluationId) {
  return requestJSON("/api/dres/evaluation", {
    method: "POST",
    body: JSON.stringify({ evaluationId }),
  });
}

export function getCurrentTaskAPI(evaluationId) {
  return requestJSON(
    `/api/dres/current-task/${encodeURIComponent(evaluationId)}`,
  );
}

export function submitResultAPI(submission) {
  return requestJSON("/api/dres/submit", {
    method: "POST",
    body: JSON.stringify(submission),
  });
}
