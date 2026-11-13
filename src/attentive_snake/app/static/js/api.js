// JSON requests to the backend. Errors become ApiError with a readable message.

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

function detailMessage(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI/pydantic validation errors: [{loc: [...], msg}]
    return detail.map((d) => `${(d.loc || []).filter((p) => p !== "body").join(".")}: ${d.msg}`).join("\n");
  }
  return JSON.stringify(detail);
}

export async function request(method, url, body) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new ApiError(`Cannot reach the server (${error.message}). Is it still running?`, 0);
  }
  if (response.status === 204) return null;
  const type = response.headers.get("Content-Type") || "";
  const payload = type.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = payload && payload.detail !== undefined ? detailMessage(payload.detail) : String(payload);
    throw new ApiError(message || `HTTP ${response.status}`, response.status);
  }
  return payload;
}

export const get = (url) => request("GET", url);
export const post = (url, body = {}) => request("POST", url, body);
export const put = (url, body) => request("PUT", url, body);
export const del = (url) => request("DELETE", url);
