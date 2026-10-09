import { storage } from "./storage.js";
const trimUrl = (value) => value.trim().replace(/\/+$/, "");
export class OllamaAPI {
  constructor() { this.settings = storage.getSettings(); }
  configure(settings) { this.settings = { ...this.settings, ...settings }; }
  get baseUrl() { return trimUrl(this.settings.baseUrl || "http://localhost:11434"); }
  async request(path, options = {}) {
    let response;
    try { response = await fetch(this.baseUrl + path, { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } }); }
    catch { throw new Error("Could not reach Ollama. Check that it is running and that browser CORS access is allowed for this page."); }
    if (!response.ok) { let message = ""; try { message = (await response.json()).error || ""; } catch {} throw new Error(message || "Ollama returned HTTP " + response.status + "."); }
    return response;
  }
  async listModels() { const response = await this.request("/api/tags"); const data = await response.json(); return (data.models || []).map((item) => ({ name: item.name, size: item.size || 0 })).sort((a,b) => a.name.localeCompare(b.name)); }
  async *chat({ model, messages, system, signal }) {
    const payload = system ? [{ role: "system", content: system }, ...messages] : messages;
    const response = await this.request("/api/chat", { method: "POST", signal, body: JSON.stringify({ model, messages: payload, stream: true }) });
    if (!response.body) throw new Error("This browser does not support streamed responses.");
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = "";
    try {
      while (true) {
        const part = await reader.read(); if (part.done) break;
        buffer += decoder.decode(part.value, { stream: true });
        const lines = buffer.split("\n"); buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          let item; try { item = JSON.parse(line); } catch { continue; }
          if (item.error) throw new Error(item.error);
          if (item.message && item.message.content) yield item.message.content;
          if (item.done) return;
        }
      }
      if (buffer.trim()) { const item = JSON.parse(buffer); if (item.error) throw new Error(item.error); if (item.message && item.message.content) yield item.message.content; }
    } finally { reader.releaseLock(); }
  }
}
export const api = new OllamaAPI();