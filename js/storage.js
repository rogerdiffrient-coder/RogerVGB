const KEY = { chats: "rogervgb.chats.v1", assistants: "rogervgb.assistants.v1", settings: "rogervgb.settings.v1" };
function read(key, fallback) { try { const raw = localStorage.getItem(key); return raw ? JSON.parse(raw) : fallback; } catch { return fallback; } }
function write(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch (error) { console.warn("Storage write failed", error); return false; } }
export const storage = {
  getChats: () => read(KEY.chats, []), saveChats: (value) => write(KEY.chats, value),
  getAssistants: () => read(KEY.assistants, []), saveAssistants: (value) => write(KEY.assistants, value),
  getSettings: () => read(KEY.settings, { baseUrl: "http://localhost:11434", model: "" }), saveSettings: (value) => write(KEY.settings, value)
};