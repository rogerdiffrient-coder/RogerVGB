import { $, renderMarkdown, toast } from "./ui.js";
import { storage } from "./storage.js";
import { api } from "./api.js";
const makeId = () => (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random());
export class ChatController {
  constructor() {
    this.chats = storage.getChats(); this.currentId = null; this.busy = false; this.abort = null; this.systemPrompt = "";
    this.messages = $("#messages"); this.welcome = $("#welcome"); this.input = $("#message"); this.form = $("#chat-form");
    this.form.addEventListener("submit", (event) => { event.preventDefault(); this.send(); });
    this.input.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); this.send(); } });
    this.input.addEventListener("input", () => { this.input.style.height = "auto"; this.input.style.height = Math.min(this.input.scrollHeight, 180) + "px"; });
    document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => { this.input.value = button.dataset.prompt; this.input.focus(); }));
    $("#new-chat").addEventListener("click", () => this.newChat());
    document.addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); this.newChat(); } });
    this.renderHistory(); if (this.chats.length) this.openChat(this.chats[0].id);
  }
  current() { return this.chats.find((chat) => chat.id === this.currentId); }
  persist() { storage.saveChats(this.chats.slice(0, 80)); }
  newChat() { this.abort?.abort(); this.currentId = null; this.systemPrompt = ""; this.messages.replaceChildren(this.welcome); this.welcome.hidden = false; this.input.value = ""; this.input.style.height = "auto"; this.renderHistory(); this.input.focus(); }
  ensureChat() { let chat = this.current(); if (!chat) { chat = { id: makeId(), title: "New conversation", updatedAt: Date.now(), systemPrompt: this.systemPrompt, messages: [] }; this.chats.unshift(chat); this.currentId = chat.id; } return chat; }
  openChat(id) { this.abort?.abort(); this.currentId = id; const chat = this.current(); if (!chat) return; this.systemPrompt = chat.systemPrompt || ""; this.messages.replaceChildren(); this.welcome.hidden = true; chat.messages.forEach((message) => this.appendMessage(message.role, message.content, false)); this.messages.scrollTop = this.messages.scrollHeight; this.renderHistory(); }
  renderHistory() {
    const root = $("#history"); root.replaceChildren();
    if (!this.chats.length) { const p = document.createElement("p"); p.textContent = "Your conversations will live here."; root.append(p); return; }
    this.chats.slice(0, 35).forEach((chat) => { const button = document.createElement("button"); button.textContent = "◷　" + (chat.title || "New conversation"); button.classList.toggle("active", chat.id === this.currentId); button.title = chat.title || "New conversation"; button.addEventListener("click", () => this.openChat(chat.id)); root.append(button); });
  }
  appendMessage(role, content, scroll = true) {
    this.welcome.hidden = true; const row = document.createElement("article"); row.className = "message " + role;
    const avatar = document.createElement("div"); avatar.className = "message-avatar"; avatar.textContent = role === "user" ? "R" : "✳";
    const box = document.createElement("div"); box.className = "message-content";
    const label = document.createElement("div"); label.className = "message-role"; label.textContent = role === "user" ? "YOU" : "ROGERVGB";
    const body = document.createElement("div"); body.className = "message-body"; body.innerHTML = renderMarkdown(content);
    box.append(label, body); row.append(avatar, box); this.messages.append(row); if (scroll) this.messages.scrollTop = this.messages.scrollHeight; return body;
  }
  async send() {
    if (this.busy) return;
    const content = this.input.value.trim(); if (!content) return;
    const settings = storage.getSettings();
    if (!settings.model) { toast("Connect Ollama and choose a model in settings first."); document.dispatchEvent(new Event("rogervgb:settings-needed")); return; }
    const chat = this.ensureChat(); chat.messages.push({ role: "user", content }); chat.title = chat.messages.filter((m) => m.role === "user").length === 1 ? content.slice(0, 44) + (content.length > 44 ? "…" : "") : chat.title; chat.updatedAt = Date.now();
    this.appendMessage("user", content); this.input.value = ""; this.input.style.height = "auto"; this.persist(); this.renderHistory();
    this.busy = true; $("#send").disabled = true; const typing = this.appendMessage("assistant", "Thinking…"); typing.classList.add("typing"); this.abort = new AbortController(); let answer = "";
    try {
      const stream = api.chat({ model: settings.model, system: this.systemPrompt, messages: chat.messages.map((m) => ({ role: m.role, content: m.content })), signal: this.abort.signal });
      let body = null;
      for await (const chunk of stream) { answer += chunk; if (!body) { typing.closest(".message").remove(); body = this.appendMessage("assistant", "", false); } body.innerHTML = renderMarkdown(answer); this.messages.scrollTop = this.messages.scrollHeight; }
      if (!answer) answer = "(The model returned an empty response.)";
      if (!body) { typing.closest(".message").remove(); this.appendMessage("assistant", answer); }
      chat.messages.push({ role: "assistant", content: answer });
    } catch (error) {
      typing.closest(".message")?.remove();
      if (error.name !== "AbortError") { const message = "Couldn't get a response: " + error.message; this.appendMessage("assistant", message); chat.messages.push({ role: "assistant", content: message }); toast("Model request failed. Check your Ollama connection."); }
    } finally { this.busy = false; $("#send").disabled = false; this.abort = null; chat.updatedAt = Date.now(); this.persist(); this.renderHistory(); }
  }
}