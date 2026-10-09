import { $, $$, toast } from "./ui.js";
import { api } from "./api.js";
import { storage } from "./storage.js";
import { ChatController } from "./chat.js";
import { AssistantManager } from "./assistants.js";
import { ModelArena } from "./arena.js";
const chat = new ChatController();
const arena = new ModelArena();
const assistants = new AssistantManager((assistant) => {
  showView("chat"); chat.newChat();
  const settings = storage.getSettings();
  if (assistant.model) { settings.model = assistant.model; storage.saveSettings(settings); api.configure(settings); updateStatus(true); }
  chat.systemPrompt = assistant.instructions + (assistant.description ? "\n\nPurpose: " + assistant.description : "");
  const activeChat = chat.ensureChat(); activeChat.systemPrompt = chat.systemPrompt; storage.saveChats(chat.chats);
  chat.input.value = "Introduce yourself briefly.";
  chat.input.focus();
});
function showView(name) {
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === name + "-view"));
  $$(".nav").forEach((button) => button.classList.toggle("active", button.dataset.view === name));
  $("#page-title").textContent = ({ chat:"Chat", assistants:"My assistants", arena:"Model arena" })[name] || "RogerVGB";
  $("#eyebrow").textContent = ({ chat:"YOUR WORKSPACE", assistants:"CUSTOM AI CREATOR", arena:"MODEL ARENA" })[name] || "ROGERVGB";
  $("#sidebar").classList.remove("open");
}
$$("[data-view]").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
$("#menu").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
$("#settings-open").addEventListener("click", openSettings); $("#settings-top").addEventListener("click", openSettings); document.addEventListener("rogervgb:settings-needed", openSettings);
function openSettings() { const settings = storage.getSettings(); $("#ollama-url").value = settings.baseUrl; $("#settings-status").textContent = "Ollama must be running and reachable from this browser."; $("#settings").showModal(); }
function updateStatus(online = false) {
  const settings = storage.getSettings(); $("#status-dot").classList.toggle("online", online); $("#status-label").textContent = online ? "Ollama connected" : "Ollama not connected"; $("#status-detail").textContent = online ? (settings.model || "Choose a model") : "Open settings to troubleshoot"; $("#model-label").textContent = "○ " + (settings.model || "Choose a model in settings");
}
async function refreshModels(showMessage = false) {
  const settings = storage.getSettings(); api.configure(settings);
  try {
    const models = await api.listModels(); arena.setModels(models, settings.model);
    if (!settings.model && models.length) { settings.model = models[0].name; storage.saveSettings(settings); api.configure(settings); }
    $("#default-model").value = settings.model; updateStatus(true);
    if (showMessage) { $("#settings-status").textContent = "Connected. Found " + models.length + " model(s)."; $("#settings-status").style.color = "var(--green)"; }
    return models;
  } catch (error) {
    updateStatus(false);
    if (showMessage) { $("#settings-status").textContent = error.message; $("#settings-status").style.color = "var(--red)"; }
    return [];
  }
}
$("#test-connection").addEventListener("click", async () => {
  const settings = { ...storage.getSettings(), baseUrl: $("#ollama-url").value.trim().replace(/\\/+$/, "") };
  api.configure(settings);
  $("#settings-status").textContent = "Checking connection…"; $("#settings-status").style.color = "var(--muted)";
  try {
    const models = await api.listModels(); arena.setModels(models, settings.model);
    $("#settings-status").textContent = "Connected. Found " + models.length + " model(s).";
    $("#settings-status").style.color = "var(--green)";
  } catch (error) {
    $("#settings-status").textContent = error.message; $("#settings-status").style.color = "var(--red)";
  }
});
$("#save-settings").addEventListener("click", async () => {
  const settings = { ...storage.getSettings(), baseUrl: $("#ollama-url").value.trim().replace(/\/+$/, ""), model: $("#default-model").value };
  if (!/^https?:\/\//i.test(settings.baseUrl)) { $("#settings-status").textContent = "Enter a valid http:// or https:// address."; $("#settings-status").style.color = "var(--red)"; return; }
  storage.saveSettings(settings); api.configure(settings);
  const models = await refreshModels(true);
  if (models.length) { settings.model = $("#default-model").value || settings.model; storage.saveSettings(settings); api.configure(settings); updateStatus(true); $("#settings").close(); toast("Model connection saved."); }
});
$("#default-model").addEventListener("change", () => { const settings = { ...storage.getSettings(), model: $("#default-model").value }; storage.saveSettings(settings); api.configure(settings); updateStatus(true); });
await refreshModels();
window.addEventListener("focus", () => refreshModels());
