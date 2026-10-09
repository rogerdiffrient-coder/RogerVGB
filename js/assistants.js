import { $, toast } from "./ui.js";
import { storage } from "./storage.js";
const makeId = () => (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random());
export class AssistantManager {
  constructor(onUse) {
    this.items = storage.getAssistants(); this.onUse = onUse;
    $("#create-assistant").addEventListener("click", () => this.open()); $("#create-first").addEventListener("click", () => this.open()); $("#cancel-assistant").addEventListener("click", () => $("#assistant-dialog").close());
    $("#assistant-form").addEventListener("submit", (event) => { event.preventDefault(); this.save(); }); this.render();
  }
  open(item = null) {
    $("#assistant-modal-title").textContent = item ? "Edit assistant" : "Create assistant";
    $("#assistant-id").value = item?.id || ""; $("#assistant-name").value = item?.name || ""; $("#assistant-description").value = item?.description || ""; $("#assistant-instructions").value = item?.instructions || ""; $("#assistant-model").value = item?.model || ""; $("#assistant-dialog").showModal();
  }
  save() {
    const oldId = $("#assistant-id").value;
    const item = { id: oldId || makeId(), name: $("#assistant-name").value.trim(), description: $("#assistant-description").value.trim(), instructions: $("#assistant-instructions").value.trim(), model: $("#assistant-model").value, updatedAt: Date.now() };
    if (!item.name || !item.instructions) return;
    const index = this.items.findIndex((entry) => entry.id === oldId); if (index >= 0) this.items[index] = item; else this.items.unshift(item);
    storage.saveAssistants(this.items); this.render(); $("#assistant-dialog").close(); toast(oldId ? "Assistant updated." : "Assistant created.");
  }
  render() {
    $("#assistant-count").textContent = this.items.length; $("#assistant-empty").hidden = this.items.length > 0; const root = $("#assistant-grid"); root.replaceChildren();
    this.items.forEach((item) => {
      const card = document.createElement("article"); card.className = "assistant-card";
      const head = document.createElement("header"); const icon = document.createElement("div"); icon.className = "assistant-icon"; icon.textContent = "✳"; const nameBox = document.createElement("div"); const title = document.createElement("h3"); title.textContent = item.name; const model = document.createElement("small"); model.textContent = item.model || "Default model"; nameBox.append(title, model); head.append(icon, nameBox);
      const description = document.createElement("p"); description.textContent = item.description || "A custom RogerVGB assistant.";
      const actions = document.createElement("div"); actions.className = "assistant-actions";
      const use = document.createElement("button"); use.className = "primary"; use.textContent = "Chat ↗"; use.addEventListener("click", () => this.onUse(item));
      const edit = document.createElement("button"); edit.className = "secondary"; edit.textContent = "Edit"; edit.addEventListener("click", () => this.open(item));
      const remove = document.createElement("button"); remove.className = "secondary"; remove.textContent = "Delete"; remove.addEventListener("click", () => { if (!confirm("Delete " + item.name + "?")) return; this.items = this.items.filter((entry) => entry.id !== item.id); storage.saveAssistants(this.items); this.render(); toast("Assistant deleted."); });
      actions.append(use, edit, remove); card.append(head, description, actions); root.append(card);
    });
  }
}