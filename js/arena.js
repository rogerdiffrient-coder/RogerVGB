import { $, renderMarkdown, toast } from "./ui.js";
import { api } from "./api.js";
export class ModelArena {
  constructor() { $("#arena-run").addEventListener("click", () => this.run()); }
  setModels(models, preferred = "") {
    for (const id of ["model-a", "model-b", "default-model", "assistant-model"]) {
      const select = $("#" + id); const previous = select.value; select.replaceChildren();
      if (id === "assistant-model") select.add(new Option("Use default model", ""));
      if (!models.length) select.add(new Option("No models found", ""));
      models.forEach((model) => select.add(new Option(model.name, model.name)));
      if (previous && models.some((model) => model.name === previous)) select.value = previous;
      else if (preferred && models.some((model) => model.name === preferred)) select.value = preferred;
    }
    if (models.length > 1 && $("#model-a").value === $("#model-b").value) $("#model-b").value = models.find((model) => model.name !== $("#model-a").value)?.name || "";
  }
  async run() {
    const prompt = $("#arena-prompt").value.trim(), a = $("#model-a").value, b = $("#model-b").value;
    if (!prompt) return toast("Give the models a prompt first.");
    if (!a || !b) return toast("Connect Ollama and load at least one model.");
    if (a === b) return toast("Choose two different models.");
    const button = $("#arena-run"); button.disabled = true; button.textContent = "⏳ Comparing…";
    const root = $("#arena-results"); root.replaceChildren();
    const cards = [a,b].map((model,index) => { const card = document.createElement("article"); card.className = "result-card"; const title = document.createElement("h3"); title.textContent = "MODEL " + (index ? "B · " : "A · ") + model; const body = document.createElement("div"); body.className = "result-body"; body.textContent = "Running…"; const time = document.createElement("div"); time.className = "result-time"; card.append(title,body,time); root.append(card); return { body,time,model }; });
    await Promise.all(cards.map(async (card) => {
      const start = performance.now(); let answer = ""; card.body.textContent = "";
      try { for await (const chunk of api.chat({ model: card.model, messages: [{ role: "user", content: prompt }] })) { answer += chunk; card.body.innerHTML = renderMarkdown(answer); } if (!answer) card.body.textContent = "(Empty response)"; card.time.textContent = "Finished in " + ((performance.now() - start) / 1000).toFixed(1) + "s"; }
      catch (error) { card.body.textContent = "Request failed: " + error.message; card.time.textContent = "Could not complete"; }
    }));
    button.disabled = false; button.textContent = "⚔ Run comparison"; toast("Both model runs finished.");
  }
}