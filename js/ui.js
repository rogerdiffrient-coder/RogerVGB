export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
export function escapeHTML(value = "") { return String(value).replace(/[&<>"']/g, (c) => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" })[c]); }
export function renderMarkdown(input = "") {
  let text = escapeHTML(input), code = [];
  text = text.replace(/\x60\x60\x60([\w+-]*)\n?([\s\S]*?)\x60\x60\x60/g, (_, lang, body) => { const token = "@@CODE" + code.length + "@@"; code.push('<pre><code class="language-' + escapeHTML(lang) + '">' + body.replace(/\n$/, "") + "</code></pre>"); return token; });
  text = text.replace(/\x60([^\x60\n]+)\x60/g, "<code>$1</code>");
  text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  text = text.replace(/^### (.+)$/gm, "<h3>$1</h3>").replace(/^## (.+)$/gm, "<h2>$1</h2>").replace(/^# (.+)$/gm, "<h1>$1</h1>");
  text = text.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\*(.+?)\*/g, "<em>$1</em>").replace(/~~(.+?)~~/g, "<del>$1</del>");
  text = text.replace(/(^|\n)([-*] .+(?:\n[-*] .+)*)/g, (_, p, list) => p + "<ul>" + list.split("\n").map((line) => "<li>" + line.slice(2) + "</li>").join("") + "</ul>");
  text = text.replace(/(^|\n)(\d+\. .+(?:\n\d+\. .+)*)/g, (_, p, list) => p + "<ol>" + list.split("\n").map((line) => "<li>" + line.replace(/^\d+\. /, "") + "</li>").join("") + "</ol>");
  text = text.split(/\n{2,}/).map((part) => /^@@CODE\d+@@$/.test(part) || /^<(h[1-3]|ul|ol|blockquote|pre)/.test(part) ? part : part.replace(/\n/g, "<br>")).join("");
  return text.replace(/@@CODE(\d+)@@/g, (_, index) => code[Number(index)] || "");
}
export function toast(message) { const node = $("#toast"); node.textContent = message; node.classList.add("show"); clearTimeout(toast.timer); toast.timer = setTimeout(() => node.classList.remove("show"), 2600); }