import test from "node:test";
import assert from "node:assert/strict";
import { escapeHTML, renderMarkdown } from "../js/ui.js";

test("escapes HTML rather than rendering user-provided tags", () => {
  const output = renderMarkdown("<script>alert(1)</script>");
  assert.match(output, /&lt;script&gt;/);
  assert.doesNotMatch(output, /<script>/);
});

test("renders separate paragraphs", () => {
  assert.equal(renderMarkdown("first paragraph\n\nsecond paragraph"), "<p>first paragraph</p><p>second paragraph</p>");
});

test("renders basic emphasis", () => {
  assert.equal(renderMarkdown("**bold** and *italic*"), "<p><strong>bold</strong> and <em>italic</em></p>");
});

test("renders fenced code without interpreting its contents", () => {
  const output = renderMarkdown("\x60\x60\x60html\n<img src=x>\n\x60\x60\x60");
  assert.match(output, /<pre><code class="language-html">&lt;img src=x&gt;<\/code><\/pre>/);
  assert.doesNotMatch(output, /<img src=x>/);
});

test("only converts HTTP(S) Markdown links", () => {
  assert.match(renderMarkdown("[docs](https://example.com)"), /href="https:\/\/example\.com"/);
  assert.doesNotMatch(renderMarkdown("[bad](javascript:alert(1))"), /href="javascript:/);
});

test("escapes individual HTML special characters", () => {
  assert.equal(escapeHTML("&<>\"'"), "&amp;&lt;&gt;&quot;&#39;");
});
