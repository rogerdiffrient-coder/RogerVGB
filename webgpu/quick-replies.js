export function quickReply(text) {
  const t = text.toLowerCase().trim();
  const normalized = t.replace(/,/g, "");
  const sqrt = normalized.match(/(?:square root of|sqrt(?: of)?)\s*(-?\d+(?:\.\d+)?)/i);
  if (sqrt) {
    const n = Number(sqrt[1]);
    if (n < 0) return "The square root of a negative number isn't a real number.";
    const answer = Math.sqrt(n);
    return "The square root of " + n + " is " + (Number.isInteger(answer) ? answer : Number(answer.toPrecision(10))) + ".";
  }

  const expression = normalized.match(/(-?\d+(?:\.\d+)?)\s*(\+|plus|-|minus|×|\*|x|times|÷|\/|divided by)\s*(-?\d+(?:\.\d+)?)/i);
  if (expression) {
    const a = Number(expression[1]), b = Number(expression[3]), op = expression[2].toLowerCase();
    let result;
    if (op === "+" || op === "plus") result = a + b;
    else if (op === "-" || op === "minus") result = a - b;
    else if (["×", "*", "x", "times"].includes(op)) result = a * b;
    else if (["÷", "/", "divided by"].includes(op)) {
      if (b === 0) return "You can't divide by zero.";
      result = a / b;
    }
    if (Number.isFinite(result)) return String(Number(result.toPrecision(12)));
  }

  if (/\b(how'?s it going|how are you|how'?s everything)\b/.test(t) && /\b(yo+|hey+|hi+|hello+)\b/.test(t)) return "Not much, how about you?";
  if (/^(i'?m|im) good\b|^good[,! ]+(thanks|thx)|^doing good\b/.test(t)) return "Nice! What can I help you with?";
  if (/^(thanks|thank you|thx)\b/.test(t)) return "You're welcome!";
  if (/^(hi+|hello+|hey+|he+y+|yo+)[!. ]*$/.test(t)) return "Hey! What's up?";
  if (/\bwhat can (i|you) help (you|me) with\b/.test(t)) return "Tell me what you're working on, ask a question, or throw a weird idea at me.";
  if (/\bwhat'?s your name\b|\bwho are you\b/.test(t)) return "I'm Roger Spark!";
  return null;
}
