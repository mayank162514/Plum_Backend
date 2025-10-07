import Tesseract from 'tesseract.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || process.env.gemini_api_key;
let geminiModel = null;
if (GEMINI_API_KEY && GoogleGenerativeAI) {
  try {
    const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    // The actual API shape may differ; keep defensive
    if (typeof genAI.getGenerativeModel === 'function') {
      geminiModel = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
    } else {
      geminiModel = null;
    }
  } catch (e) {
    geminiModel = null;
  }
}

function detectCurrencyHint(text) {
  if (!text) return null;
  const t = text.toUpperCase();
  if (/\bINR\b|\bRS\b|RS\.|₹/.test(t)) return 'INR';
  if (/\bUSD\b|\$/.test(t)) return 'USD';
  if (/\bEUR\b|€/.test(t)) return 'EUR';
  return null;
}

// Extract raw numeric-like tokens from text. Keep percent tokens.
function extractRawNumericTokens(text) {
  if (!text) return [];
  // normalize common separators
  const normalized = text.replace(/[\r\n\|]/g, ' ');

  // tokenization: find sequences containing digits and optionally % or . and commas
  // Support ₹, $, € prefixes and common OCR-confused letters inside numbers
  const regex = /(₹|\$|€)\s*[0-9,]+(?:\.[0-9]{1,2})?|\b(?:Rs\.?|INR)\s*[0-9,]+(?:\.[0-9]{1,2})?|[0-9OIlSBTAGQ\.,%]{1,}\d|\d+[\d\.,%]*/g; // prioritize currency-prefixed amounts
  const matches = normalized.match(regex) || [];

  // clean matches: remove trailing punctuation
  const cleaned = matches
    .map(s => s.replace(/^[:\s"']+|[:\s"']+$/g, ''))
    .map(s => s.replace(/^(?!(₹|\$|€))[^0-9OIlSBTAGQ₹$€]+|[^0-9%\.,₹$€]+$/g, ''))
    .filter(Boolean);

  return cleaned;
}

// Extract tokens only from lines that look financial (contain keywords or currency)
function extractFinancialTokens(text) {
  if (!text) return [];
  const lines = text.split(/\r?\n/);
  const keywords = /(total|amount|paid|due|balance|subtotal|grand\s*total|discount|invoice|tax|gst|vat)/i;
  const currencyRe = /₹|\$|€|\bRs\.?\b|\bINR\b/i;
  const numRe = /(₹|\$|€)\s*[0-9,]+(?:\.[0-9]{1,2})?|\b(?:Rs\.?|INR)\s*[0-9,]+(?:\.[0-9]{1,2})?|\b[0-9][\d,]*(?:\.[0-9]{1,2})?%?/g;
  const out = [];
  for (const line of lines) {
    if (!line) continue;
    if (keywords.test(line) || currencyRe.test(line)) {
      const matches = line.match(numRe) || [];
      for (const m of matches) out.push(m.trim());
    }
  }
  return out;
}

// Safely find the index of a numeric token without matching substrings inside larger numbers
function indexOfNumberToken(haystack, needle) {
  if (typeof haystack !== 'string' || typeof needle !== 'string') return -1;
  const nlen = needle.length;
  let from = 0;
  while (true) {
    const idx = haystack.indexOf(needle, from);
    if (idx === -1) return -1;
    const prev = idx > 0 ? haystack[idx - 1] : '';
    const next = idx + nlen < haystack.length ? haystack[idx + nlen] : '';
    const prevIsDigitish = /[0-9]/.test(prev);
    const nextIsDigitish = /[0-9]/.test(next);
    if (!prevIsDigitish && !nextIsDigitish) return idx;
    from = idx + 1;
  }
}

// OCR digit confusion map (common mistakes)
const DIGIT_MAP = {
  O: '0', o: '0', D: '0',
  I: '1', l: '1', i: '1', '|': '1',
  Z: '2',
  A: '4', a: '4',
  T: '7', t: '7',
  G: '6', g: '9',
  Q: '9', q: '9',
  S: '5', s: '5',
  B: '8'
};

function correctOcrDigits(token) {
  // if token contains letters that look like digits, map them
  let changed = false;
  const arr = token.split('');
  for (let i = 0; i < arr.length; i++) {
    const ch = arr[i];
    if (DIGIT_MAP[ch]) { arr[i] = DIGIT_MAP[ch]; changed = true; }
  }
  const corrected = arr.join('');
  return { corrected, changed };
}

function normalizeTokenToNumber(token) {
  if (!token || typeof token !== 'string') return { ok: false };

  // handle percentages
  if (token.includes('%')) {
    // attempt to strip and parse
    const num = token.replace(/[^0-9\.]/g, '');
    const val = Number(num);
    if (Number.isFinite(val)) return { ok: true, value: val, type: 'percent' };
    return { ok: false };
  }

  // strip currency symbols and commas (keep digits)
  let s = token.replace(/₹\s*/ig, '').replace(/[,$€$]/g, '').replace(/Rs\.?|INR\.?/ig, '');
  // correct OCR chars too
  const { corrected } = correctOcrDigits(s);
  s = corrected;
  // remove non-digit and non-dot
  s = s.replace(/[^0-9\.]/g, '');
  if (s.length === 0) return { ok: false };
  // parse float then convert to integer if looks integer
  const num = Number(s);
  if (!Number.isFinite(num)) return { ok: false };
  return { ok: true, value: num, type: 'amount' };
}

// Confidence heuristics
function computeConfidence(scores) {
  // scores: array of 0..1 values, return average
  if (!scores || scores.length === 0) return 0.0;
  const sum = scores.reduce((a, b) => a + b, 0);
  return Math.max(0, Math.min(1, sum / scores.length));
}

function snippetAround(text, token, window = 30) {
  if (!text || !token) return text || '';
  const idx = text.indexOf(token);
  if (idx === -1) return text;
  const start = Math.max(0, idx - window);
  const end = Math.min(text.length, idx + token.length + window);
  return text.substring(start, end);
}

function classifyByContext(token, raw_text) {
  if (!token || typeof raw_text !== 'string') return 'unknown';

  const text = raw_text.toLowerCase();
  const tokenStr = String(token).toLowerCase();

  const idx = indexOfNumberToken(text, tokenStr);
  const localBefore = idx !== -1 ? text.slice(Math.max(0, idx - 30), idx) : text;
  const localSnippet = idx !== -1 ? text.slice(Math.max(0, idx - 50), Math.min(text.length, idx + tokenStr.length + 50)) : text;

  // Prefer immediate-left context near the number to avoid cross-label contamination
  if (/\b(due|balance|outstanding|remaining)\b/.test(localBefore)) return 'due';
  if (/\b(paid|payment|received)\b/.test(localBefore)) return 'paid';
  if (/\b(discount|rebate|offer)\b/.test(localBefore)) return 'discount';
  if (/\b(tax|gst|vat)\b/.test(localBefore)) return 'tax';
  if (/\b(total|amount|bill|subtotal|grand total)\b/.test(localBefore)) return 'total_bill';

  // If not found immediately before, use a small local window preference
  if (/\b(due|balance|outstanding|remaining)\b/.test(localSnippet)) return 'due';
  if (/\b(paid|payment|received)\b/.test(localSnippet)) return 'paid';
  if (/\b(discount|rebate|offer)\b/.test(localSnippet)) return 'discount';
  if (/\b(tax|gst|vat)\b/.test(localSnippet)) return 'tax';
  if (/\b(total|amount|bill|subtotal|grand total)\b/.test(localSnippet)) return 'total_bill';

  return 'unknown';
}

// Fallback: extract labeled amounts like "Total: $1200" directly from text
function extractExplicitLabeledAmounts(raw_text) {
  if (typeof raw_text !== 'string' || raw_text.length === 0) return [];
  const text = raw_text;
  const patterns = [
    { label: 'total_bill', re: /(total|grand\s*total|amount\s*due)\s*[:\-]?\s*(₹|\$|€)?\s*([0-9][\d,]*(?:\.[0-9]{1,2})?)/gi },
    { label: 'paid', re: /(paid|payment\s*received)\s*[:\-]?\s*(₹|\$|€)?\s*([0-9][\d,]*(?:\.[0-9]{1,2})?)/gi },
    { label: 'due', re: /(due|balance|outstanding|remaining)\s*[:\-]?\s*(₹|\$|€)?\s*([0-9][\d,]*(?:\.[0-9]{1,2})?)/gi }
  ];
  const found = [];
  for (const { label, re } of patterns) {
    let m;
    while ((m = re.exec(text)) !== null) {
      const numStr = m[3] || '';
      const clean = numStr.replace(/[,]/g, '');
      const val = Number(clean);
      if (Number.isFinite(val)) {
        found.push({ type: label, value: val });
        break; // take first match per label
      }
    }
  }
  return found;
}

// ---------------------- Step Implementations ----------------------

function withTimeout(promise, ms, onTimeoutMessage = 'timeout') {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      const t = setTimeout(() => {
        clearTimeout(t);
        reject(new Error(onTimeoutMessage));
      }, ms);
    })
  ]);
}

async function ocrImageToText(filePath) {
  try {
    // Prefer local tessdata if present to avoid network fetch failures
    const localLangPath = path.resolve(__dirname, 'tessdata');
    const hasLocalEng = fs.existsSync(path.join(localLangPath, 'eng.traineddata')) || fs.existsSync(path.join(localLangPath, 'eng.traineddata.gz'));
    const remoteLangPath = process.env.TESSDATA_URL || 'https://tessdata.projectnaptha.com/4.0.0';

    // Prefer local data, but always fall back to remote tessdata if not available
    let langPathToUse = hasLocalEng ? localLangPath : remoteLangPath;

    // Tesseract.recognize can accept (image, lang, options) or (image, options)
    // Using the 3-arg form here: second arg 'eng'
    const recognition = Tesseract.recognize(
      filePath,
      'eng',
      {
        langPath: langPathToUse,
        tessedit_char_whitelist: '0123456789.%$,₹€ INRRS',
        preserve_interword_spaces: 1,
        psm: 6, // Assume a single uniform block of text
        logger: () => {}
      }
    );
    const { data } = await withTimeout(recognition, 35000, 'ocr_timeout');

    const text = data && data.text ? data.text : null;
    if (!text) return null;

    // 1. Remove trailing spaces/newlines
    let cleanedText = text.replace(/[\s\n]+$/g, '');

    // 2. Optional: remove stray symbols at start of lines but keep currency signs
    cleanedText = cleanedText.replace(/^[^a-zA-Z0-9₹$€]+/gm, '');
    // console.log("OCR cleanedText:",cleanedText);
    return cleanedText;

  } catch (err) {
    console.error('OCR error', err && err.message ? err.message : err);
    return null;
  }
}

async function step1_extract({ textInput, imagePath }) {
  // If imagePath provided, run OCR then append provided text
//   console.log("step1_extract", { textInput, imagePath });
  let ocrText = '';
  if (imagePath) {
    const t = await ocrImageToText(imagePath);
    if (t) ocrText += t + '\n';
  }
  if (textInput) ocrText += textInput;
//   console.log("ocrText",ocrText);
  // If nothing found
  if (!ocrText || ocrText.trim().length === 0) {
    return { guardrail: true, output: { status: 'no_amounts_found', reason: 'document too noisy' } };
  }

  let currency_hint = detectCurrencyHint(ocrText) || 'UNKNOWN';
  const raw_tokens_all = extractRawNumericTokens(ocrText);
  const nearFinancial = extractFinancialTokens(ocrText);
  const raw_tokens = (nearFinancial.length > 0 ? nearFinancial : raw_tokens_all);

  if (currency_hint === 'UNKNOWN' && /₹|\bRs\.?\b|\bINR\b/i.test(ocrText)) {
    currency_hint = 'INR';
  }

  // compute confidence: based on amount of tokens and presence of currency
  const tokenScore = Math.min(1, raw_tokens.length / 5);
  const currencyScore = currency_hint === 'UNKNOWN' ? 0.5 : 1.0;
  const confidence = computeConfidence([tokenScore, currencyScore]);

  return { guardrail: false, output: { raw_tokens, currency_hint, confidence, raw_text: ocrText } };
}

function step2_normalize({ raw_tokens }) {
  if (!raw_tokens || raw_tokens.length === 0) {
    return { guardrail: true, output: { status: 'no_amounts_found', reason: 'no numeric tokens' } };
  }

  const normalized_amounts = [];
  const confidences = [];
  for (const tok of raw_tokens) {
    // Try direct parse
    const parsed = normalizeTokenToNumber(String(tok));
    if (parsed.ok && parsed.type === 'amount') {
      normalized_amounts.push(parsed.value);
      confidences.push(0.95);
      continue;
    }
    if (parsed.ok && parsed.type === 'percent') {
      // Ignore percent in amounts array for now
      confidences.push(0.6);
      continue;
    }
    // Try correcting OCR chars then parse
    const { corrected } = correctOcrDigits(String(tok));
    const retry = normalizeTokenToNumber(corrected);
    if (retry.ok && retry.type === 'amount') {
      normalized_amounts.push(retry.value);
      // slightly lower confidence because correction applied
      confidences.push(0.75);
      continue;
    }
    confidences.push(0.2);
  }

  if (normalized_amounts.length === 0) {
    return { guardrail: true, output: { status: 'no_amounts_found', reason: 'normalized nothing' } };
  }

  const normalization_confidence = computeConfidence(confidences);
  return { guardrail: false, output: { normalized_amounts, normalization_confidence } };
}

async function step3_classify({ normalized_amounts, raw_text, raw_tokens }) {
  // create local copies so we don't try to reassign parameters
  const amountsList = Array.isArray(normalized_amounts) ? normalized_amounts : [];
  const tokensList = Array.isArray(raw_tokens) ? raw_tokens : [];
  const rawText = typeof raw_text === 'string' ? raw_text : '';

  if (amountsList.length === 0) {
    return { guardrail: true, output: { status: 'no_amounts_found', reason: 'no normalized amounts' } };
  }

  const amounts = [];
  const confidences = [];

  // Try Gemini classification first if configured
  let geminiMap = null;
  if (geminiModel) {
    try {
      const prompt = [
        'You are labeling amounts in medical bills/receipts.',
        'Allowed labels: total_bill, paid, due, discount, tax, unknown.',
        'Given the full text and a list of numeric values, assign one label to each value.',
        'Return strict JSON only in this schema:',
        '{"labels": [{"value": number, "type": "total_bill|paid|due|discount|tax|unknown"}]}',
        '',
        'Full text:',
        rawText,
        '',
        'Values:',
        JSON.stringify(amountsList)
      ].join('\n');

      // Gemeni call shape is guarded — real SDK might differ
      const result = await geminiModel.generateContent({ contents: [{ role: 'user', parts: [{ text: prompt }] }] });
      let textOut = '';
      if (result) {
        if (result.response) {
          // some SDKs expose .text() function
          if (typeof result.response.text === 'function') {
            try { textOut = result.response.text(); } catch (e) { textOut = ''; }
          } else if (typeof result.response === 'string') {
            textOut = result.response;
          } else if (typeof result === 'string') {
            textOut = result;
          }
        } else if (typeof result === 'string') {
          textOut = result;
        }
      }

      if (textOut) {
        // Try to extract JSON
        const jsonStart = textOut.indexOf('{');
        const jsonEnd = textOut.lastIndexOf('}');
        if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
          const jsonStr = textOut.substring(jsonStart, jsonEnd + 1);
          try {
            const parsed = JSON.parse(jsonStr);
            if (parsed && Array.isArray(parsed.labels)) {
              geminiMap = new Map();
              for (const item of parsed.labels) {
                if (!item) continue;
                const v = Number(item.value);
                const t = String(item.type || 'unknown');
                if (Number.isFinite(v)) geminiMap.set(v, t);
              }
            }
          } catch (e) {
            // ignore parse errors; keep geminiMap null
            geminiMap = null;
          }
        }
      }
    } catch (e) {
      geminiMap = null;
    }
  }

  for (const value of amountsList) {
    let chosenToken = null;
    for (const tok of tokensList) {
      if (!tok) continue;
      const parsed = normalizeTokenToNumber(String(tok));
      if (parsed.ok && Number(parsed.value) === Number(value)) {
        chosenToken = tok;
        break;
      }
    }
    if (!chosenToken) chosenToken = String(value);

    let label = 'unknown';
    if (geminiMap && geminiMap.has(Number(value))) {
      label = geminiMap.get(Number(value));
    } else {
      label = classifyByContext(chosenToken, rawText);
    }

    let conf = 0.7;
    if (label === 'total_bill') conf = 0.9;
    else if (label === 'paid') conf = 0.88;
    else if (label === 'due') conf = 0.86;
    else if (label === 'discount') conf = 0.75;
    else if (label === 'tax') conf = 0.8;

    // Keep source internally (used by Step 4)
    const snippet = snippetAround(rawText, String(chosenToken)).replace(/'/g, "\\'");
    amounts.push({ type: label, value, source: `text: '${snippet}'` });
    confidences.push(conf);
  }

  const confidence = parseFloat(computeConfidence(confidences).toFixed(2));
  // If none of the required labels detected, attempt explicit labeled extraction fallback
  const hasRequired = amounts.some(a => a && (a.type === 'total_bill' || a.type === 'paid' || a.type === 'due'));
  if (!hasRequired) {
    const explicit = extractExplicitLabeledAmounts(rawText);
    if (explicit.length > 0) {
      // Merge fallback labels; avoid duplicates on same value
      const seenVals = new Set(amounts.map(a => a && a.value));
      for (const e of explicit) {
        if (!seenVals.has(e.value)) {
          amounts.push({ type: e.type, value: e.value, source: "text: 'explicit labeled'" });
        }
      }
    }
  }
  return { guardrail: false, output: { amounts, confidence } };
}

function canonicalLabel(label) {
  if (label === 'total_bill') return 'Total';
  if (label === 'paid') return 'Paid';
  if (label === 'due') return 'Due';
  return label;
}

function formatCurrencyPrefix(currency_hint) {
  if (!currency_hint) return '';
  if (currency_hint === 'INR') return 'INR ';
  if (currency_hint === 'USD') return '$';
  if (currency_hint === 'EUR') return '€';
  return '';
}

function buildProvenance(label, value, raw_text, currency_hint) {
  const canon = canonicalLabel(label);
  const cur = label === 'total_bill' ? formatCurrencyPrefix(currency_hint) : '';
  // Try to find exact pattern "Label: [CUR]value"
  const escapedVal = String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const safeCurForRegex = cur ? cur.replace(/[$]/g, '\\$&') : '';
  const labelRegex = new RegExp(`\\b${canon}\\s*:?\\s*${safeCurForRegex}${escapedVal}\\b`, 'i');
  const m = raw_text.match(labelRegex);
  if (m) return `text: '${canon}: ${cur}${value}'`;

  // Fallback: just return compact version
  return `text: '${canon}: ${cur}${value}'`;
}

function step4_finalize({ amounts, currency_hint, raw_text }) {
  // prepare final JSON
  if (!Array.isArray(amounts) || amounts.length === 0) {
    return { guardrail: true, output: { status: 'no_amounts_found', reason: 'no classified amounts' } };
  }

  // Filter to only required labels and pick first occurrence per type
  const wanted = new Set(['total_bill', 'paid', 'due']);
  const seen = new Set();
  const finalAmounts = [];
  for (const a of amounts) {
    if (!a || !wanted.has(a.type)) continue;
    if (seen.has(a.type)) continue;
    const source = buildProvenance(a.type, a.value, raw_text || '', currency_hint);
    finalAmounts.push({ type: a.type, value: a.value, source });
    seen.add(a.type);
  }

  if (finalAmounts.length === 0) {
    // Final fallback: attempt explicit labeled extraction directly from raw_text
    const explicit = extractExplicitLabeledAmounts(raw_text || '');
    if (explicit.length > 0) {
      const seen = new Set();
      for (const e of explicit) {
        if (seen.has(e.type)) continue;
        const source = buildProvenance(e.type, e.value, raw_text || '', currency_hint);
        finalAmounts.push({ type: e.type, value: e.value, source });
        seen.add(e.type);
      }
    }
    if (finalAmounts.length === 0) {
      return { guardrail: true, output: { status: 'no_amounts_found', reason: 'no required labels detected' } };
    }
  }

  return { guardrail: false, output: { currency: currency_hint || 'UNKNOWN', amounts: finalAmounts, status: 'ok' } };
}

export const extract=async (req, res) => {
  try {
    const textInput = req.body.text || null;
    let imagePath = null;
    if (req.file) imagePath = path.resolve(req.file.path);
    // console.log("imagePath",imagePath);
    const result = await step1_extract({ textInput, imagePath });

    // cleanup uploaded file
    if (imagePath) {
      try { fs.unlinkSync(imagePath); } catch (e) { /* ignore */ }
    }

    if (result.guardrail) return res.status(400).json(result.output);
    // Return only the fields required by spec (omit raw_text)
    const { raw_tokens, currency_hint, confidence } = result.output;
    return res.json({ raw_tokens, currency_hint, confidence });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'server_error' });
  }
}
export const normalizeValues= (req, res) => {
  try {
    const raw_tokens = req.body.raw_tokens;
    const result = step2_normalize({ raw_tokens });
    if (result.guardrail) return res.status(400).json(result.output);
    return res.json(result.output);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'server_error' });
  }
}
export const classify =async (req, res) => {
  try {
    const normalized_amounts = req.body.normalized_amounts || [];
    const raw_text = req.body.raw_text || '';
    const raw_tokens = req.body.raw_tokens || [];
    const result = await step3_classify({ normalized_amounts, raw_text, raw_tokens });
    if (result.guardrail) return res.status(400).json(result.output);
    // Strip source for Step 3 response to match expected schema
    const sanitized = {
      amounts: (result.output.amounts || []).map(a => ({ type: a.type, value: a.value })),
      confidence: result.output.confidence
    };
    return res.json(sanitized);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'server_error' });
  }
}
export const finalize = (req, res) => {
  try {
    const amounts = req.body.amounts || [];
    const currency_hint = req.body.currency || 'UNKNOWN';
    const raw_text = req.body.raw_text || '';
    const result = step4_finalize({ amounts, currency_hint, raw_text });
    if (result.guardrail) return res.status(400).json(result.output);
    return res.json(result.output);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'server_error' });
  }
}
export const fullPipeline = async (req, res) => {
  try {
    const textInput = req.body.text || null;
    let imagePath = null;
    if (req.file) imagePath = path.resolve(req.file.path);

    // Step1
    const s1 = await step1_extract({ textInput, imagePath });
    if (imagePath) { try { fs.unlinkSync(imagePath); } catch (e) { /* ignore */ } }
    if (s1.guardrail) return res.status(400).json(s1.output);

    const raw_tokens = s1.output.raw_tokens;
    const currency_hint = s1.output.currency_hint || 'UNKNOWN';
    const raw_text = s1.output.raw_text || '';

    // Step2
    const s2 = step2_normalize({ raw_tokens });
    if (s2.guardrail) return res.status(400).json(s2.output);
    const normalized_amounts = s2.output.normalized_amounts;

    // Step3
    const s3 = await step3_classify({ normalized_amounts, raw_text, raw_tokens });
    if (s3.guardrail) return res.status(400).json(s3.output);
    const amounts = s3.output.amounts;

    // Step4
    const s4 = step4_finalize({ amounts, currency_hint, raw_text });
    if (s4.guardrail) return res.status(400).json(s4.output);

    // Return only final expected fields
    return res.json(s4.output);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'server_error' });
  }
}