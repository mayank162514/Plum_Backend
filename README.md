# Plum Backend — AI-Powered Amount Detection

Extracts monetary amounts (Total, Paid, Due) from medical bills/receipts (typed or scanned). Handles OCR noise, digit corrections, and context classification, returning structured JSON with provenance and confidence.

## Postman endpoints video
"https://drive.google.com/file/d/1tI-0bx1nNUy_JR8TtHgciwoqLY5Wt4o9/view?usp=sharing"

## Setup

- Prerequisites: Node.js 18+, npm
- Install dependencies:
```bash
npm install
```
- Run the server:
```bash
npm start
# or: node server.js
```
- Default port: 3000. Health check:
```bash
http://127.0.0.1:3000/health
```

### Environment variables
- PORT: server port (default 3000)
- GEMINI_API_KEY: optional, enables Gemini-assisted classification
- TESSDATA_URL: remote tessdata (default https://tessdata.projectnaptha.com/4.0.0)
- Optional local OCR data: place `tessdata/eng.traineddata` under project root for offline OCR

## Architecture

Single-file implementation with a 4-step pipeline:
- Step 1: Extract (OCR image and/or combine provided text) → raw tokens, currency hint
- Step 2: Normalize tokens to numbers (OCR digit correction)
- Step 3: Classify amounts by nearby context (total/paid/due/discount/tax), with explicit-label fallback and optional Gemini
- Step 4: Finalize response (currency + first detected total/paid/due with provenance)

Endpoints provided for each step and a single end-to-end `/process` route.

```
Plum_Backend/
  server.js        # Express app, OCR, tokenization, classification, routes
  uploads/         # Temp upload dir for images (multer)
```

## API
Base URL: `http://127.0.0.1:3000`

### GET /health
Health check.
```json
{ "status": "ok" }
```

### POST /step1
Extract raw numeric tokens and currency hint from text or image.
- Accepts: multipart/form-data (file) or JSON (text)
- Returns: `{ raw_tokens, currency_hint, confidence }`


### POST /step2
Normalize tokens to numeric values.
- Body: `{ raw_tokens: string[] }`
- Returns: `{ normalized_amounts, normalization_confidence }`


### POST /step3
Classify normalized amounts using `raw_text` and `raw_tokens`.
- Body: `{ normalized_amounts: number[], raw_text: string, raw_tokens: string[] }`
- Returns: `{ amounts: [{type, value}], confidence }`


### POST /step4
Finalize output structure.
- Body: `{ amounts: [{type,value}], currency: string, raw_text: string }`
- Returns: `{ currency, amounts: [{type,value,source}], status: "ok" }`



### POST /process
End-to-end pipeline for image and/or text.
- Accepts: multipart/form-data (file) and/or JSON (text)
- Returns: `{ currency, amounts: [{type,value,source}], status: "ok" }` or guardrail status



## Postman Samples
Use a new request tab to avoid hidden base URLs. Replace 3000 if you changed PORT.

### Health (GET)
- Method: GET
- URL: `http://127.0.0.1:3000/health`
- Send
- Response:
```json
{ "status": "ok" }
```

### Step 1 (Image)
- Method: POST
- URL: `http://127.0.0.1:3000/step1`
- Body: form-data → Key `file` (Type File) → choose image
- Send

### Step 1 (Text)
- Method: POST
- URL: `http://127.0.0.1:3000/step1`
- Headers: `Content-Type: application/json`
- Body (raw JSON):
```json
{
  "text": "Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%"
}
```

### Step 2
- Method: POST
- URL: `http://127.0.0.1:3000/step2`
- Headers: `Content-Type: application/json`
- Body:
```json
{
  "raw_tokens": ["1200", "1000", "200", "10%"]
}
```

### Step 3
- Method: POST
- URL: `http://127.0.0.1:3000/step3`
- Headers: `Content-Type: application/json`
- Body:
```json
{
  "normalized_amounts": [1200, 1000, 200],
  "raw_text": "Total: INR 1200 | Paid: 1000 | Due: 200",
  "raw_tokens": ["1200", "1000", "200"]
}
```

### Step 4
- Method: POST
- URL: `http://127.0.0.1:3000/step4`
- Headers: `Content-Type: application/json`
- Body:
```json
{
  "amounts": [
    {"type": "total_bill", "value": 1200},
    {"type": "paid", "value": 1000},
    {"type": "due", "value": 200}
  ],
  "currency": "INR"
}
```

### Full Pipeline `/process` (Image)
- Method: POST
- URL: `http://127.0.0.1:3000/process`
- Body: form-data → Key `file` (File)
- Send

### Full Pipeline `/process` (Text)
- Method: POST
- URL: `http://127.0.0.1:3000/process`
- Headers: `Content-Type: application/json`

- Response:
```json
{
    "currency": "INR",
    "amounts": [
        {
            "type": "total_bill",
            "value": 1200,
            "source": "text: 'Total: INR 1200'"
        },
        {
            "type": "paid",
            "value": 1000,
            "source": "text: 'Paid: 1000'"
        },
        {
            "type": "due",
            "value": 200,
            "source": "text: 'Due: 200'"
        }
    ],
    "status": "ok"
}
```

## Expected Responses
Guardrail (examples)
```json
{ "status": "no_amounts_found", "reason": "document too noisy" }
```
```json
{ "status": "no_amounts_found", "reason": "no required labels detected" }
```
Error (example)
```json
{ "error": "server_error" }
```

## Troubleshooting
- Ensure the server log shows the correct port and your Postman URL matches it
- For image tests, use form-data with key `file` (type File)
- If results are empty, try clearer images or provide text directly
- You can add `tessdata/eng.traineddata` locally for offline OCR consistency
