# APE Test Agent - FastAPI Implementation

A complete implementation of the APE (Agent Protocol Evaluation) test agent that supports all 6 basic capabilities.

## Features

✅ **LLM-style General QA** - Multi-model voting for accurate answers
✅ **Tool Usage** - Hash functions (MD5, SHA512) and calculations  
✅ **Image Understanding** - Cat/dog classification using Claude 3.5 Sonnet
✅ **Web Browsing** - Real Tic-Tac-Toe gameplay with minimax algorithm
✅ **Code Execution** - Generate and execute Python/JavaScript code
✅ **Memory** - Persistent storage across sessions

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with your OpenRouter API key:
```
OPENROUTER_API_KEY=your_api_key_here
```

3. (Optional) Install ChromeDriver for Selenium:
- Download from https://chromedriver.chromium.org/
- Add to PATH

## Running the Agent

```bash
python fastapi_app.py
```

The agent will start on `http://localhost:5000`

## Testing with APE

1. Open the APE evaluation toolkit
2. Enter agent URL: `http://localhost:5000`
3. Run individual tests or full suite

## API Endpoints

- `POST /` - Main JSON-RPC endpoint
- `GET /.well-known/agent-card.json` - Agent capabilities

## Implementation Details

- Uses A2A (Agent-to-Agent) protocol with JSON-RPC 2.0
- Supports multimodal inputs (text and images)
- Real web interaction for Tic-Tac-Toe using Selenium
- Perfect game strategy with minimax algorithm