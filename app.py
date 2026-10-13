"""AI chatbot web server with live tools.

Flask app that sends the conversation to a model on AWS Bedrock. The model
can call tools (web search, weather, current time) to get up-to-date
information, then answers using the results. On EC2 it authenticates
through the instance's IAM role, so no API keys are stored anywhere.
"""
import datetime
import os
from zoneinfo import ZoneInfo

import boto3
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from flask import Flask, jsonify, request, send_file

REGION = os.environ.get("AWS_REGION", "eu-north-1")
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
)
TIMEZONE = os.environ.get("TIMEZONE", "Europe/Stockholm")
BASE_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful, friendly assistant. Answer clearly and concisely, in "
    "the same language the user writes in. You have tools for live information: "
    "use get_current_time for the date or time, get_weather for weather, and "
    "web_search for news, prices, sports, people in current roles, or anything "
    "that may have changed recently. Use a tool whenever a question depends on "
    "current facts instead of guessing. When you use web_search, mention the "
    "source website names. If a tool fails, say so honestly.",
)
MAX_HISTORY = 20        # messages kept per request
MAX_MESSAGE_CHARS = 4000
MAX_TOOL_ROUNDS = 4     # how many tool calls in a row the model may make
HTTP_TIMEOUT = 10

app = Flask(__name__)
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    config=Config(read_timeout=60, retries={"max_attempts": 2}),
)


# ---------------------------------------------------------------- tools ----
def tool_get_current_time(_args):
    now = datetime.datetime.now(ZoneInfo(TIMEZONE))
    return {
        "timezone": TIMEZONE,
        "iso": now.isoformat(timespec="seconds"),
        "readable": now.strftime("%A %d %B %Y, %H:%M"),
    }


def tool_get_weather(args):
    place = str(args.get("location", "")).strip()
    if not place:
        return {"error": "location is required"}
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": place, "count": 1},
        timeout=HTTP_TIMEOUT,
    ).json()
    if not geo.get("results"):
        return {"error": f"could not find a place called '{place}'"}
    spot = geo["results"][0]
    data = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": spot["latitude"],
            "longitude": spot["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                       "precipitation,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "forecast_days": 3,
            "timezone": "auto",
        },
        timeout=HTTP_TIMEOUT,
    ).json()
    return {
        "place": f"{spot['name']}, {spot.get('country', '')}",
        "current": data.get("current"),
        "units": data.get("current_units"),
        "daily_forecast": data.get("daily"),
        "note": "weather_code is a WMO code (0 clear, 1-3 cloudy, 45-48 fog, "
                "51-67 rain/drizzle, 71-77 snow, 80-82 showers, 95-99 thunderstorm)",
    }


def tool_web_search(args):
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "query is required"}
    from ddgs import DDGS  # imported here so the app starts even if missing

    results = DDGS().text(query, max_results=5)
    return {
        "results": [
            {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
            for r in results
        ]
    }


TOOL_FUNCTIONS = {
    "get_current_time": tool_get_current_time,
    "get_weather": tool_get_weather,
    "web_search": tool_web_search,
}

TOOL_CONFIG = {
    "tools": [
        {"toolSpec": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }},
        {"toolSpec": {
            "name": "get_weather",
            "description": "Get the current weather and a 3-day forecast for a city or place.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"location": {
                    "type": "string", "description": "City or place name, e.g. 'Stockholm'"}},
                "required": ["location"],
            }},
        }},
        {"toolSpec": {
            "name": "web_search",
            "description": "Search the web for current information: news, prices, "
                           "sports results, recent events, facts that may have changed.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            }},
        }},
    ]
}


def run_tool(name, args):
    """Run one tool and return (result_dict, status)."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"unknown tool {name}"}, "error"
    try:
        return func(args or {}), "success"
    except Exception as err:  # tools must never crash the chat
        app.logger.error("Tool %s failed: %s", name, err)
        return {"error": f"{name} failed: {err}"}, "error"


# ----------------------------------------------------------------- chat ----
def system_prompt():
    now = datetime.datetime.now(ZoneInfo(TIMEZONE)).strftime("%A %d %B %Y, %H:%M")
    return f"{BASE_PROMPT}\n\nCurrent date and time ({TIMEZONE}): {now}."


def clean_history(raw):
    """Validate the client-supplied history and return Bedrock-formatted messages."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("messages must be a non-empty list")
    messages = []
    for item in raw[-MAX_HISTORY:]:
        role = item.get("role") if isinstance(item, dict) else None
        text = item.get("content") if isinstance(item, dict) else None
        if role not in ("user", "assistant") or not isinstance(text, str) or not text.strip():
            raise ValueError("invalid message format")
        messages.append({"role": role, "content": [{"text": text[:MAX_MESSAGE_CHARS]}]})
    # The conversation must start with and end on a user turn.
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("conversation must end with a user message")
    return messages


def run_conversation(messages):
    """Call the model, run any tools it asks for, and return the final text."""
    system = [{"text": system_prompt()}]
    for _ in range(MAX_TOOL_ROUNDS + 1):
        response = bedrock.converse(
            modelId=MODEL_ID,
            system=system,
            messages=messages,
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": 1000, "temperature": 0.5},
        )
        message = response["output"]["message"]
        messages.append(message)
        if response.get("stopReason") != "tool_use":
            text = "".join(b["text"] for b in message["content"] if "text" in b)
            return text.strip() or "Sorry, I could not produce an answer."
        results = []
        for block in message["content"]:
            if "toolUse" in block:
                use = block["toolUse"]
                app.logger.info("Tool call: %s %s", use["name"], use.get("input"))
                output, status = run_tool(use["name"], use.get("input"))
                results.append({"toolResult": {
                    "toolUseId": use["toolUseId"],
                    "content": [{"json": output}],
                    "status": status,
                }})
        messages.append({"role": "user", "content": results})
    return "Sorry, that took too many steps. Please try a simpler question."


@app.get("/")
def index():
    return send_file("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL_ID)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    try:
        messages = clean_history(data.get("messages"))
    except ValueError as err:
        return jsonify(error=str(err)), 400

    try:
        reply = run_conversation(messages)
    except ClientError as err:
        app.logger.error("Bedrock error: %s", err)
        return jsonify(error="The AI service is unavailable right now."), 502
    return jsonify(reply=reply)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
