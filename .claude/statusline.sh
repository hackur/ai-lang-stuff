#!/bin/bash

# PCR Card Project Status Line for Claude Code
# Displays: Project | Model | Token Usage | Thinking Status | Session Usage | Plan

# Read JSON input from stdin
input=$(cat)

# Extract model information
model_id=$(echo "$input" | jq -r '.model.id // "unknown"')
model_display=$(echo "$input" | jq -r '.model.display_name // "Unknown Model"')

# Simplify model display name
case "$model_display" in
  *"Claude 3.5 Sonnet"*)
    model_short="Sonnet 3.5"
    ;;
  *"Sonnet 4.5"*|*"claude-sonnet-4-5"*)
    model_short="Sonnet 4.5"
    ;;
  *"Claude 3 Opus"*)
    model_short="Opus 3"
    ;;
  *"Claude 3 Haiku"*)
    model_short="Haiku 3"
    ;;
  *)
    model_short="$model_display"
    ;;
esac

# Get token budget (hardcoded 200K budget for status line)
token_budget=200000

# Try to extract current token usage from the most recent system warning
# This is approximate - reads from transcript file
transcript_path=$(echo "$input" | jq -r '.transcript_path // ""')
current_tokens=0

if [ -f "$transcript_path" ]; then
  # Extract last token usage warning from transcript
  last_usage=$(grep -o 'Token usage: [0-9]*' "$transcript_path" | tail -1 | grep -o '[0-9]*')
  if [ ! -z "$last_usage" ]; then
    current_tokens=$last_usage
  fi
fi

# Format token usage (convert to K notation)
if [ $current_tokens -ge 1000 ]; then
  tokens_display="$((current_tokens / 1000))K"
else
  tokens_display="${current_tokens}"
fi

budget_display="$((token_budget / 1000))K"

# Check thinking status from global settings
thinking_status="OFF"
if [ -f "$HOME/.claude/settings.json" ]; then
  thinking_enabled=$(jq -r '.alwaysThinkingEnabled // false' "$HOME/.claude/settings.json")
  if [ "$thinking_enabled" = "true" ]; then
    thinking_status="ON"
  fi
fi

# Calculate session usage - count messages in this conversation
session_messages=0
if [ -f "$transcript_path" ]; then
  # Count Human: and Assistant: occurrences as message pairs
  human_count=$(grep -c "^Human:" "$transcript_path" 2>/dev/null || echo "0")
  assistant_count=$(grep -c "^Assistant:" "$transcript_path" 2>/dev/null || echo "0")
  session_messages=$((human_count + assistant_count))
fi

# Get session start time and calculate duration
session_duration="0m"
if [ -f "$transcript_path" ]; then
  # Get file creation time (session start)
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    session_start=$(stat -f "%B" "$transcript_path" 2>/dev/null || echo "0")
  else
    # Linux
    session_start=$(stat -c "%Y" "$transcript_path" 2>/dev/null || echo "0")
  fi

  current_time=$(date +%s)
  if [ "$session_start" != "0" ]; then
    duration_seconds=$((current_time - session_start))
    duration_minutes=$((duration_seconds / 60))

    if [ $duration_minutes -ge 60 ]; then
      duration_hours=$((duration_minutes / 60))
      remaining_minutes=$((duration_minutes % 60))
      session_duration="${duration_hours}h${remaining_minutes}m"
    else
      session_duration="${duration_minutes}m"
    fi
  fi
fi

# Session usage display
session_info="${session_messages}msg/${session_duration}"

# Build status line with color codes (dim colors for terminal)
# Format: [PCR Card] Sonnet 4.5 | 38K/200K | Think: ON | 12msg/15m | Max 20x
printf "\033[2m[\033[0m\033[36mPCR Card\033[0m\033[2m]\033[0m "
printf "\033[33m%s\033[0m \033[2m|\033[0m " "$model_short"
printf "\033[32m%s\033[0m\033[2m/\033[0m\033[32m%s\033[0m \033[2m|\033[0m " "$tokens_display" "$budget_display"
printf "Think: \033[35m%s\033[0m \033[2m|\033[0m " "$thinking_status"
printf "\033[94m%s\033[0m \033[2m|\033[0m " "$session_info"
printf "\033[36m%s\033[0m" "Max 20x"
printf "\n"
