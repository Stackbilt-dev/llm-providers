#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${NPM_CONFIG_CACHE:-/tmp/.npm}"
export NPM_CONFIG_CACHE="$CACHE_DIR"
export npm_config_cache="$CACHE_DIR"

cd "$REPO_ROOT"
npm run build >/dev/null
PACK_NAME="$(npm pack --ignore-scripts | tail -1)"

TMP_DIR="$(mktemp -d /tmp/llmproviders-smoke-XXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

cp "$PACK_NAME" "$TMP_DIR/"
cd "$TMP_DIR"

NPM_CONFIG_CACHE="$CACHE_DIR" npm init -y >/dev/null
NPM_CONFIG_CACHE="$CACHE_DIR" npm install "./$PACK_NAME" >/dev/null

node -e "const p=require('@stackbilt/llm-providers'); if(!p||!p.LLMProviders) throw new Error('Missing LLMProviders from require');"
node --input-type=module -e "import * as p from '@stackbilt/llm-providers'; if(!p||!p.LLMProviders) throw new Error('Missing LLMProviders from import');"

echo "package smoke test passed"
