#!/bin/sh
# Render /config.js at container start so one image works in every environment.
# The SPA reads window.__CONFIG__.apiBase; default is the same-origin /api proxy.
set -eu

API_BASE="${API_BASE_URL:-/api}"
cat > /usr/share/nginx/html/config.js <<EOF
window.__CONFIG__ = { apiBase: "${API_BASE}" };
EOF
echo "rendered /config.js with apiBase=${API_BASE}"
