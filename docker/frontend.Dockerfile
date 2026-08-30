# --- frontend image ----------------------------------------------------------
# Static single-page UI served by nginx. nginx also reverse-proxies /api to the
# app service, so the browser talks to one origin and the API base URL is not
# baked into the build - it is resolved at runtime by /config.js, which nginx
# renders from the API_BASE_URL environment variable at container start.

FROM nginx:1.27-alpine

COPY frontend/index.html /usr/share/nginx/html/index.html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY docker/frontend-entrypoint.sh /docker-entrypoint.d/40-render-config.sh
RUN chmod +x /docker-entrypoint.d/40-render-config.sh

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
    CMD wget -q -O /dev/null http://localhost/ || exit 1
