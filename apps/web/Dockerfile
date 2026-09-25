# Build from the repository root:  docker build -f apps/web/Dockerfile .
# Production build served by `next start`. NEXT_PUBLIC_API_URL is baked into the client
# bundle, so it is a build argument (docker-compose.yml passes it).

FROM node:22-alpine AS build
WORKDIR /repo
COPY packages/shared-types packages/shared-types
COPY apps/web/package.json apps/web/package-lock.json apps/web/
WORKDIR /repo/apps/web
RUN npm ci
COPY apps/web .
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL NEXT_TELEMETRY_DISABLED=1
RUN npm run build && npm prune --omit=dev

FROM node:22-alpine
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
WORKDIR /repo
# node_modules links @music-transcriber/shared-types to ../../packages/shared-types.
COPY --from=build --chown=node:node /repo/packages/shared-types packages/shared-types
COPY --from=build --chown=node:node /repo/apps/web apps/web
# Run as the image's unprivileged "node" user.
USER node
WORKDIR /repo/apps/web
EXPOSE 3000
CMD ["npm", "run", "start", "--", "-H", "0.0.0.0", "-p", "3000"]
