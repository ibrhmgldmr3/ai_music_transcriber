# Build from the repository root:  docker build -f apps/web/Dockerfile .
FROM node:20-alpine

# Run as the image's unprivileged "node" user; it owns the app directory so the dev
# server can write .next/.
RUN mkdir -p /repo/apps/web /repo/packages && chown -R node:node /repo
USER node

WORKDIR /repo
COPY --chown=node:node packages/shared-types packages/shared-types
COPY --chown=node:node apps/web/package.json apps/web/package-lock.json apps/web/

WORKDIR /repo/apps/web
RUN npm ci
COPY --chown=node:node apps/web .

EXPOSE 3000
CMD ["npm", "run", "dev"]
