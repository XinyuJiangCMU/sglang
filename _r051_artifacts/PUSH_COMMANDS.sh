#!/bin/bash
# r051 — push + open PR commands. Needs GH_TOKEN exported.
# Run from /mnt/data/data/hai/workspace/sglang
set -e

if [ -z "$GH_TOKEN" ]; then
  echo "ERROR: export GH_TOKEN=<your_github_token> first"
  exit 1
fi

# 1. Push the dedicated PR branch
git push https://XinyuJiangCMU:${GH_TOKEN}@github.com/XinyuJiangCMU/sglang.git \
  pr/r051-flydsl-attention:pr/r051-flydsl-attention

# 2. Push the dev branch too (preserves full history)
git push https://XinyuJiangCMU:${GH_TOKEN}@github.com/XinyuJiangCMU/sglang.git \
  dev/flydsl-attention:dev/flydsl-attention

# 3. Open PR via GitHub API (gh CLI not installed in this env, use curl)
BODY_FILE=_r051_artifacts/PR_DESCRIPTION.md
TITLE="[AMD/gfx950] FlyDSL DSv4 sparse FP8 MLA decode — 3.89x faster than tilelang"
BASE="pr/dsv4-tilelang-bs-adaptive-block-per-cu"   # parent of round-1 commit; tilelang_kernel + adapter already present

# escape body for JSON
BODY_JSON=$(python3 -c "import json; print(json.dumps(open('$BODY_FILE').read()))")

curl -X POST \
  -H "Authorization: token ${GH_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/XinyuJiangCMU/sglang/pulls \
  -d "{
    \"title\": \"${TITLE}\",
    \"head\":  \"pr/r051-flydsl-attention\",
    \"base\":  \"${BASE}\",
    \"body\":  ${BODY_JSON}
  }"
