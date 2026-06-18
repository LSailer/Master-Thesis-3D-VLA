---
name: linear-cli
description: Create and inspect Linear issues from this repo using the installed `lin` CLI when possible, with a direct GraphQL fallback. Use when the user asks to create, read, update, or comment on Linear issues, mentions Linear CLI, or needs a Linear issue key before creating a worktree.
---

# Linear CLI

## Scope

Use Linear as canonical tracker for this repo's `3D-WM-ObjectNAV` team. Prefer `lin` when it works; fall back to Linear GraphQL when the installed CLI is unauthenticated or incompatible with Linear's current schema.

## Safety

- Never print API keys or `.env` contents.
- Load `LINEAR_API_KEY` from `.env` or the parent checkout `.env` only inside the command environment.
- Do not commit `.env` or `~/.lin/store.apiKey.json`.
- If no API key is available, prepare exact issue/comment text for the user to paste.

## Quick start

Check CLI and auth:

```bash
command -v lin
lin --help
```

If `.env` contains `LINEAR_API_KEY`, seed the CLI cache without echoing the key:

```bash
export LINEAR_API_KEY=$(awk -F= '$1=="LINEAR_API_KEY" {print substr($0, index($0,"=")+1)}' .env | tr -d '"\r')
mkdir -p ~/.lin
python - <<'PY'
import json, os, pathlib
path = pathlib.Path.home() / '.lin' / 'store.apiKey.json'
path.write_text(json.dumps(os.environ['LINEAR_API_KEY']))
path.chmod(0o600)
PY
```

Create issue via CLI:

```bash
lin new \
  --team "3D-WM-ObjectNAV" \
  --title "Issue title" \
  --description "$(cat /tmp/issue.md)"
```

## GraphQL fallback

Use when `lin` fails, for example with schema errors like `Cannot query field "milestone" on type "Project"`.

Find team ID:

```bash
export LINEAR_API_KEY=$(awk -F= '$1=="LINEAR_API_KEY" {print substr($0, index($0,"=")+1)}' .env | tr -d '"\r')
python - <<'PY'
import json, os, urllib.request
query = 'query { teams(first: 100) { nodes { id key name } } }'
req = urllib.request.Request(
    'https://api.linear.app/graphql',
    data=json.dumps({'query': query}).encode(),
    headers={'Content-Type': 'application/json', 'Authorization': os.environ['LINEAR_API_KEY']},
)
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.load(resp)
for team in data['data']['teams']['nodes']:
    if team['key'] == '3D' or team['name'] == '3D-WM-ObjectNAV':
        print(team['id'], team['key'], team['name'])
PY
```

Create issue:

```bash
export LINEAR_API_KEY=$(awk -F= '$1=="LINEAR_API_KEY" {print substr($0, index($0,"=")+1)}' .env | tr -d '"\r')
python - <<'PY'
import json, os, pathlib, urllib.request
team_id = '063d398d-f214-4db9-9d46-54bcd83a51a8'  # 3D-WM-ObjectNAV
query = '''mutation IssueCreate($input: IssueCreateInput!) {
  issueCreate(input: $input) { success issue { id identifier title url } }
}'''
variables = {'input': {
    'teamId': team_id,
    'title': 'Issue title',
    'description': pathlib.Path('/tmp/issue.md').read_text(),
}}
req = urllib.request.Request(
    'https://api.linear.app/graphql',
    data=json.dumps({'query': query, 'variables': variables}).encode(),
    headers={'Content-Type': 'application/json', 'Authorization': os.environ['LINEAR_API_KEY']},
)
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.load(resp)
if 'errors' in data:
    raise SystemExit(json.dumps(data['errors'], indent=2))
issue = data['data']['issueCreate']['issue']
print(issue['identifier'])
print(issue['url'])
PY
```

Read issue by key:

```bash
export LINEAR_API_KEY=$(awk -F= '$1=="LINEAR_API_KEY" {print substr($0, index($0,"=")+1)}' .env | tr -d '"\r')
python - <<'PY'
import json, os, urllib.request
query = '''query($id:String!) {
  issue(id:$id) { identifier title description url state { name } }
}'''
req = urllib.request.Request(
    'https://api.linear.app/graphql',
    data=json.dumps({'query': query, 'variables': {'id': '3D-87'}}).encode(),
    headers={'Content-Type': 'application/json', 'Authorization': os.environ['LINEAR_API_KEY']},
)
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.load(resp)
issue = data['data']['issue']
print(issue['identifier'], issue['title'])
print(issue['url'])
print('state:', issue['state']['name'])
PY
```

## Worktree after issue creation

Once Linear returns a key, create the worktree from the main checkout:

```bash
git worktree add worktrees/<linear-key>-<slug> -b <linear-key>-<slug>
```
