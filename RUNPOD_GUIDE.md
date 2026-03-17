# RunPod SSH Guide for Claude Code

RunPod's SSH environment has several quirks that make automated interaction difficult. This guide documents what works, what doesn't, and how to work around the issues.

## The Core Problem

RunPod's SSH server **blocks PTY allocation** from non-interactive clients. This means:

- `ssh user@pod "command"` → **FAILS** with `Error: Your SSH client doesn't support PTY`
- `ssh -T user@pod "command"` → **FAILS** (same error)
- `scp` → **FAILS** with `subsystem request failed on channel 0` (SFTP subsystem blocked)
- `rsync` over SSH → **FAILS** (same reason)

Interactive SSH from a terminal works fine (the user can SSH in manually).

## What DOES Work

### 1. Forced TTY with stdin piping (`-tt` flag)

```bash
printf 'ls /workspace\nexit\n' | ssh -tt -o StrictHostKeyChecking=no -i KEY user@ssh.runpod.io 2>&1 | grep -v "RUNPOD\|Enjoy\|_____\|runpod\|docs\.\|blog\."
```

- `-tt` forces PTY allocation even when stdin is not a terminal
- Commands are piped via `printf` with `\nexit\n` at the end
- Output includes the RunPod banner, echoed commands, and ANSI escape codes — must be filtered
- **Reliability**: works ~80% of the time. Sometimes hangs or drops connection.
- **Timeout**: always set a timeout (15-30s) on these commands

### 2. Reading files via `cat` through forced TTY

```bash
printf 'cat /path/to/file.json\nexit\n' | ssh -tt -o StrictHostKeyChecking=no -i KEY user@ssh.runpod.io 2>&1 | sed 's/\r//g' > /tmp/raw_output.txt
```

**CRITICAL**: The output will contain:
1. The echoed command itself (appears 2-3 times due to TTY echo)
2. The RunPod MOTD banner
3. The actual file content
4. ANSI escape codes (`\r`, `[?2004h`, color codes, etc.)

To extract clean JSON from this mess:
```python
import json

raw = open("/tmp/raw_output.txt").read()
# Find the largest top-level JSON object (tracks brace depth)
depth = 0; start = None; best_obj = None; best_len = 0
for i, c in enumerate(raw):
    if c == '{':
        if depth == 0: start = i
        depth += 1
    elif c == '}':
        depth -= 1
        if depth == 0 and start is not None:
            candidate = raw[start:i+1]
            if len(candidate) > best_len:
                best_obj = candidate; best_len = len(candidate)
            start = None
parsed = json.loads(best_obj)
```

This approach takes the **largest** complete JSON object, which avoids grabbing nested sub-objects or echoed duplicates.

### 3. HTTP file transfer (if port is exposed)

If a port is exposed in the RunPod dashboard:
```bash
# On pod:
python3 -m http.server 9999 --directory /workspace

# Locally:
curl -o file.tar.gz https://PODID-9999.proxy.runpod.net/file.tar.gz
```

**Note**: Port 8888 is usually taken by Jupyter. Use a different port AND make sure it's listed in the pod's exposed ports config. RunPod's proxy only forwards pre-configured ports.

### 4. GitHub push from pod (HTTPS token)

The most reliable way to transfer large/many files:
1. Generate a GitHub Personal Access Token (classic, `repo` scope, short expiry)
2. On the pod:
```bash
cd /workspace/project
git remote set-url origin https://USERNAME:TOKEN@github.com/USERNAME/repo.git
git add files
git commit -m "message"
git push
```
3. Pull locally

## What DOES NOT Work

| Method | Error | Why |
|--------|-------|-----|
| `ssh user@pod "cmd"` | PTY not supported | RunPod rejects non-interactive PTY |
| `ssh -T user@pod "cmd"` | Same | Disabling PTY still rejected |
| `scp` | Subsystem request failed | SFTP subsystem not available |
| `rsync -e ssh` | Same | Uses SFTP under the hood |
| `ssh-copy-id` | Same | Uses SFTP |

## Important: Preserve Your Checkpoints!

**RunPod pod storage is ephemeral.** When you terminate a pod, everything on local storage (`/root/`, `/tmp/`) is deleted. Only the **network volume** (mounted at `/workspace/`) persists.

**ALWAYS save model checkpoints to `/workspace/`**, not to the pod's local filesystem. If training scripts default to saving in the current directory or `/root/`, override the output path to point to `/workspace/`.

## SSH Connection String Format

```
ssh PODID-USERID@ssh.runpod.io -i ~/.ssh/id_ed25519
```

The pod ID changes each time you create a new pod, even with the same network volume. Check the RunPod dashboard for the current SSH command.

## Tips

- Always use `tmux` or `screen` on the pod for long-running jobs — SSH disconnects are common
- Set `--num-workers 0` in PyTorch DataLoader to avoid shared memory bus errors
- The Jupyter terminal (via web browser) is sometimes more reliable than SSH for interactive work
- For image files: do NOT cat/pipe binary files through SSH — they will corrupt. Use HTTP transfer or git instead.
