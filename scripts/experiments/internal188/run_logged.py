"""Stream a subprocess to Colab while preserving a diagnostic log on Drive."""
import subprocess
from collections import deque
from pathlib import Path


def run(command, log_path, cwd=None):
    log_path = Path(log_path); log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=35)
    with log_path.open('w') as log, subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                                     stderr=subprocess.STDOUT, text=True,
                                                     bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line); log.flush()
            tail.append(line.rstrip())
        code = process.wait()
    if code:
        raise RuntimeError(f'Command exited {code}. Log: {log_path}\nLast output:\n' + '\n'.join(tail))
    return log_path
