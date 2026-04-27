"""Manage tutor and student vLLM servers on a single GPU via sleep/wake.

Both servers are launched with `--enable-sleep-mode` and put to sleep (level 1)
right after they become ready, which offloads weights to CPU RAM and frees GPU
memory. Before each LLM call the active role is woken and the previous role is
put to sleep, so only one model holds GPU weights at any moment.
"""
from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests


@dataclass
class ServerSpec:
    name: str
    model: str
    host: str = "localhost"
    port: int = 8000
    gpu_memory_utilization: float = 0.45
    max_model_len: int = 8192
    dtype: str = "float16"
    extra_args: List[str] = field(default_factory=list)

    @property
    def root_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class VLLMServer:
    def __init__(self, spec: ServerSpec, cuda_visible: str = "0"):
        self.spec = spec
        self.cuda_visible = cuda_visible
        self.process: Optional[subprocess.Popen] = None
        self.is_sleeping = False

    def start(self, startup_timeout: int = 900) -> None:
        if self.process is not None:
            return
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible
        # /sleep and /wake_up are only registered when dev mode is on.
        env["VLLM_SERVER_DEV_MODE"] = "1"
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.spec.model,
            "--host", self.spec.host,
            "--port", str(self.spec.port),
            "--dtype", self.spec.dtype,
            "--max-model-len", str(self.spec.max_model_len),
            "--gpu-memory-utilization", str(self.spec.gpu_memory_utilization),
            "--enable-sleep-mode",
        ] + list(self.spec.extra_args)
        print(f"[vllm_manager] launching {self.spec.name}: {' '.join(cmd)}",
              file=sys.stderr)
        self.process = subprocess.Popen(cmd, env=env, start_new_session=True)
        self._wait_ready(startup_timeout)
        print(f"[vllm_manager] {self.spec.name} ready on :{self.spec.port}",
              file=sys.stderr)

    def _wait_ready(self, timeout: int) -> None:
        assert self.process is not None
        t0 = time.time()
        url = f"{self.spec.root_url}/v1/models"
        while time.time() - t0 < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM {self.spec.name} exited before ready "
                    f"(code={self.process.returncode})"
                )
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(3)
        raise RuntimeError(
            f"vLLM {self.spec.name} not ready within {timeout}s"
        )

    def sleep(self, level: int = 1) -> None:
        if self.is_sleeping:
            return
        r = requests.post(
            f"{self.spec.root_url}/sleep",
            params={"level": level},
            timeout=180,
        )
        r.raise_for_status()
        self.is_sleeping = True
        print(f"[vllm_manager] {self.spec.name} slept (level={level})",
              file=sys.stderr)

    def wake_up(self) -> None:
        if not self.is_sleeping:
            return
        r = requests.post(f"{self.spec.root_url}/wake_up", timeout=300)
        r.raise_for_status()
        self.is_sleeping = False
        print(f"[vllm_manager] {self.spec.name} woke up", file=sys.stderr)

    def stop(self) -> None:
        if self.process is None:
            return
        print(f"[vllm_manager] stopping {self.spec.name}", file=sys.stderr)
        try:
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                self.process.terminate()
            except Exception:
                pass
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                self.process.kill()
            self.process.wait()
        self.process = None


class VLLMManager:
    def __init__(self, servers: Dict[str, VLLMServer]):
        self.servers = servers
        self.active_role: Optional[str] = None
        self._lock = threading.Lock()
        self._started = False

    def startup(self) -> None:
        if self._started:
            return
        # Register cleanup before touching any server, so a failure partway
        # through startup still tears down whichever processes did start.
        atexit.register(self.shutdown)
        # Start sequentially; sleep each one right after readiness so the
        # next server can allocate its share of GPU memory without OOM.
        for name, srv in self.servers.items():
            srv.start()
            srv.sleep(level=1)
        self._started = True

    def ensure_active(self, role: str) -> None:
        with self._lock:
            if self.active_role == role:
                return
            if role not in self.servers:
                raise KeyError(
                    f"unknown role {role}; have {list(self.servers)}"
                )
            if self.active_role is not None:
                self.servers[self.active_role].sleep(level=1)
            self.servers[role].wake_up()
            self.active_role = role

    def shutdown(self) -> None:
        for srv in self.servers.values():
            try:
                srv.stop()
            except Exception as e:
                print(f"[vllm_manager] shutdown error for {srv.spec.name}: {e}",
                      file=sys.stderr)


def _split_host_port(base_url: str) -> Tuple[str, int]:
    p = urlparse(base_url)
    if p.hostname is None or p.port is None:
        raise ValueError(f"cannot parse host/port from {base_url!r}")
    return p.hostname, p.port


def build_manager_from_config(cfg: dict) -> Optional[VLLMManager]:
    mcfg = cfg.get("vllm_manager", {})
    if not mcfg.get("enabled", False):
        return None

    tutor_cfg = cfg["tutor_server"]
    student_cfg = cfg["student_server"]

    gmu = float(mcfg.get("gpu_memory_utilization", 0.45))
    max_len = int(mcfg.get("max_model_len", 8192))
    dtype = str(mcfg.get("dtype", "float16"))
    cuda = str(mcfg.get("cuda_visible_devices", "0"))
    extra = list(mcfg.get("extra_args", []))

    t_host, t_port = _split_host_port(tutor_cfg["base_url"])
    s_host, s_port = _split_host_port(student_cfg["base_url"])

    tutor_spec = ServerSpec(
        name="tutor", model=tutor_cfg["model"], host=t_host, port=t_port,
        gpu_memory_utilization=gmu, max_model_len=max_len, dtype=dtype,
        extra_args=extra,
    )
    student_spec = ServerSpec(
        name="student", model=student_cfg["model"], host=s_host, port=s_port,
        gpu_memory_utilization=gmu, max_model_len=max_len, dtype=dtype,
        extra_args=extra,
    )

    return VLLMManager({
        "tutor": VLLMServer(tutor_spec, cuda_visible=cuda),
        "student": VLLMServer(student_spec, cuda_visible=cuda),
    })
