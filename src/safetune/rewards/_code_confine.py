"""Kernel confinement for the child process that runs model-written code (Linux).

The graders start the candidate in a fresh ``sys.executable -I`` process with limits on
CPU, memory and file size. This adds what limits can't: before the candidate runs, the
child confines itself with

* seccomp: no ``kill``/``tgkill``/``pidfd_send_signal`` (a completion could otherwise
  SIGKILL the trainer through ``os.getppid()``), no ``ptrace``, no sockets, no
  ``io_uring_setup``, no new processes or programs (``fork``/``clone``/``execve``), and
  nothing that changes another process: ``prlimit64`` and the ``sched_set*`` calls only on
  the caller (pid 0), no ``setpriority``/``ioprio_set``/``perf_event_open``/``keyctl``, and no
  ``F_SETOWN``/``FIOSETOWN`` (they make the kernel signal another process);
* Landlock: writes only under its own temporary directory; reads only from the interpreter's
  prefixes and library paths (stdlib, site-packages), the system library directories, its own
  ``/proc/<pid>``, and a short list of files (``/dev/urandom``, ``/proc/cpuinfo``, ``/etc/ld.so.cache``
  and the like), so not the user's home, a repository checkout, any other ``sys.path`` entry,
  ``/etc/passwd``, a listing of ``/dev`` or ``/proc``, or another process's ``/proc/<pid>``
  (the trainer's ``mem``, ``maps``, ``environ`` and ``cmdline``); and no TCP on ABI 4+.

Both are best effort. Without them (non-Linux, an old kernel, some containers) the child
is still a separate process with resource limits, and a one-time warning says so; without
seccomp it can also start processes, so the grader kills its process group, then finds and kills every
process that still carries the run's marker in its environment (Linux, best effort: that is why
``SAFETUNE_SANDBOX_REQUIRE_ISOLATION=1`` exists). Each child writes what isolation it really got over a pipe that it closes before the
candidate runs (the candidate can't write to it), and the grader decides on that report, not on any earlier probe. On Linux the
grading process also marks itself non-dumpable for as long as a candidate child is alive,
whatever the kernel offers.

``SAFETUNE_SANDBOX_REQUIRE_ISOLATION`` refuses to run candidate code (score 0, with a warning that
says which layer is missing) when the child didn't get the isolation asked for (the child itself
exits before the candidate runs): ``1``, ``true``,
``yes`` or ``full`` need BOTH seccomp and Landlock; ``any`` needs at least one. Unset (or ``0``,
``false``, ``no``, ``off``) keeps the best-effort default. Threads can't be started in confined
code; BLAS libraries are told to use one thread so ``numpy`` still imports.
"""

import json
import logging
import os
import sys
import threading
from typing import Any, Dict

logger = logging.getLogger(__name__)

# BLAS libraries start threads on import, which the confined child may not do.
SINGLE_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
}

# Defines ``isolate(tmp, require)`` for the child; ``tmp`` is the one writable directory.
ISOLATE_SOURCE = "import os, sys\n" + r'''
# The seccomp deny list, in one place. name -> (x86_64, aarch64); None: that architecture has no such call.
# The numbers are checked against the kernel headers (tests/runtime/test_code_confinement.py).
SECCOMP_DENY = [
    ("signals to other processes (a completion could SIGKILL the trainer through os.getppid())", {
        "kill": (62, 129), "tkill": (200, 130), "tgkill": (234, 131), "rt_sigqueueinfo": (129, 138),
        "rt_tgsigqueueinfo": (297, 240), "pidfd_send_signal": (424, 424)}),
    ("handles on other processes", {
        "pidfd_open": (434, 434), "pidfd_getfd": (438, 438), "process_madvise": (440, 440),
        "process_mrelease": (448, 448)}),
    ("reading or tracing other processes", {
        "ptrace": (101, 117), "process_vm_readv": (310, 270), "process_vm_writev": (311, 271),
        "kcmp": (312, 272), "perf_event_open": (298, 241)}),
    ("new processes, threads and programs", {
        "clone": (56, 220), "clone3": (435, 435), "fork": (57, None), "vfork": (58, None),
        "execve": (59, 221), "execveat": (322, 281)}),
    ("sockets and io_uring (which can do what the filter refuses)", {
        "socket": (41, 198), "io_uring_setup": (425, 425), "io_uring_enter": (426, 426),
        "io_uring_register": (427, 427)}),
    ("SysV IPC and POSIX message queues: segments and queues the trainer or NCCL created are reachable by any "
     "process of the same user", {
        "shmget": (29, 194), "shmat": (30, 196), "shmctl": (31, 195), "shmdt": (67, 197),
        "semget": (64, 190), "semop": (65, 193), "semctl": (66, 191), "semtimedop": (220, 192),
        "msgget": (68, 186), "msgsnd": (69, 189), "msgrcv": (70, 188), "msgctl": (71, 187),
        "mq_open": (240, 180), "mq_unlink": (241, 181), "mq_timedsend": (242, 182),
        "mq_timedreceive": (243, 183), "mq_notify": (244, 184), "mq_getsetattr": (245, 185)}),
    ("namespaces, mounts and other system administration", {
        "unshare": (272, 97), "setns": (308, 268), "mount": (165, 40), "umount2": (166, 39),
        "pivot_root": (155, 41), "chroot": (161, 51), "open_by_handle_at": (304, 265),
        "name_to_handle_at": (303, 264), "move_mount": (429, 429), "open_tree": (428, 428),
        "fsopen": (430, 430), "fsconfig": (431, 431), "fsmount": (432, 432), "fspick": (433, 433),
        "mount_setattr": (442, 442), "quotactl": (179, 60), "quotactl_fd": (443, 443),
        "swapon": (167, 224), "swapoff": (168, 225), "reboot": (169, 142), "kexec_load": (246, 104),
        "kexec_file_load": (320, 294), "init_module": (175, 105), "finit_module": (313, 273),
        "delete_module": (176, 106), "acct": (163, 89), "syslog": (103, 116)}),
    ("kernel objects shared with the trainer (the user's keyrings, bpf, userfaultfd, fanotify)", {
        "bpf": (321, 280), "userfaultfd": (323, 282), "fanotify_init": (300, 262), "add_key": (248, 217),
        "request_key": (249, 218), "keyctl": (250, 219)}),
    ("credentials (no-ops without privilege, refused anyway)", {
        "setuid": (105, 146), "setgid": (106, 144), "setreuid": (113, 145), "setregid": (114, 143),
        "setresuid": (117, 147), "setresgid": (119, 149), "setfsuid": (122, 151), "setfsgid": (123, 152),
        "setgroups": (116, 159), "capset": (126, 91)}),
    ("scheduling and I/O priority that can name a whole user (setpriority(PRIO_USER, 0))", {
        "setpriority": (141, 140), "ioprio_set": (251, 30)}),
]
# Allowed only on the caller itself: name -> (x86_64, aarch64, index of the pid argument); pid 0 is the caller.
SECCOMP_SELF_ONLY = {
    "prlimit64": (302, 261, 0), "sched_setparam": (142, 118, 0), "sched_setscheduler": (144, 119, 0),
    "sched_setaffinity": (203, 122, 0), "sched_setattr": (314, 274, 0), "migrate_pages": (256, 238, 0),
    "move_pages": (279, 239, 0),
}
# Refused for some commands: name -> (x86_64, aarch64, index of the command argument, commands). The kernel reads
# the command as 32 bits. F_SETOWN, F_SETSIG, F_SETOWN_EX and FIOSETOWN/SIOCSPGRP make the kernel signal another
# process (SIGIO, or the F_SETSIG signal) when a pipe or file becomes ready.
SECCOMP_CMD_DENY = {
    "fcntl": (72, 25, 1, (8, 10, 15)),
    "ioctl": (16, 29, 1, (0x8901, 0x8902)),
}


def seccomp_program(m):
    """The BPF program, a list of (code, jt, jf, k), for machine ``m``; None if the machine is unknown."""
    if m in ("x86_64", "AMD64"):
        arch, x32, col = 0xC000003E, True, 0
    elif m in ("aarch64", "arm64"):
        arch, x32, col = 0xC00000B7, False, 1
    else:
        return None
    deny = [nrs[col] for _, group in SECCOMP_DENY for nrs in group.values() if nrs[col] is not None]
    pid0 = [(v[col], v[2]) for v in SECCOMP_SELF_ONLY.values()]
    cmds = [(v[col], v[2], v[3]) for v in SECCOMP_CMD_DENY.values()]
    LD, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06
    ALLOW, EPERM = 0x7FFF0000, 0x00050001
    D = -1  # a jump to the final EPERM, patched below
    blocks = [(nr, [(LD, 0, 0, 16 + 8 * i), (JEQ, 0, D, 0), (LD, 0, 0, 20 + 8 * i), (JEQ, 0, D, 0),
                    (RET, 0, 0, ALLOW)]) for nr, i in pid0]
    blocks += [(nr, [(LD, 0, 0, 16 + 8 * i)] + [(JEQ, D, 0, v) for v in vals] + [(RET, 0, 0, ALLOW)])
               for nr, i, vals in cmds]
    n_checks = (1 if x32 else 0) + len(deny) + len(blocks)
    head = [(LD, 0, 0, 4), (JEQ, 1, 0, arch), (RET, 0, 0, EPERM), (LD, 0, 0, 0)]
    body_at = len(head) + n_checks + 1  # after the default ALLOW
    deny_at = body_at + sum(len(b) for _, b in blocks)
    prog = list(head)
    if x32:
        prog.append((JGE, deny_at - len(prog) - 1, 0, 0x40000000))
    for n in deny:
        prog.append((JEQ, deny_at - len(prog) - 1, 0, n))
    start = body_at
    for nr, b in blocks:
        prog.append((JEQ, start - len(prog) - 1, 0, nr))
        start += len(b)
    prog.append((RET, 0, 0, ALLOW))
    for nr, b in blocks:
        for code, jt, jf, k in b:
            here = len(prog)
            prog.append((code, deny_at - here - 1 if jt == D else jt, deny_at - here - 1 if jf == D else jf, k))
    prog.append((RET, 0, 0, EPERM))
    assert len(prog) == deny_at + 1 and deny_at < 255  # jump offsets are one byte
    return prog


def isolate(tmp, require):
    st = {"seccomp": False, "landlock": 0}
    try:
        import ctypes, platform
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        libc.prctl.restype = ctypes.c_int
        libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise OSError("no_new_privs")
    except Exception:
        return st
    # --- Landlock: writes only under tmp, reads only from Python + system dirs, no TCP ---
    try:
        def sc(*a):
            return libc.syscall(*[ctypes.c_long(x) if isinstance(x, int) else x for x in a])
        abi = sc(444, ctypes.c_void_p(0), 0, 1)  # LANDLOCK_CREATE_RULESET_VERSION
        if abi >= 1:
            fs = (1 << 13) - 1
            if abi >= 2: fs |= 1 << 13
            if abi >= 3: fs |= 1 << 14
            if abi >= 5: fs |= 1 << 15
            net = 3 if abi >= 4 else 0
            scoped = 3 if abi >= 6 else 0
            size = 24 if abi >= 6 else (16 if abi >= 4 else 8)
            attr = (ctypes.c_uint64 * 3)(fs, net, scoped)
            rfd = sc(444, ctypes.byref(attr), size, 0)
            if rfd < 0:
                raise OSError("create_ruleset")
            class PB(ctypes.Structure):
                _pack_ = 1
                _fields_ = [("allowed", ctypes.c_uint64), ("fd", ctypes.c_int32)]
            file_ok = 1 | 2 | 4 | (1 << 14) | (1 << 15)
            def allow(path, access):
                try:
                    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                except OSError:
                    return
                try:
                    import stat
                    a = access & fs
                    if not stat.S_ISDIR(os.fstat(fd).st_mode):
                        a &= file_ok
                    if a:
                        pb = PB(a, fd)
                        sc(445, rfd, 1, ctypes.byref(pb), 0)  # LANDLOCK_RULE_PATH_BENEATH
                finally:
                    os.close(fd)
            read = 1 | 4 | 8  # EXECUTE | READ_FILE | READ_DIR
            # Trees: the interpreter's own prefixes and library paths (and nothing else from sys.path: an
            # editable install or a PYTHONPATH-style entry is often a checkout under the user's home, and a
            # candidate needs none of it), the system library directories, this process's own /proc entry
            # (/proc/self is a symlink to it) and the CPU topology that BLAS and OpenMP read.
            trees = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
                     "/usr", "/lib", "/lib64", "/lib32", "/bin", "/sbin", "/opt", "/nix", "/run/current-system",
                     "/proc/%d" % os.getpid(), "/sys/devices/system/cpu", "/etc/ld.so.conf.d"}
            import sysconfig
            for key in ("stdlib", "platstdlib", "purelib", "platlib"):
                try:
                    trees.add(sysconfig.get_path(key))
                except Exception:
                    pass
            trees.discard(None)
            # Single files: what the dynamic loader, the C library and Python read. Not /proc/<other pid>
            # (so not the trainer's mem, maps, environ or cmdline), not a listing of /dev or /etc, not
            # /etc/passwd.
            files = ["/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/localtime", "/etc/timezone", "/etc/os-release",
                     "/etc/nsswitch.conf", "/dev/urandom", "/dev/random", "/dev/zero",
                     "/proc/cpuinfo", "/proc/meminfo", "/proc/stat", "/proc/version", "/proc/loadavg",
                     "/proc/uptime", "/proc/sys/vm/overcommit_memory"]
            for p in trees:
                allow(p, read)
            for p in files:
                allow(p, 4)  # READ_FILE
            allow(tmp, fs)
            allow("/dev/null", 2 | 4 | (1 << 14))
            if sc(446, rfd, 0) == 0:  # landlock_restrict_self
                st["landlock"] = abi
            os.close(rfd)
    except Exception:
        pass
    # --- seccomp: no signals to others, no ptrace, no sockets, no new processes or programs,
    #     and nothing that changes or reads another process (limits, scheduling, memory, perf) ---
    try:
        prog = seccomp_program(platform.machine())
        if prog:
            class F(ctypes.Structure):
                _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte),
                            ("k", ctypes.c_uint32)]
            class P(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(F))]
            arr = (F * len(prog))(*[F(c, jt, jf, k) for c, jt, jf, k in prog])
            fp = P(len(prog), arr)
            if libc.prctl(22, 2, ctypes.addressof(fp), 0, 0) == 0:  # PR_SET_SECCOMP, FILTER
                st["seccomp"] = True
    except Exception:
        pass
    return st
'''


_unrecognised_logged = set()


def require_isolation() -> str:
    """``""`` (best effort), ``"full"`` (seccomp and Landlock both) or ``"any"`` (at least one)."""
    value = os.environ.get("SAFETUNE_SANDBOX_REQUIRE_ISOLATION", "").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return ""
    if value == "any":
        return "any"
    if value not in ("1", "true", "yes", "full") and value not in _unrecognised_logged:
        _unrecognised_logged.add(value)
        logger.warning("Unrecognised SAFETUNE_SANDBOX_REQUIRE_ISOLATION=%r: treating it as 'full' "
                       "(1, true, yes, full: seccomp and Landlock; any: at least one; 0: off).", value)
    return "full"


def isolation_met(mode: str, state: Dict[str, Any]) -> bool:
    if mode == "full":
        return bool(state.get("seccomp") and state.get("landlock"))
    return bool(state.get("seccomp") or state.get("landlock"))


_unmet_logged = set()
_partial_logged: Dict[str, bool] = {}


def judge_report(state: Any) -> bool:
    """Judge what a candidate child reported about its own isolation; True if its result may be used.

    ``state`` is the child's ``{"seccomp": bool, "landlock": ABI}`` (what its own ``isolate()`` really achieved,
    written before the candidate ran), or None if it reported nothing. Logs once if the isolation is partial.
    With ``SAFETUNE_SANDBOX_REQUIRE_ISOLATION`` set, an unmet requirement (including no report at all) is
    logged once per reason and refused: the child itself exits without running the candidate in that case, so
    this is the second check. Nothing here is taken from a probe: only from the child that ran the candidate."""
    st = state if isinstance(state, dict) else {}
    if not (st.get("seccomp") and st.get("landlock")) and not _partial_logged:
        _partial_logged["done"] = True
        logger.warning(
            "SafeTune code-reward sandbox: kernel isolation is partial (seccomp=%s, landlock ABI=%s%s). Candidate "
            "code still runs in a separate process with limits, but may be able to signal other "
            "processes or read/write the user's files%s. Set SAFETUNE_SANDBOX_REQUIRE_ISOLATION=1 (both layers) or "
            "=any (at least one) to refuse to run candidate code without them.",
            st.get("seccomp"), st.get("landlock"), "" if state is not None else "; the child reported nothing",
            "; without seccomp it can also start processes" if not st.get("seccomp") else "")
    mode = require_isolation()
    if not mode or isolation_met(mode, st):
        return True
    missing = [name for name, on in (("seccomp", st.get("seccomp")), ("Landlock", st.get("landlock"))) if not on]
    if mode == "any":
        missing = ["seccomp and Landlock (neither is available)"]
    why = ("SAFETUNE_SANDBOX_REQUIRE_ISOLATION=%s needs %s, but the candidate's process got %s; candidate code "
           "is not run and scores 0." % (os.environ.get("SAFETUNE_SANDBOX_REQUIRE_ISOLATION", ""),
                                         "both seccomp and Landlock" if mode == "full" else "seccomp or Landlock",
                                         " and ".join(missing) if state is not None else "no report"))
    if why not in _unmet_logged:
        _unmet_logged.add(why)
        logger.warning(why)
    return False


_MAX_REPORT_BYTES = 4096


def parse_report(data: bytes) -> Any:
    """The state in ``data`` (what a child wrote to the report pipe); None unless it is exactly a plain state."""
    try:
        obj = json.loads(data.decode("utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("seccomp"), bool) and type(obj.get("landlock")) is int:
            return {"seccomp": obj["seccomp"], "landlock": obj["landlock"]}
    except Exception:
        pass
    return None


def read_report(fd: int, timeout: float = 10.0) -> Any:
    """Read the child's report from the read end of its report pipe (at most 4 KiB, until it closes the pipe or
    ``timeout`` passes). None if there is no valid report: a child that died or hung before reporting, or sent
    anything but a plain state. The child writes it right after ``isolate()`` and closes the pipe before the
    candidate runs, so the candidate can't write to it or even know its number."""
    import select
    import time

    data, deadline = b"", time.monotonic() + timeout
    try:
        while len(data) <= _MAX_REPORT_BYTES:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                break
            chunk = os.read(fd, _MAX_REPORT_BYTES + 1 - len(data))
            if not chunk:
                break  # the child closed its end
            data += chunk
    except OSError:
        return None
    return parse_report(data) if len(data) <= _MAX_REPORT_BYTES else None


def kill_marked(marker: str, wait: float = 2.0) -> bool:
    """Linux: kill every process whose initial environment carries ``SAFETUNE_GRADER_RUN=<marker>`` (a candidate's
    descendants, which inherit it) and return True once none is left; False if some survive ``wait`` seconds.

    Best effort: a process that rewrites its own environment block or starts a program with another
    environment isn't found. Without seccomp a candidate can do both, which is what
    ``SAFETUNE_SANDBOX_REQUIRE_ISOLATION=1`` is for. Nothing to do (True) where there is no /proc."""
    import signal
    import time

    if not os.path.isdir("/proc"):
        return True
    needle = b"SAFETUNE_GRADER_RUN=" + marker.encode()  # the variable the grader puts in the child's environment
    me = os.getpid()
    deadline = time.monotonic() + wait
    while True:
        found = []
        for name in os.listdir("/proc"):
            if not name.isdigit() or int(name) == me:
                continue
            try:
                with open("/proc/%s/environ" % name, "rb") as f:
                    if needle in f.read():
                        found.append(int(name))
            except OSError:
                continue  # gone, or not ours to read
        if not found:
            return True
        for pid in found:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)


def group_gone(pgid: int, wait: float = 1.0) -> bool:
    """Wait briefly for every process of group ``pgid`` to be gone (after it was killed); True if it is."""
    import time
    deadline = time.monotonic() + wait
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return True  # not ours to signal any more: nothing of ours is in it
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)


# Defence in depth for the trainer's memory. Landlock refuses a confined process any ptrace-checked access to
# the trainer (``/proc/<pid>/mem``, ``process_vm_readv``) even where Yama allows it, but nothing here relies on
# it being there: the grading process also drops its "dumpable" flag for as long as any candidate child is
# alive, which a process of the same user without CAP_SYS_PTRACE can't read or write through /proc either
# (``mem``, ``environ``, ``oom_score_adj``). The flag is restored afterwards; concurrent graders share one hold,
# counted under a lock. The only cost is that debuggers and profilers of the trainer can't attach for the
# seconds a candidate runs. Where seccomp is missing the candidate can fork, so the hold also lasts until the
# child's process group is gone and no process carrying the run's environment marker is left.
_hold_lock = threading.Lock()
_hold: Dict[str, Any] = {"n": 0, "restore": False}


def hold_parent() -> bool:
    """Call before starting a candidate child (Linux, unconditionally); pair with ``release_parent()``."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        with _hold_lock:
            if _hold["n"] == 0:
                # PR_GET_DUMPABLE == 1 (not 0, not 2) is the only state it is safe to flip and flip back
                _hold["restore"] = libc.prctl(3, 0, 0, 0, 0) == 1 and libc.prctl(4, 0, 0, 0, 0) == 0
            _hold["n"] += 1
        return True
    except Exception:
        return False


def release_parent() -> None:
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        with _hold_lock:
            _hold["n"] -= 1
            if _hold["n"] == 0 and _hold["restore"]:
                libc.prctl(4, 1, 0, 0, 0)  # PR_SET_DUMPABLE
                _hold["restore"] = False
    except Exception:
        pass
