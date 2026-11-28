/**
 * Lifecycle of the local inference backends, tied to pi. Two backends, mutually exclusive (both in memory at once would
 * exceed 24 GB):
 *   - provider "lamq":  lamq/lamq-server (127.0.0.1:9932), lamq MLX runtime + runtime refusal ablation + vision.
 *   - provider "local": llama.cpp/router (127.0.0.1:9931), llama.cpp fallback.
 * Both load the model on the first request. The repo is found from this file's real path; INKLING_MAC_REPO overrides it.
 *
 * - pi does nothing at startup. The first request to one of these providers starts the matching backend. If the other
 *   backend was started by this extension, it is stopped first (freeing its memory). If another pi is using that other
 *   backend at the moment, its in-flight request breaks: on a single-user machine, switching models is a deliberate act.
 * - Idle unload is handled by each backend (both after 10 minutes), keeping the process and prompt cache until then.
 * - Every pi process that used a local model leaves a lease in ~/.cache/inkling-mac/sessions/<pid> (content: the process
 *   start time, to guard against pid reuse). When pi exits (Ctrl+C/Ctrl+D/SIGHUP/SIGTERM all go through
 *   session_shutdown "quit") the lease is dropped. Once no live leases remain, every backend this extension started is
 *   stopped and memory is freed immediately.
 * - Only backends this extension started are stopped: <name>.json must match on pid, start time and command line.
 *   Backends run by hand in the foreground are left alone.
 * - When pi is SIGKILLed or crashes no hook runs; the backends' idle unload is the fallback, and stale leases are
 *   cleared on the next start.
 */
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { spawn, spawnSync } from "node:child_process";
import { closeSync, mkdirSync, openSync, readdirSync, readFileSync, realpathSync, rmdirSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";

const HOME = homedir();
const REPO = process.env.INKLING_MAC_REPO ?? dirname(dirname(realpathSync(fileURLToPath(import.meta.url))));
const RUN = join(HOME, ".cache/inkling-mac");
const LEASES = join(RUN, "sessions");
const START_TIMEOUT_MS = 90_000;
const STOP_TIMEOUT_MS = 8_000;
const STATUS_KEY = "local-llm";

type Backend = {
  provider: string;
  base: string;
  healthPath: string;
  launcher: string;
  log: string;
  /** Command-line check: is this process the one we started (not some other process that reused the pid)? */
  owns: (cmd: string) => boolean;
};

const BACKENDS: Backend[] = [
  {
    provider: "local",
    base: "http://127.0.0.1:9931",
    healthPath: "/health",
    launcher: join(REPO, "llama.cpp/router"),
    log: join(HOME, "Library/Logs/inkling-mac-llama.log"),
    owns: (cmd) => cmd.includes("--models-preset") && cmd.includes("--port 9931"),
  },
  {
    provider: "lamq",
    base: "http://127.0.0.1:9932",
    healthPath: "/health",
    launcher: join(REPO, "lamq/lamq-server"),
    log: join(HOME, "Library/Logs/inkling-mac-lamq.log"),
    owns: (cmd) => cmd.includes(join(REPO, "lamq/server.py")),
  },
  {
    provider: "splash",
    base: "http://127.0.0.1:9933",
    healthPath: "/ready",
    launcher: join(REPO, "splash/splash-server"),
    log: join(HOME, "Library/Logs/inkling-mac-splash.log"),
    owns: (cmd) => cmd.includes("splash") && cmd.includes("9933"),
  },
];

const recordFile = (b: Backend) => join(RUN, `${b.provider}.json`);
const startLock = (b: Backend) => join(RUN, `${b.provider}.start.lock`);

/** Process start time (ps lstart), or undefined if the process doesn't exist. A reused pid always has a different start time. */
function startTime(pid: number): string | undefined {
  const r = spawnSync("ps", ["-o", "lstart=", "-p", String(pid)], { encoding: "utf8" });
  const s = r.status === 0 ? r.stdout.trim() : "";
  return s || undefined;
}

function command(pid: number): string {
  const r = spawnSync("ps", ["-o", "command=", "-p", String(pid)], { encoding: "utf8" });
  return r.status === 0 ? r.stdout.trim() : "";
}

function children(pid: number): number[] {
  const r = spawnSync("pgrep", ["-P", String(pid)], { encoding: "utf8" });
  return r.stdout.split("\n").map(Number).filter((n) => n > 0);
}

function readJson<T>(path: string): T | undefined {
  try { return JSON.parse(readFileSync(path, "utf8")) as T; } catch { return undefined; }
}

type ProcRecord = { pid: number; start: string };

/** The backend process this extension started, if it is still running. */
function owned(b: Backend): ProcRecord | undefined {
  const rec = readJson<ProcRecord>(recordFile(b));
  if (!rec || startTime(rec.pid) !== rec.start) return undefined;
  return b.owns(command(rec.pid)) ? rec : undefined;
}

/** true if the backend responds 2xx on its readiness path; false if not listening or 503 (still loading). Other errors are thrown rather than taking over the port. */
async function reachable(b: Backend): Promise<boolean> {
  try {
    const r = await fetch(`${b.base}${b.healthPath}`, { signal: AbortSignal.timeout(1500) });
    if (r.status === 503) return false;
    return r.ok;
  } catch (e: any) {
    if (e?.cause?.code === "ECONNREFUSED") return false;
    throw new Error(`port ${new URL(b.base).port} is taken by a process that does not respond: ${e?.cause?.code ?? e?.message ?? e}`);
  }
}

function leaseAlive(pid: number): boolean {
  try { return startTime(pid) === readFileSync(join(LEASES, String(pid)), "utf8"); } catch { return false; }
}

function pruneLeases(): number[] {
  let pids: number[] = [];
  try { pids = readdirSync(LEASES).map(Number).filter((n) => n > 0); } catch { return []; }
  const alive: number[] = [];
  for (const pid of pids) {
    if (leaseAlive(pid)) alive.push(pid);
    else try { unlinkSync(join(LEASES, String(pid))); } catch {}
  }
  return alive;
}

function takeLease() {
  const me = startTime(process.pid);
  if (!me) return;
  mkdirSync(LEASES, { recursive: true });
  writeFileSync(join(LEASES, String(process.pid)), me);
}

function dropLease(): boolean {
  try { unlinkSync(join(LEASES, String(process.pid))); return true; } catch { return false; }
}

/** Start the backend if none is running. When several pi send their first request at once, an mkdir lock serializes them; those that lose the lock wait for the winner. */
async function ensure(b: Backend, signal?: AbortSignal) {
  if (await reachable(b)) return;
  mkdirSync(RUN, { recursive: true });
  const lock = startLock(b);
  const deadline = Date.now() + START_TIMEOUT_MS;
  for (;;) {
    try { mkdirSync(lock); break; } catch (e: any) {
      if (e.code !== "EEXIST") throw e;
      // If the lock holder crashed it leaves the directory behind: treat it as stale after the start timeout
      try { if (Date.now() - statSync(lock).mtimeMs > START_TIMEOUT_MS) rmdirSync(lock); } catch {}
      if (await reachable(b)) return;
      if (Date.now() > deadline) throw new Error(`timed out waiting for another pi to start ${b.provider}, see ${b.log}`);
      await delay(200, undefined, { signal });
    }
  }
  try {
    if (await reachable(b)) return;
    const log = openSync(b.log, "a");
    const child = spawn(b.launcher, [], { detached: true, stdio: ["ignore", log, log] });
    closeSync(log);
    await new Promise<void>((ok, fail) => { child.once("spawn", ok); child.once("error", fail); });
    const start = startTime(child.pid!);
    if (start) writeFileSync(recordFile(b), JSON.stringify({ pid: child.pid, start } satisfies ProcRecord));
    child.unref();
    while (Date.now() < deadline) {
      if (child.exitCode !== null || child.signalCode) throw new Error(`${b.provider} exited right after starting, see ${b.log}`);
      if (await reachable(b)) return;
      await delay(200, undefined, { signal });
    }
    throw new Error(`${b.provider} did not come up in time, see ${b.log}`);
  } finally {
    try { rmdirSync(lock); } catch {}
  }
}

/** Stop the backend this extension started; any of its children still alive after it exits are ended too. */
async function stop(b: Backend) {
  const rec = owned(b);
  if (!rec) return;
  const kids = children(rec.pid);
  try { process.kill(rec.pid, "SIGTERM"); } catch {}
  const deadline = Date.now() + STOP_TIMEOUT_MS;
  const alive = (pid: number) => { try { process.kill(pid, 0); return true; } catch { return false; } };
  while (Date.now() < deadline && (alive(rec.pid) || kids.some(alive))) {
    if (!alive(rec.pid)) for (const k of kids) if (alive(k)) try { process.kill(k, "SIGTERM"); } catch {}
    await delay(100);
  }
  try { unlinkSync(recordFile(b)); } catch {}
}

type ServedModel = { id: string; status?: { value?: string } };

async function modelStatus(b: Backend, id: string): Promise<string | undefined> {
  try {
    const r = await fetch(`${b.base}/models`, { signal: AbortSignal.timeout(1500) });
    const body = (await r.json()) as { data?: ServedModel[] };
    return body.data?.find((m) => m.id === id)?.status?.value;
  } catch { return undefined; }
}

export default function localLlm(pi: ExtensionAPI) {
  const setStatus = (ctx: ExtensionContext, text?: string) => { if (ctx.hasUI) ctx.ui.setStatus(STATUS_KEY, text); };

  pi.on("session_start", () => { pruneLeases(); });

  pi.on("before_provider_request", async (_event, ctx) => {
    const model = ctx.model;
    const b = BACKENDS.find((x) => x.provider === model?.provider);
    if (!model || !b) return;
    takeLease();
    try {
      for (const other of BACKENDS) if (other !== b && owned(other)) {
        setStatus(ctx, `stopping ${other.provider} to free memory…`);
        await stop(other);
      }
      await ensure(b, ctx.signal);
    } catch (e) {
      setStatus(ctx);
      if (ctx.hasUI) ctx.ui.notify(`local model: ${(e as Error).message}`, "error");
      throw e;
    }
    const st = await modelStatus(b, model.id);
    setStatus(ctx, st && st !== "loaded" ? `${model.id} loading…` : undefined);
  });

  pi.on("message_start", (event, ctx) => { if (event.message.role === "assistant") setStatus(ctx); });
  pi.on("agent_end", (_event, ctx) => setStatus(ctx));

  pi.on("session_shutdown", async (event) => {
    // reload/new/resume/fork only switch sessions; the process stays and keeps its lease
    if (event.reason !== "quit") return;
    const hadLease = dropLease();
    if (!hadLease) return;
    if (pruneLeases().length > 0) return;
    for (const b of BACKENDS) await stop(b);
  });
}
