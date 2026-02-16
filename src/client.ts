// QMD Daemon Client - Connect CLI to daemon
import { getDaemonPaths } from "./daemon.js";
import { openSync, closeSync } from "node:fs";
import { resolve } from "node:path";

export type ProgressInfo = {
  phase?: string;
  current?: number;
  total?: number;
  message?: string;
};

export type ClientOptions = {
  onProgress?: (progress: ProgressInfo) => void;
  timeout?: number; // ms
};

export class DaemonClient {
  private socketPath: string;
  private connected: boolean = false;

  constructor(socketPath?: string) {
    this.socketPath = socketPath || getDaemonPaths().socket;
  }

  async connect(): Promise<void> {
    try {
      const res = await fetch("http://localhost/health", {
        unix: this.socketPath,
      });

      if (res.ok) {
        this.connected = true;
        return;
      }

      throw new Error(`Health check failed: ${res.status}`);
    } catch (error: any) {
      this.connected = false;
      throw new Error(`Cannot connect to daemon: ${error.message}`);
    }
  }

  isConnected(): boolean {
    return this.connected;
  }

  async request<T>(
    method: string,
    params?: Record<string, any>,
    options?: ClientOptions
  ): Promise<T> {
    if (!this.connected) {
      throw new Error("Not connected to daemon. Call connect() first.");
    }

    try {
      const res = await fetch("http://localhost/rpc", {
        method: "POST",
        unix: this.socketPath,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ method, params }),
      });

      const data = await res.json() as any;

      if (data.error) {
        throw new Error(data.error.message || "RPC error");
      }

      return data.result as T;
    } catch (error: any) {
      throw new Error(`RPC request failed: ${error.message}`);
    }
  }

  async close(): Promise<void> {
    this.connected = false;
  }
}

// Ensure daemon is running, auto-start if needed
export async function ensureDaemon(): Promise<DaemonClient> {
  const paths = getDaemonPaths();
  const client = new DaemonClient(paths.socket);

  // Try to connect
  try {
    await client.connect();
    return client;
  } catch {
    // Daemon not running, start it
    console.log("Starting daemon...");

    // Spawn daemon as background process (quiet mode)
    const logFd = openSync(paths.log, "a");

    const child = Bun.spawn(
      [process.execPath, resolve(import.meta.dir, "daemon.ts"), "--quiet"],
      {
        stdout: logFd,
        stderr: logFd,
        stdin: "ignore",
      }
    );

    child.unref();
    closeSync(logFd);

    // Wait for daemon to be ready (poll socket)
    const maxWaitMs = 15000;
    const pollIntervalMs = 100;
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitMs) {
      try {
        await client.connect();
        console.log("Daemon started successfully");
        return client;
      } catch {
        await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
      }
    }

    throw new Error("Daemon failed to start within 15 seconds");
  }
}

// Get daemon status without starting it
export async function getDaemonStatus(): Promise<any | null> {
  const paths = getDaemonPaths();

  try {
    const res = await fetch("http://localhost/health", {
      unix: paths.socket,
    });

    if (res.ok) {
      return await res.json();
    }

    return null;
  } catch {
    return null;
  }
}
