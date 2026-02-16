// QMD Daemon - Persistent server for CLI and MCP
import { resolve } from "node:path";
import { homedir } from "node:os";
import { existsSync, readFileSync, writeFileSync, unlinkSync, mkdirSync, openSync, closeSync } from "node:fs";
import { createStore } from "./store.js";
import { getDefaultLlamaCpp, disposeDefaultLlamaCpp, withLLMSession } from "./llm.js";
import { createMcpServer } from "./mcp.js";
import type { Server as McpServer } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type { Store } from "./store.js";

const startTime = Date.now();
let requestCount = 0;
let lastActivityTime = Date.now();

// Daemon paths
export function getDaemonPaths() {
  const cacheDir = Bun.env.XDG_CACHE_HOME
    ? resolve(Bun.env.XDG_CACHE_HOME, "qmd")
    : resolve(homedir(), ".cache", "qmd");

  return {
    socket: resolve(cacheDir, "daemon.sock"),
    pid: resolve(cacheDir, "daemon.pid"),
    log: resolve(cacheDir, "daemon.log"),
    cacheDir,
  };
}

// RPC handler types
type RpcRequest = {
  method: string;
  params?: Record<string, any>;
};

type RpcResponse = {
  result?: any;
  error?: { code: number; message: string; data?: any };
};

type ProgressCallback = (progress: any) => void;

// Handler context
type HandlerContext = {
  store: Store;
  onProgress?: ProgressCallback;
};

// Import handler implementations
async function handleSearch(params: any, ctx: HandlerContext) {
  const { searchFTS } = await import("./store.js");
  const { query, limit = 20, collection, minScore = 0 } = params;

  // Get collection ID if specified
  let collectionId: number | undefined;
  if (collection) {
    const coll = ctx.store.getCollectionByName(collection);
    if (!coll) throw new Error(`Collection not found: ${collection}`);
    const collRow = ctx.store.db.prepare(`SELECT id FROM documents WHERE collection = ? LIMIT 1`).get(collection) as { id: number } | undefined;
    collectionId = collRow?.id;
  }

  const results = searchFTS(ctx.store.db, query, limit, collectionId);
  return results.filter(r => r.score >= minScore);
}

async function handleVsearch(params: any, ctx: HandlerContext) {
  const { query, limit = 20, collection, minScore = 0.3 } = params;

  return await withLLMSession(async (session) => {
    const { searchVec } = await import("./store.js");
    const results = await searchVec(ctx.store.db, query, limit, collection, session);
    return results.filter(r => r.score >= minScore);
  });
}

async function handleQuery(params: any, ctx: HandlerContext) {
  const { query, limit = 20, collection, minScore = 0 } = params;

  return await withLLMSession(async (session) => {
    const { hybridQuery } = await import("./store.js");
    const results = await hybridQuery(ctx.store, query, { limit, collection, session });
    return results.filter(r => r.score >= minScore);
  });
}

async function handleGet(params: any, ctx: HandlerContext) {
  const { file, fromLine, maxLines, lineNumbers } = params;
  const doc = ctx.store.findDocument(file, { includeBody: true });

  if (!doc || "error" in doc) {
    // Try to find similar files
    const similar = ctx.store.findSimilarFiles(file, 3, 5);
    throw new Error(`Document not found: ${file}${similar.length > 0 ? `\n\nDid you mean:\n${similar.map(s => `  - ${s}`).join("\n")}` : ""}`);
  }

  let body = doc.body || "";

  // Apply line filtering
  if (fromLine !== undefined || maxLines !== undefined) {
    const lines = body.split("\n");
    const start = fromLine ? parseInt(fromLine) - 1 : 0;
    const end = maxLines ? start + parseInt(maxLines) : lines.length;
    body = lines.slice(start, end).join("\n");
  }

  // Add line numbers if requested
  if (lineNumbers) {
    const lines = body.split("\n");
    const startNum = fromLine ? parseInt(fromLine) : 1;
    body = lines.map((line, i) => `${startNum + i}  ${line}`).join("\n");
  }

  return { ...doc, body };
}

async function handleMultiGet(params: any, ctx: HandlerContext) {
  const { pattern, maxLines, maxBytes = 10240, lineNumbers } = params;
  const result = ctx.store.findDocuments(pattern, { includeBody: true, maxBytes: parseInt(maxBytes) });

  // Apply line limiting and line numbers
  const processedDocs = result.docs.map(doc => {
    let body = doc.body || "";

    if (maxLines) {
      const lines = body.split("\n");
      body = lines.slice(0, parseInt(maxLines)).join("\n");
    }

    if (lineNumbers) {
      const lines = body.split("\n");
      body = lines.map((line, i) => `${i + 1}  ${line}`).join("\n");
    }

    return { ...doc, body };
  });

  return { docs: processedDocs, errors: result.errors };
}

async function handleLs(params: any, ctx: HandlerContext) {
  const { prefix } = params;
  const { listCollections } = await import("./collections.js");
  const { parseVirtualPath, isVirtualPath } = await import("./store.js");

  if (!prefix) {
    // List all collections
    const collections = listCollections();
    return { type: "collections", collections };
  }

  // Parse virtual path
  const vpath = isVirtualPath(prefix) ? parseVirtualPath(prefix) : null;

  if (vpath) {
    // List files in collection with path prefix
    const files = ctx.store.db.prepare(`
      SELECT filepath, title FROM documents
      WHERE collection = ? AND active = 1 AND filepath LIKE ?
      ORDER BY filepath
    `).all(vpath.collectionName, vpath.path + "%") as { filepath: string; title: string }[];

    return { type: "files", files };
  }

  return { type: "error", error: "Invalid path" };
}

async function handleStatus(params: any, ctx: HandlerContext) {
  const status = ctx.store.getStatus();

  // Add daemon info
  const uptime = Math.floor((Date.now() - startTime) / 1000);
  const { getLoadedModelInfo } = await import("./llm.js");
  const loadedModels = getLoadedModelInfo();

  return {
    ...status,
    daemon: {
      pid: process.pid,
      uptime,
      startTime: new Date(startTime).toISOString(),
      requestCount,
      lastActivity: new Date(lastActivityTime).toISOString(),
      loadedModels,
    },
  };
}

async function handleEmbed(params: any, ctx: HandlerContext) {
  const { force } = params;
  const { vectorIndex } = await import("./qmd.js");

  // This will need refactoring to support progress callbacks
  // For now, just call the existing function
  await vectorIndex(force, ctx.onProgress);

  return { success: true };
}

async function handleUpdate(params: any, ctx: HandlerContext) {
  const { pull, refresh } = params;
  // TODO: Implement update with progress
  return { success: true };
}

async function handleCleanup(params: any, ctx: HandlerContext) {
  const cacheCount = ctx.store.deleteLLMCache();
  const orphanedVecs = ctx.store.cleanupOrphanedVectors();
  const inactiveDocs = ctx.store.deleteInactiveDocuments();

  ctx.store.vacuumDatabase();

  return { cacheCount, orphanedVecs, inactiveDocs };
}

async function handleCollectionList(params: any, ctx: HandlerContext) {
  const { listCollections } = await import("./collections.js");
  return listCollections();
}

async function handleCollectionAdd(params: any, ctx: HandlerContext) {
  const { path, name, mask } = params;
  const { addCollection } = await import("./collections.js");
  addCollection(name, path, mask);
  return { success: true };
}

async function handleCollectionRemove(params: any, ctx: HandlerContext) {
  const { name } = params;
  const { removeCollection } = await import("./collections.js");
  const removed = removeCollection(name);
  if (!removed) throw new Error(`Collection not found: ${name}`);
  return { success: true };
}

async function handleCollectionRename(params: any, ctx: HandlerContext) {
  const { oldName, newName } = params;
  const { renameCollection } = await import("./collections.js");
  const renamed = renameCollection(oldName, newName);
  if (!renamed) throw new Error(`Collection not found: ${oldName}`);
  return { success: true };
}

async function handleContextAdd(params: any, ctx: HandlerContext) {
  const { path, text } = params;
  const { addContext } = await import("./collections.js");
  addContext(path || "/", text);
  return { success: true };
}

async function handleContextList(params: any, ctx: HandlerContext) {
  const { listAllContexts } = await import("./collections.js");
  return listAllContexts();
}

async function handleContextRm(params: any, ctx: HandlerContext) {
  const { path } = params;
  const { removeContext } = await import("./collections.js");
  removeContext(path);
  return { success: true };
}

async function handleContextCheck(params: any, ctx: HandlerContext) {
  const collectionsWithout = ctx.store.getCollectionsWithoutContext();
  // TODO: Also get paths without context
  return { collectionsWithout };
}

async function handleProviderGet(params: any, ctx: HandlerContext) {
  const provider = ctx.store.getSetting("embedding_provider") || "local";
  const model = ctx.store.getSetting("embedding_model") || "";
  return { provider, model };
}

async function handleProviderSet(params: any, ctx: HandlerContext) {
  const { provider, model } = params;
  ctx.store.setSetting("embedding_provider", provider);
  if (model) ctx.store.setSetting("embedding_model", model);
  return { success: true };
}

async function handlePing(params: any, ctx: HandlerContext) {
  const uptime = Math.floor((Date.now() - startTime) / 1000);
  const { getLoadedModelInfo } = await import("./llm.js");
  const loadedModels = getLoadedModelInfo();

  return {
    pid: process.pid,
    uptime,
    loadedModels,
  };
}

// Main RPC dispatcher
async function handleRpc(req: RpcRequest, ctx: HandlerContext): Promise<RpcResponse> {
  try {
    lastActivityTime = Date.now();
    requestCount++;

    let result: any;

    switch (req.method) {
      case "search": result = await handleSearch(req.params || {}, ctx); break;
      case "vsearch": result = await handleVsearch(req.params || {}, ctx); break;
      case "query": result = await handleQuery(req.params || {}, ctx); break;
      case "get": result = await handleGet(req.params || {}, ctx); break;
      case "multi_get": result = await handleMultiGet(req.params || {}, ctx); break;
      case "ls": result = await handleLs(req.params || {}, ctx); break;
      case "status": result = await handleStatus(req.params || {}, ctx); break;
      case "embed": result = await handleEmbed(req.params || {}, ctx); break;
      case "update": result = await handleUpdate(req.params || {}, ctx); break;
      case "cleanup": result = await handleCleanup(req.params || {}, ctx); break;
      case "collection_list": result = await handleCollectionList(req.params || {}, ctx); break;
      case "collection_add": result = await handleCollectionAdd(req.params || {}, ctx); break;
      case "collection_remove": result = await handleCollectionRemove(req.params || {}, ctx); break;
      case "collection_rename": result = await handleCollectionRename(req.params || {}, ctx); break;
      case "context_add": result = await handleContextAdd(req.params || {}, ctx); break;
      case "context_list": result = await handleContextList(req.params || {}, ctx); break;
      case "context_rm": result = await handleContextRm(req.params || {}, ctx); break;
      case "context_check": result = await handleContextCheck(req.params || {}, ctx); break;
      case "provider_get": result = await handleProviderGet(req.params || {}, ctx); break;
      case "provider_set": result = await handleProviderSet(req.params || {}, ctx); break;
      case "ping": result = await handlePing(req.params || {}, ctx); break;
      default:
        return { error: { code: -32601, message: `Method not found: ${req.method}` } };
    }

    return { result };
  } catch (error: any) {
    return {
      error: {
        code: -32000,
        message: error.message || "Internal error",
        data: error.stack,
      },
    };
  }
}

// Start daemon server
export async function startDaemon(options?: { quiet?: boolean }): Promise<void> {
  const paths = getDaemonPaths();
  const quiet = options?.quiet ?? false;

  // Ensure cache directory exists
  mkdirSync(paths.cacheDir, { recursive: true });

  // Check if daemon already running
  if (existsSync(paths.socket)) {
    try {
      // Try to connect to existing socket
      const testRes = await fetch("http://localhost/health", { unix: paths.socket });
      if (testRes.ok) {
        const data = await testRes.json() as any;
        console.error(`Daemon already running (PID ${data.pid})`);
        console.error(`Run 'qmd daemon stop' to stop it first.`);
        process.exit(1);
      }
    } catch {
      // Socket exists but not accepting connections - stale socket
      if (!quiet) console.log("Removing stale socket file...");
      unlinkSync(paths.socket);
    }
  }

  // Enable production mode and create store
  const { enableProductionMode } = await import("./store.js");
  enableProductionMode();
  const store = createStore();

  // Create MCP server
  const mcpServer = createMcpServer(store);

  // Start Unix socket server
  const server = Bun.serve({
    unix: paths.socket,
    async fetch(req) {
      const url = new URL(req.url);

      // Health endpoint
      if (url.pathname === "/health") {
        const uptime = Math.floor((Date.now() - startTime) / 1000);
        const { getLoadedModelInfo } = await import("./llm.js");
        const loadedModels = getLoadedModelInfo();

        return Response.json({
          status: "ok",
          pid: process.pid,
          uptime,
          loadedModels,
          requestCount,
        });
      }

      // RPC endpoint
      if (url.pathname === "/rpc" && req.method === "POST") {
        const rpcReq = await req.json() as RpcRequest;
        const rpcRes = await handleRpc(rpcReq, { store });
        return Response.json(rpcRes);
      }

      // MCP endpoints - delegate to MCP server
      if (url.pathname.startsWith("/mcp")) {
        // TODO: Integrate MCP HTTP transport here
        // For now, MCP will be served separately via stdio or HTTP
        return new Response("MCP endpoint - not yet integrated", { status: 501 });
      }

      return new Response("Not found", { status: 404 });
    },
  });

  // Write PID file
  writeFileSync(paths.pid, String(process.pid));

  if (!quiet) {
    console.log(`QMD daemon started (PID ${process.pid})`);
    console.log(`Socket: ${paths.socket}`);
  }

  // Signal handling
  const cleanup = async () => {
    if (!quiet) console.log("\nShutting down daemon...");

    try {
      server.stop();
      if (existsSync(paths.socket)) unlinkSync(paths.socket);
      if (existsSync(paths.pid)) unlinkSync(paths.pid);
      store.close();
      await disposeDefaultLlamaCpp();
    } catch (e) {
      console.error("Error during cleanup:", e);
    }

    process.exit(0);
  };

  process.on("SIGTERM", cleanup);
  process.on("SIGINT", cleanup);

  // Keep process alive
  await new Promise(() => {});
}

// Stop running daemon
export async function stopDaemon(): Promise<boolean> {
  const paths = getDaemonPaths();

  if (!existsSync(paths.pid)) {
    console.log("Daemon not running (no PID file)");
    return false;
  }

  const pid = parseInt(readFileSync(paths.pid, "utf-8").trim());

  try {
    // Check if process is alive
    process.kill(pid, 0);

    // Send SIGTERM
    process.kill(pid, "SIGTERM");

    // Wait a bit for graceful shutdown
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Clean up PID file if still exists
    if (existsSync(paths.pid)) unlinkSync(paths.pid);

    console.log(`Stopped daemon (PID ${pid})`);
    return true;
  } catch {
    // Process not running
    if (existsSync(paths.pid)) unlinkSync(paths.pid);
    if (existsSync(paths.socket)) unlinkSync(paths.socket);
    console.log("Cleaned up stale PID file (daemon was not running)");
    return false;
  }
}

// Run as standalone daemon if executed directly
if (import.meta.main) {
  const quiet = Bun.argv.includes("--quiet");
  await startDaemon({ quiet });
}
