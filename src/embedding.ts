/**
 * embedding.ts - Embedding provider abstraction layer
 *
 * Supports local (node-llama-cpp), OpenAI, and Gemini embedding providers.
 */

import { getDefaultLlamaCpp, formatQueryForEmbedding, formatDocForEmbedding, type EmbeddingResult, type ILLMSession } from "./llm";
import type { Database } from "bun:sqlite";

// =============================================================================
// Provider Interface
// =============================================================================

export interface EmbeddingProvider {
  readonly name: string;
  readonly modelId: string;
  readonly dimensions: number | null;  // null = auto-detect from first embed
  readonly hasTokenizer: boolean;      // whether provider has local tokenizer

  formatQuery(query: string): string;
  formatDocument(text: string, title?: string): string;

  embed(text: string, isQuery?: boolean): Promise<EmbeddingResult | null>;
  embedBatch(texts: string[], isQuery?: boolean): Promise<(EmbeddingResult | null)[]>;

  dispose(): Promise<void>;
}

// =============================================================================
// Local Provider (node-llama-cpp)
// =============================================================================

export class LocalEmbeddingProvider implements EmbeddingProvider {
  readonly name = "local";
  readonly modelId = "embeddinggemma";
  readonly dimensions = 768;
  readonly hasTokenizer = true;

  formatQuery(query: string): string {
    return formatQueryForEmbedding(query);
  }

  formatDocument(text: string, title?: string): string {
    return formatDocForEmbedding(text, title);
  }

  async embed(text: string, isQuery?: boolean): Promise<EmbeddingResult | null> {
    const llm = getDefaultLlamaCpp();
    return await llm.embed(text, { isQuery });
  }

  async embedBatch(texts: string[], isQuery?: boolean): Promise<(EmbeddingResult | null)[]> {
    // Local provider uses withLLMSession for batch operations
    // Individual embeds are fine through getDefaultLlamaCpp()
    const results: (EmbeddingResult | null)[] = [];
    for (const text of texts) {
      const result = await this.embed(text, isQuery);
      results.push(result);
    }
    return results;
  }

  async dispose(): Promise<void> {
    // Dispose is handled by global disposeDefaultLlamaCpp()
  }
}

// =============================================================================
// OpenAI Provider
// =============================================================================

const OPENAI_DIMENSIONS: Record<string, number> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  "text-embedding-ada-002": 1536,
};

export class OpenAIEmbeddingProvider implements EmbeddingProvider {
  readonly name = "openai";
  readonly modelId: string;
  readonly hasTokenizer = false;
  dimensions: number | null = null;

  private apiKey: string;
  private baseUrl: string;

  constructor(modelId?: string) {
    this.modelId = modelId || "text-embedding-3-small";
    this.apiKey = process.env.OPENAI_API_KEY || "";
    this.baseUrl = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";

    if (!this.apiKey) {
      throw new Error("OpenAI API key required. Set OPENAI_API_KEY environment variable.");
    }

    // Set known dimensions if available
    this.dimensions = OPENAI_DIMENSIONS[this.modelId] || null;
  }

  formatQuery(query: string): string {
    return query;  // No special formatting needed for OpenAI
  }

  formatDocument(text: string, _title?: string): string {
    return text;  // No special formatting needed for OpenAI
  }

  async embed(text: string, _isQuery?: boolean): Promise<EmbeddingResult | null> {
    const results = await this.embedBatch([text], _isQuery);
    return results[0] || null;
  }

  async embedBatch(texts: string[], _isQuery?: boolean): Promise<(EmbeddingResult | null)[]> {
    // OpenAI supports up to 2048 inputs per request, we'll use 100 to be safe
    const BATCH_SIZE = 100;
    const allResults: (EmbeddingResult | null)[] = [];

    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);
      const batchResults = await this.embedBatchRequest(batch);
      allResults.push(...batchResults);
    }

    return allResults;
  }

  private async embedBatchRequest(texts: string[], retries = 3): Promise<(EmbeddingResult | null)[]> {
    try {
      const response = await fetch(`${this.baseUrl}/embeddings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model: this.modelId,
          input: texts,
        }),
      });

      if (!response.ok) {
        // Handle rate limiting with retry
        if (response.status === 429 && retries > 0) {
          const retryAfter = response.headers.get("retry-after");
          const waitMs = retryAfter ? parseInt(retryAfter) * 1000 : 1000;
          await new Promise(resolve => setTimeout(resolve, waitMs));
          return this.embedBatchRequest(texts, retries - 1);
        }

        const errorText = await response.text();
        throw new Error(`OpenAI API error (${response.status}): ${errorText}`);
      }

      const data = await response.json() as {
        data: Array<{ embedding: number[]; index: number }>;
        model: string;
        usage: { prompt_tokens: number; total_tokens: number };
      };

      // Auto-detect dimensions from first embedding
      if (this.dimensions === null && data.data.length > 0) {
        this.dimensions = data.data[0]!.embedding.length;
      }

      // Sort by index to maintain order
      const sorted = data.data.sort((a, b) => a.index - b.index);
      return sorted.map(item => ({
        embedding: item.embedding,
        model: this.modelId,
      }));
    } catch (error) {
      console.error(`OpenAI embedding error:`, error);
      return texts.map(() => null);
    }
  }

  async dispose(): Promise<void> {
    // No cleanup needed for stateless HTTP client
  }
}

// =============================================================================
// Gemini Provider
// =============================================================================

const GEMINI_DIMENSIONS: Record<string, number> = {
  "text-embedding-004": 768,
  "embedding-001": 768,
};

export class GeminiEmbeddingProvider implements EmbeddingProvider {
  readonly name = "gemini";
  readonly modelId: string;
  readonly hasTokenizer = false;
  dimensions: number | null = null;

  private apiKey: string;

  constructor(modelId?: string) {
    this.modelId = modelId || "text-embedding-004";
    this.apiKey = process.env.GEMINI_API_KEY || "";

    if (!this.apiKey) {
      throw new Error("Gemini API key required. Set GEMINI_API_KEY environment variable.");
    }

    // Set known dimensions if available
    this.dimensions = GEMINI_DIMENSIONS[this.modelId] || null;
  }

  formatQuery(query: string): string {
    return query;  // No special formatting needed for Gemini
  }

  formatDocument(text: string, _title?: string): string {
    return text;  // No special formatting needed for Gemini
  }

  async embed(text: string, isQuery?: boolean): Promise<EmbeddingResult | null> {
    const results = await this.embedBatch([text], isQuery);
    return results[0] || null;
  }

  async embedBatch(texts: string[], isQuery?: boolean): Promise<(EmbeddingResult | null)[]> {
    // Gemini supports up to 100 inputs per request
    const BATCH_SIZE = 100;
    const allResults: (EmbeddingResult | null)[] = [];

    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);
      const batchResults = await this.embedBatchRequest(batch, isQuery);
      allResults.push(...batchResults);
    }

    return allResults;
  }

  private async embedBatchRequest(texts: string[], isQuery?: boolean, retries = 3): Promise<(EmbeddingResult | null)[]> {
    try {
      const taskType = isQuery ? "RETRIEVAL_QUERY" : "RETRIEVAL_DOCUMENT";
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${this.modelId}:batchEmbedContents?key=${this.apiKey}`;

      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          requests: texts.map(text => ({
            model: `models/${this.modelId}`,
            content: { parts: [{ text }] },
            taskType,
          })),
        }),
      });

      if (!response.ok) {
        // Handle rate limiting with retry
        if (response.status === 429 && retries > 0) {
          await new Promise(resolve => setTimeout(resolve, 1000));
          return this.embedBatchRequest(texts, isQuery, retries - 1);
        }

        const errorText = await response.text();
        throw new Error(`Gemini API error (${response.status}): ${errorText}`);
      }

      const data = await response.json() as {
        embeddings: Array<{ values: number[] }>;
      };

      // Auto-detect dimensions from first embedding
      if (this.dimensions === null && data.embeddings.length > 0) {
        this.dimensions = data.embeddings[0]!.values.length;
      }

      return data.embeddings.map(item => ({
        embedding: item.values,
        model: this.modelId,
      }));
    } catch (error) {
      console.error(`Gemini embedding error:`, error);
      return texts.map(() => null);
    }
  }

  async dispose(): Promise<void> {
    // No cleanup needed for stateless HTTP client
  }
}

// =============================================================================
// Provider Factory and Singleton
// =============================================================================

let currentProvider: EmbeddingProvider | null = null;

/**
 * Create an embedding provider from database settings or CLI override
 *
 * @param db - Database instance
 * @param overrideProvider - Optional CLI override for provider name (e.g., "local", "openai", "gemini")
 * @param overrideModel - Optional CLI override for model ID
 */
export async function createEmbeddingProvider(
  db: Database,
  overrideProvider?: string,
  overrideModel?: string
): Promise<EmbeddingProvider> {
  const { getSetting } = await import("./store.js");

  // Use override if provided, otherwise read from DB settings
  const provider = overrideProvider || getSetting(db, "embedding_provider") || "local";
  const model = overrideModel || getSetting(db, "embedding_model");

  switch (provider) {
    case "local":
      return new LocalEmbeddingProvider();
    case "openai":
      return new OpenAIEmbeddingProvider(model || undefined);
    case "gemini":
      return new GeminiEmbeddingProvider(model || undefined);
    default:
      throw new Error(`Unknown embedding provider: ${provider}`);
  }
}

/**
 * Get the current embedding provider singleton
 * Creates one from DB settings if not already set
 *
 * @param db - Database instance
 * @param overrideProvider - Optional CLI override for provider name
 * @param overrideModel - Optional CLI override for model ID
 */
export async function getEmbeddingProvider(
  db: Database,
  overrideProvider?: string,
  overrideModel?: string
): Promise<EmbeddingProvider> {
  // If override is provided, always create fresh provider (don't use singleton)
  if (overrideProvider) {
    return await createEmbeddingProvider(db, overrideProvider, overrideModel);
  }

  // Otherwise use singleton pattern
  if (!currentProvider) {
    currentProvider = await createEmbeddingProvider(db);
  }
  return currentProvider;
}

/**
 * Set the current embedding provider
 * Pass null to reset (will recreate from DB on next getEmbeddingProvider call)
 */
export function setEmbeddingProvider(provider: EmbeddingProvider | null): void {
  currentProvider = provider;
}
