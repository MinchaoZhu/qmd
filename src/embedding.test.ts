/**
 * Tests for embedding providers
 */

import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import {
  createEmbeddingProvider,
  getEmbeddingProvider,
  setEmbeddingProvider,
  LocalEmbeddingProvider,
  OpenAIEmbeddingProvider,
  GeminiEmbeddingProvider,
  type EmbeddingProvider,
} from "./embedding";
import type { EmbeddingConfig } from "./collections";

describe("Embedding Provider Factory", () => {
  afterAll(() => {
    // Reset singleton after tests
    setEmbeddingProvider(null);
  });

  test("creates local provider by default", () => {
    const provider = createEmbeddingProvider({ provider: "local" });
    expect(provider.name).toBe("local");
    expect(provider.modelId).toBe("embeddinggemma");
    expect(provider.hasTokenizer).toBe(true);
    expect(provider.dimensions).toBe(768);
  });

  test("creates OpenAI provider with config", () => {
    const config: EmbeddingConfig = {
      provider: "openai",
      model: "text-embedding-3-small",
      api_key: "test-key",
    };
    const provider = createEmbeddingProvider(config);
    expect(provider.name).toBe("openai");
    expect(provider.modelId).toBe("text-embedding-3-small");
    expect(provider.hasTokenizer).toBe(false);
    expect(provider.dimensions).toBe(1536);
  });

  test("creates Gemini provider with config", () => {
    const config: EmbeddingConfig = {
      provider: "gemini",
      model: "text-embedding-004",
      api_key: "test-key",
    };
    const provider = createEmbeddingProvider(config);
    expect(provider.name).toBe("gemini");
    expect(provider.modelId).toBe("text-embedding-004");
    expect(provider.hasTokenizer).toBe(false);
    expect(provider.dimensions).toBe(768);
  });

  test("throws error for unknown provider", () => {
    expect(() => {
      createEmbeddingProvider({ provider: "unknown" as any });
    }).toThrow("Unknown embedding provider");
  });

  test("singleton returns same instance", () => {
    setEmbeddingProvider(null); // Reset
    const provider1 = getEmbeddingProvider();
    const provider2 = getEmbeddingProvider();
    expect(provider1).toBe(provider2);
  });

  test("setEmbeddingProvider overrides singleton", () => {
    const customProvider = createEmbeddingProvider({
      provider: "openai",
      api_key: "test-key",
    });
    setEmbeddingProvider(customProvider);
    const provider = getEmbeddingProvider();
    expect(provider.name).toBe("openai");
    setEmbeddingProvider(null); // Reset
  });
});

describe("LocalEmbeddingProvider", () => {
  test("formats query with nomic prefix", () => {
    const provider = new LocalEmbeddingProvider();
    const formatted = provider.formatQuery("test query");
    expect(formatted).toBe("task: search result | query: test query");
  });

  test("formats document with nomic prefix", () => {
    const provider = new LocalEmbeddingProvider();
    const formatted = provider.formatDocument("test content", "Test Title");
    expect(formatted).toBe("title: Test Title | text: test content");
  });

  test("formats document without title", () => {
    const provider = new LocalEmbeddingProvider();
    const formatted = provider.formatDocument("test content");
    expect(formatted).toBe("title: none | text: test content");
  });
});

describe("OpenAIEmbeddingProvider", () => {
  test("uses passthrough formatting", () => {
    const provider = new OpenAIEmbeddingProvider({
      provider: "openai",
      api_key: "test-key",
    });
    expect(provider.formatQuery("test query")).toBe("test query");
    expect(provider.formatDocument("test content", "title")).toBe("test content");
  });

  test("throws error without API key", () => {
    const originalKey = process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_API_KEY;

    expect(() => {
      new OpenAIEmbeddingProvider({
        provider: "openai",
      });
    }).toThrow("OpenAI API key required");

    if (originalKey) process.env.OPENAI_API_KEY = originalKey;
  });

  test("uses environment variable for API key", () => {
    const originalKey = process.env.OPENAI_API_KEY;
    process.env.OPENAI_API_KEY = "env-test-key";

    const provider = new OpenAIEmbeddingProvider({
      provider: "openai",
    });
    expect(provider.name).toBe("openai");

    if (originalKey) {
      process.env.OPENAI_API_KEY = originalKey;
    } else {
      delete process.env.OPENAI_API_KEY;
    }
  });

  test("uses default model", () => {
    const provider = new OpenAIEmbeddingProvider({
      provider: "openai",
      api_key: "test-key",
    });
    expect(provider.modelId).toBe("text-embedding-3-small");
  });

  test("uses custom model", () => {
    const provider = new OpenAIEmbeddingProvider({
      provider: "openai",
      model: "text-embedding-3-large",
      api_key: "test-key",
    });
    expect(provider.modelId).toBe("text-embedding-3-large");
    expect(provider.dimensions).toBe(3072);
  });

  test("uses custom base_url", () => {
    const provider = new OpenAIEmbeddingProvider({
      provider: "openai",
      api_key: "test-key",
      base_url: "https://custom.api.com/v1",
    });
    expect(provider.name).toBe("openai");
  });
});

describe("GeminiEmbeddingProvider", () => {
  test("uses passthrough formatting", () => {
    const provider = new GeminiEmbeddingProvider({
      provider: "gemini",
      api_key: "test-key",
    });
    expect(provider.formatQuery("test query")).toBe("test query");
    expect(provider.formatDocument("test content", "title")).toBe("test content");
  });

  test("throws error without API key", () => {
    const originalKey = process.env.GEMINI_API_KEY;
    delete process.env.GEMINI_API_KEY;

    expect(() => {
      new GeminiEmbeddingProvider({
        provider: "gemini",
      });
    }).toThrow("Gemini API key required");

    if (originalKey) process.env.GEMINI_API_KEY = originalKey;
  });

  test("uses environment variable for API key", () => {
    const originalKey = process.env.GEMINI_API_KEY;
    process.env.GEMINI_API_KEY = "env-test-key";

    const provider = new GeminiEmbeddingProvider({
      provider: "gemini",
    });
    expect(provider.name).toBe("gemini");

    if (originalKey) {
      process.env.GEMINI_API_KEY = originalKey;
    } else {
      delete process.env.GEMINI_API_KEY;
    }
  });

  test("uses default model", () => {
    const provider = new GeminiEmbeddingProvider({
      provider: "gemini",
      api_key: "test-key",
    });
    expect(provider.modelId).toBe("text-embedding-004");
    expect(provider.dimensions).toBe(768);
  });

  test("uses custom model", () => {
    const provider = new GeminiEmbeddingProvider({
      provider: "gemini",
      model: "embedding-001",
      api_key: "test-key",
    });
    expect(provider.modelId).toBe("embedding-001");
    expect(provider.dimensions).toBe(768);
  });
});

// Integration tests would require real API keys and make actual network requests
// These are skipped by default but can be enabled for manual testing
describe.skip("OpenAI Integration Tests", () => {
  test("embeds single text", async () => {
    if (!process.env.OPENAI_API_KEY) {
      console.log("Skipping OpenAI integration test - no API key");
      return;
    }

    const provider = new OpenAIEmbeddingProvider({
      provider: "openai",
      api_key: process.env.OPENAI_API_KEY,
    });

    const result = await provider.embed("test text", false);
    expect(result).not.toBeNull();
    expect(result!.embedding.length).toBe(1536);
    expect(result!.model).toBe("text-embedding-3-small");
  });

  test("embeds batch", async () => {
    if (!process.env.OPENAI_API_KEY) {
      console.log("Skipping OpenAI integration test - no API key");
      return;
    }

    const provider = new OpenAIEmbeddingProvider({
      provider: "openai",
      api_key: process.env.OPENAI_API_KEY,
    });

    const results = await provider.embedBatch(["text 1", "text 2", "text 3"], false);
    expect(results.length).toBe(3);
    expect(results.every(r => r !== null)).toBe(true);
    expect(results.every(r => r!.embedding.length === 1536)).toBe(true);
  });
});

describe.skip("Gemini Integration Tests", () => {
  test("embeds single text", async () => {
    if (!process.env.GEMINI_API_KEY) {
      console.log("Skipping Gemini integration test - no API key");
      return;
    }

    const provider = new GeminiEmbeddingProvider({
      provider: "gemini",
      api_key: process.env.GEMINI_API_KEY,
    });

    const result = await provider.embed("test text", false);
    expect(result).not.toBeNull();
    expect(result!.embedding.length).toBe(768);
    expect(result!.model).toBe("text-embedding-004");
  });

  test("embeds batch", async () => {
    if (!process.env.GEMINI_API_KEY) {
      console.log("Skipping Gemini integration test - no API key");
      return;
    }

    const provider = new GeminiEmbeddingProvider({
      provider: "gemini",
      api_key: process.env.GEMINI_API_KEY,
    });

    const results = await provider.embedBatch(["text 1", "text 2", "text 3"], false);
    expect(results.length).toBe(3);
    expect(results.every(r => r !== null)).toBe(true);
    expect(results.every(r => r!.embedding.length === 768)).toBe(true);
  });
});
