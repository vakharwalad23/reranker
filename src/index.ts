import { Hono } from "hono";
import { cors } from "hono/cors";
import { logger } from "hono/logger";
import { CACHE_CONFIG } from "./config";
import {
  ExcludeFactor,
  RerankItem,
  RerankRequest,
  RerankResponse,
} from "./types";

const app = new Hono<{ Bindings: CloudflareBindings }>();

app.use(
  "*",
  cors({
    origin: "*",
    allowMethods: ["GET", "POST"],
  })
);
app.use("*", logger());

// Lazy load SmartReranker to avoid startup overhead
let smartRerankerInstance: any = null;

async function getSmartReranker() {
  if (!smartRerankerInstance) {
    const { SmartReranker } = await import("./services/SmartReranker");
    smartRerankerInstance = new SmartReranker();
  }
  return smartRerankerInstance;
}

async function mathRerank(
  query: string,
  items: RerankItem[],
  excludeFactors: ExcludeFactor[] = []
): Promise<RerankItem[]> {
  const reranker = await getSmartReranker();
  return reranker.rerank(query, items, excludeFactors);
}

async function aiRerank(
  query: string,
  items: RerankItem[],
  ai: CloudflareBindings["AI"],
  topK?: number
): Promise<RerankItem[]> {
  try {
    const contexts = items.map((item) => ({
      text: item.content,
    }));

    const input = {
      query: query,
      contexts: contexts,
      ...(topK && { top_k: Math.min(topK, items.length) }),
    };

    const response = await ai.run("@cf/baai/bge-reranker-base", input, {
      gateway: {
        id: "reranker",
      },
    });

    if (!response || !response.response || !Array.isArray(response.response)) {
      return await mathRerank(query, items);
    }

    const rankedResults = response.response
      .filter(
        (result): result is { id: number; score: number } =>
          typeof result.id === "number" && typeof result.score === "number"
      )
      .map((result: { id: number; score: number }) => ({
        ...items[result.id],
        finalScore: result.score,
      }));

    return rankedResults.sort(
      (a, b) => (b.finalScore || 0) - (a.finalScore || 0)
    );
  } catch (error) {
    return await mathRerank(query, items);
  }
}

async function generateCacheKey(
  query: string,
  items: RerankItem[],
  mode: string,
  excludeFactors: string[]
): Promise<string> {
  const itemsHash = items.map((item) => `${item.id}:${item.content}`).join("|");
  const key = `${query}:${itemsHash}:${mode}:${excludeFactors
    .sort()
    .join(",")}`;

  const encoder = new TextEncoder();
  const data = encoder.encode(key);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

  return `${CACHE_CONFIG.keyPrefix}${hashHex.slice(0, 16)}`;
}

app.get("/", (c) => {
  return c.json({
    service: "Reranker API",
    version: "1.0.0",
    endpoints: {
      rerank: "POST /rerank",
    },
  });
});

app.get("/health", (c) => {
  return c.json({ status: "healthy", timestamp: new Date().toISOString() });
});

app.post("/rerank", async (c) => {
  try {
    const body: RerankRequest = await c.req.json();
    const { query, items, excludeFactors = [], mode = "math", topK } = body;

    if (!query || !items || !Array.isArray(items)) {
      return c.json(
        { error: "Invalid request. Query and items array are required." },
        400
      );
    }

    if (items.length === 0) {
      return c.json({ error: "Items array cannot be empty." }, 400);
    }

    if (items.some((item) => !item.id || !item.content)) {
      return c.json(
        { error: "All items must have id and content fields." },
        400
      );
    }

    const cacheKey = await generateCacheKey(query, items, mode, excludeFactors);

    let cached = false;
    try {
      const cachedResult = await c.env.RERANK_CACHE.get(cacheKey);
      if (cachedResult) {
        const parsedResult: RerankResponse = JSON.parse(cachedResult);
        parsedResult.cached = true;

        if (topK && topK > 0) {
          parsedResult.items = parsedResult.items.slice(0, topK);
        }

        return c.json(parsedResult);
      }
    } catch (cacheError) {
      console.error("Cache read error:", cacheError);
    }

    let rankedItems: RerankItem[];

    if (mode === "ai") {
      rankedItems = await aiRerank(query, items, c.env.AI, topK);
    } else {
      rankedItems = await mathRerank(query, items, excludeFactors);
    }

    if (topK && topK > 0) {
      rankedItems = rankedItems.slice(0, topK);
    }

    const response: RerankResponse = {
      items: rankedItems,
      method: mode,
      cached,
    };

    try {
      await c.env.RERANK_CACHE.put(cacheKey, JSON.stringify(response), {
        expirationTtl: CACHE_CONFIG.ttl,
      });
    } catch (cacheError) {
      console.error("Cache write error:", cacheError);
    }

    return c.json(response);
  } catch (error) {
    console.error("Rerank error:", error);
    return c.json({ error: "Internal server error" }, 500);
  }
});

app.notFound((c) => {
  return c.json({ error: "Endpoint not found" }, 404);
});

app.onError((err, c) => {
  console.error("Unhandled error:", err);
  return c.json({ error: "Internal server error" }, 500);
});

export default app;
