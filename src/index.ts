import { Hono } from "hono";
import { cors } from "hono/cors";
import { logger } from "hono/logger";
import { CACHE_CONFIG } from "./config";

const app = new Hono<{ Bindings: CloudflareBindings }>();

app.use("*", cors());
app.use("*", logger());

// Math-based reranking functions
function calculateJaccardSimilarity(query: string, content: string): number {
  const queryTokens = new Set(query.toLowerCase().split(/\s+/));
  const contentTokens = new Set(content.toLowerCase().split(/\s+/));

  const intersection = new Set(
    [...queryTokens].filter((x) => contentTokens.has(x))
  );
  const union = new Set([...queryTokens, ...contentTokens]);

  return union.size === 0 ? 0 : intersection.size / union.size;
}

function calculateQueryTermMatch(query: string, content: string): number {
  const queryTerms = query.toLowerCase().split(/\s+/);
  const contentLower = content.toLowerCase();

  const matchedTerms = queryTerms.filter((term) => contentLower.includes(term));
  return queryTerms.length > 0 ? matchedTerms.length / queryTerms.length : 0;
}

function calculateRecencyScore(content: string): number {
  // Look for year patterns (2020-2024 get higher scores)
  const currentYear = new Date().getFullYear();
  const yearMatches = content.match(/\b(20[0-2][0-9])\b/g);

  if (!yearMatches) return 0.5; // Neutral score if no years found

  const years = yearMatches.map((y) => parseInt(y));
  const mostRecentYear = Math.max(...years);

  // Score based on how recent the year is (max 5 years back)
  const yearDiff = currentYear - mostRecentYear;
  return Math.max(0, 1 - yearDiff / 5);
}

function calculateVectorScore(query: string, content: string): number {
  // Simplified vector-like scoring using character n-grams
  const getCharNgrams = (text: string, n: number = 3) => {
    const ngrams = new Set<string>();
    const cleanText = text.toLowerCase().replace(/[^a-z0-9]/g, "");
    for (let i = 0; i <= cleanText.length - n; i++) {
      ngrams.add(cleanText.slice(i, i + n));
    }
    return ngrams;
  };

  const queryNgrams = getCharNgrams(query);
  const contentNgrams = getCharNgrams(content);

  const intersection = new Set(
    [...queryNgrams].filter((x) => contentNgrams.has(x))
  );
  const union = new Set([...queryNgrams, ...contentNgrams]);

  return union.size === 0 ? 0 : intersection.size / union.size;
}

function calculateSemanticScore(query: string, content: string): number {
  // Combine multiple semantic indicators
  const jaccardScore = calculateJaccardSimilarity(query, content);
  const editScore = calculateEditDistanceScore(query, content);
  const termMatchScore = calculateQueryTermMatch(query, content);

  // Weighted average of semantic indicators
  return jaccardScore * 0.4 + editScore * 0.3 + termMatchScore * 0.3;
}

function calculateLengthScore(query: string, content: string): number {
  // Prefer content that's similar in length to query, but not too short
  const queryLen = query.length;
  const contentLen = content.length;

  if (contentLen < queryLen * 0.5) {
    return 0.3; // Penalize very short content
  }

  const ratio = Math.min(queryLen, contentLen) / Math.max(queryLen, contentLen);
  return ratio * 0.8 + 0.2; // Base score of 0.2, up to 1.0
}

function calculateLevenshteinDistance(a: string, b: string): number {
  const matrix = Array(b.length + 1)
    .fill(null)
    .map(() => Array(a.length + 1).fill(null));

  for (let i = 0; i <= a.length; i++) matrix[0][i] = i;
  for (let j = 0; j <= b.length; j++) matrix[j][0] = j;

  for (let j = 1; j <= b.length; j++) {
    for (let i = 1; i <= a.length; i++) {
      const indicator = a[i - 1] === b[j - 1] ? 0 : 1;
      matrix[j][i] = Math.min(
        matrix[j][i - 1] + 1,
        matrix[j - 1][i] + 1,
        matrix[j - 1][i - 1] + indicator
      );
    }
  }

  return matrix[b.length][a.length];
}

function calculateEditDistanceScore(query: string, content: string): number {
  const distance = calculateLevenshteinDistance(
    query.toLowerCase(),
    content.toLowerCase()
  );
  const maxLength = Math.max(query.length, content.length);
  return maxLength === 0 ? 1 : 1 - distance / maxLength;
}

function calculateBM25Score(
  query: string,
  content: string,
  avgDocLength: number = 100
): number {
  const k1 = 1.2;
  const b = 0.75;

  const queryTerms = query.toLowerCase().split(/\s+/);
  const contentTokens = content.toLowerCase().split(/\s+/);
  const docLength = contentTokens.length;

  let score = 0;

  for (const term of queryTerms) {
    const termFreq = contentTokens.filter((token) => token === term).length;
    if (termFreq > 0) {
      const idf = Math.log((1 + 1) / (1 + termFreq)); // Simplified IDF
      const tf =
        (termFreq * (k1 + 1)) /
        (termFreq + k1 * (1 - b + b * (docLength / avgDocLength)));
      score += idf * tf;
    }
  }

  return score;
}

function mathRerank(
  query: string,
  items: RerankItem[],
  excludeFactors: Array<
    "vectorScore" | "semanticScore" | "length" | "recency" | "queryTermMatch"
  > = []
): RerankItem[] {
  return items
    .map((item) => {
      const factors = {
        vectorScore: calculateVectorScore(query, item.content),
        semanticScore: calculateSemanticScore(query, item.content),
        length: calculateLengthScore(query, item.content),
        recency: calculateRecencyScore(item.content),
        queryTermMatch: calculateQueryTermMatch(query, item.content),
      };

      // Apply exclusions
      const activeFactors = Object.entries(factors)
        .filter(
          ([key]) => !excludeFactors.includes(key as keyof typeof factors)
        )
        .map(([, value]) => value);

      const finalScore =
        activeFactors.length > 0
          ? activeFactors.reduce((sum, score) => sum + score, 0) /
            activeFactors.length
          : 0;

      return {
        ...item,
        length: item.content.length,
        finalScore,
      };
    })
    .sort((a, b) => (b.finalScore || 0) - (a.finalScore || 0));
}

async function aiRerank(
  query: string,
  items: RerankItem[],
  ai: CloudflareBindings["AI"],
  topK?: number
): Promise<RerankItem[]> {
  try {
    // Prepare contexts in the format expected by BGE reranker
    const contexts = items.map((item) => ({
      text: item.content,
    }));

    const input = {
      query: query,
      contexts: contexts,
      ...(topK && { top_k: Math.min(topK, items.length) }),
    };

    const response = await ai.run("@cf/baai/bge-reranker-base", input);

    if (!response || !response.response || !Array.isArray(response.response)) {
      return mathRerank(query, items);
    }

    // Map the BGE response back to our items
    const rankedResults = response.response
      .filter(
        (result): result is { id: number; score: number } =>
          typeof result.id === "number" && typeof result.score === "number"
      )
      .map((result: { id: number; score: number }) => ({
        ...items[result.id],
        finalScore: result.score,
      }));

    // Sort by score in descending order
    return rankedResults.sort(
      (a, b) => (b.finalScore || 0) - (a.finalScore || 0)
    );
  } catch (error) {
    // Fallback to math reranking
    return mathRerank(query, items);
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

    // Validation
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

    // Generate cache key
    const cacheKey = await generateCacheKey(query, items, mode, excludeFactors);

    // Check cache first
    let cached = false;
    try {
      const cachedResult = await c.env.RERANK_CACHE.get(cacheKey);
      if (cachedResult) {
        const parsedResult: RerankResponse = JSON.parse(cachedResult);
        parsedResult.cached = true;

        // Apply topK if specified
        if (topK && topK > 0) {
          parsedResult.items = parsedResult.items.slice(0, topK);
        }

        return c.json(parsedResult);
      }
    } catch (cacheError) {
      console.error("Cache read error:", cacheError);
    }

    // Perform reranking
    let rankedItems: RerankItem[];

    if (mode === "ai") {
      rankedItems = await aiRerank(query, items, c.env.AI, topK);
    } else {
      rankedItems = mathRerank(query, items, excludeFactors);
    }

    // Apply topK if specified
    if (topK && topK > 0) {
      rankedItems = rankedItems.slice(0, topK);
    }

    const response: RerankResponse = {
      items: rankedItems,
      method: mode,
      cached,
    };

    // Cache the result
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

// Handle 404
app.notFound((c) => {
  return c.json({ error: "Endpoint not found" }, 404);
});

// Handle errors
app.onError((err, c) => {
  console.error("Unhandled error:", err);
  return c.json({ error: "Internal server error" }, 500);
});

export default app;
