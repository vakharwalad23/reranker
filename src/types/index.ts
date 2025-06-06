interface RerankItem {
  id: string;
  content: string;
  length?: number;
  finalScore?: number;
}

interface RerankRequest {
  query: string;
  items: RerankItem[];
  excludeFactors?: Array<
    "vectorScore" | "semanticScore" | "length" | "recency" | "queryTermMatch"
  >;
  mode?: "math" | "ai";
  topK?: number;
}

interface RerankResponse {
  items: RerankItem[];
  method: "math" | "ai";
  cached?: boolean;
}
