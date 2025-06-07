export interface RerankItem {
  id: string;
  content: string;
  length?: number;
  finalScore?: number;
}

export type ExcludeFactor =
  | "vectorScore"
  | "semanticScore"
  | "length"
  | "recency"
  | "queryTermMatch";

export interface RerankRequest {
  query: string;
  items: RerankItem[];
  excludeFactors?: ExcludeFactor[];
  mode?: "math" | "ai";
  topK?: number;
}

export interface RerankResponse {
  items: RerankItem[];
  method: "math" | "ai";
  cached?: boolean;
}
