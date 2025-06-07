# 🚀 Smart Reranker API

<div align="center">

[![Cloudflare Workers](https://img.shields.io/badge/Cloudflare-Workers-orange?style=for-the-badge&logo=cloudflare)](https://workers.cloudflare.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Hono](https://img.shields.io/badge/Hono-E36002?style=for-the-badge&logo=hono&logoColor=fff)](https://hono.dev/)
[![Bun](https://img.shields.io/badge/Bun-000000?style=for-the-badge&logo=bun&logoColor=white)](https://bun.sh/)

_Intelligent content reranking powered by AI and advanced algorithms_

</div>

## ✨ Features

- 🧠 **Dual Reranking Modes**: Math-based and AI-powered reranking
- ⚡ **Lightning Fast**: Built on Cloudflare Workers for global edge performance
- 🎯 **Smart Algorithms**: TF-IDF, semantic analysis, fuzzy matching, and more
- 🔄 **Intelligent Caching**: Cloudflare KV-powered caching for optimal performance
- 🛡️ **Type Safe**: Full TypeScript support with auto-generated types
- 🎨 **Flexible Scoring**: Customizable factor exclusion for fine-tuned results

## 🚀 Quick Start

### Development

```bash
# Install dependencies
bun install

# Start development server
bun run dev

# Generate Cloudflare types
bun run cf-typegen
```

### Deployment

```bash
# Deploy to Cloudflare Workers
bun run deploy
```

## 📋 API Reference

### `POST /rerank`

Rerank a list of items based on a query using intelligent scoring algorithms.

#### Parameters

| Parameter        | Type                                    | Default      | Description                     |
| ---------------- | --------------------------------------- | ------------ | ------------------------------- |
| `query`          | `string`                                | **required** | Search query to rank against    |
| `items`          | [`RerankItem[]`](src/types/index.ts)    | **required** | Array of items to rerank        |
| `mode`           | `"math" \| "ai"`                        | `"math"`     | Reranking algorithm mode        |
| `topK`           | `number`                                | `undefined`  | Limit results to top K items    |
| `excludeFactors` | [`ExcludeFactor[]`](src/types/index.ts) | `[]`         | Factors to exclude from scoring |

### Exclude Factors

Fine-tune your reranking by excluding specific scoring factors:

- `vectorScore` - TF-IDF vector similarity
- `semanticScore` - NLP semantic analysis
- `length` - Content length optimization
- `recency` - Temporal relevance scoring
- `queryTermMatch` - Direct query term matching

## 🎯 Reranking Modes

### 🧮 Math Mode (`"math"`)

Advanced mathematical reranking using:

- **TF-IDF Vectorization** - Statistical term importance
- **Semantic Analysis** - NLP-powered concept extraction
- **String Similarity** - Jaro-Winkler & Dice coefficients
- **Fuzzy Matching** - Flexible text matching with Fuse.js
- **Multi-factor Scoring** - Weighted combination of relevance signals

### 🤖 AI Mode (`"ai"`)

Powered by Cloudflare's AI models:

- **BGE Reranker Base** - State-of-the-art neural reranking
- **Automatic Fallback** - Falls back to math mode if AI fails
- **Context-Aware** - Understanding of semantic relationships

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                 Cloudflare Edge                 │
├─────────────────────────────────────────────────┤
│  🌐 Hono API Router                            │
│  ├── CORS & Logging Middleware                 │
│  ├── Request Validation                        │
│  └── Error Handling                            │
├─────────────────────────────────────────────────┤
│  🧠 Smart Reranking Engine                     │
│  ├── Math Mode: Multi-Algorithm Scoring        │
│  │   ├── TF-IDF Vector Analysis               │
│  │   ├── Semantic Feature Extraction          │
│  │   ├── String Similarity Metrics            │
│  │   └── Fuzzy Search Integration             │
│  └── AI Mode: Neural Reranking                 │
│      └── Cloudflare AI Workers Integration     │
├─────────────────────────────────────────────────┤
│  ⚡ Performance Layer                           │
│  ├── SHA-256 Cache Key Generation              │
│  ├── Cloudflare KV Caching                    │
│  └── Lazy Loading Optimization                 │
└─────────────────────────────────────────────────┘
```

## 🔧 Configuration

The service is configured through [Cloudflare bindings](worker-configuration.d.ts):

- **[`AI`](worker-configuration.d.ts)** - Cloudflare AI binding for neural reranking
- **[`RERANK_CACHE`](worker-configuration.d.ts)** - KV namespace for caching results

Cache configuration is managed in [src/config/index.ts](src/config/index.ts):

```typescript
export const CACHE_CONFIG = {
  ttl: 3600, // 1 hour
  keyPrefix: "rerank:",
};
```

## 📊 Performance

- **Global Edge Deployment** - Sub-100ms response times worldwide
- **Intelligent Caching** - 1-hour TTL with SHA-256 cache keys stored in [Cloudflare KV](worker-configuration.d.ts)
- **Lazy Loading** - Smart module loading for optimal cold start performance
- **Automatic Fallbacks** - Robust error handling with graceful degradation

## 🛠️ Development

### Project Structure

```
src/
├── index.ts              # Main API routes and handlers
├── config/
│   └── index.ts          # Configuration constants
├── services/
│   └── SmartReranker.ts  # Core reranking algorithms
└── types/
    └── index.ts          # TypeScript type definitions
```

### Adding New Scoring Factors

1. Update the [`ExcludeFactor`](src/types/index.ts) type
2. Implement the scoring logic in [`SmartReranker`](src/services/SmartReranker.ts)
3. Add the factor to the weighted scoring calculation in [`rerank`](src/services/SmartReranker.ts)

### Environment Setup

This project uses **Bun** as the JavaScript runtime and package manager. Make sure you have Bun installed:

```bash
# Install Bun
curl -fsSL https://bun.sh/install | bash

# Verify installation
bun --version
```

### Available Scripts

```bash
# Development
bun run dev          # Start development server
bun run cf-typegen   # Generate Cloudflare Worker types

# Deployment
bun run deploy       # Deploy to Cloudflare Workers

# Testing (if available)
./test.sh           # Run tests
./testCompre.sh     # Run comprehensive tests
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ using Cloudflare Workers + Bun**

_Rerank smarter, not harder_ 🎯

</div>
