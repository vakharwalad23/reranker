import * as natural from "natural";
import stringSimilarity from "string-similarity";
import nlp from "compromise";
import Fuse from "fuse.js";
import { removeStopwords, eng } from "stopword";
import { ExcludeFactor, RerankItem } from "../types";

export class SmartReranker {
  private tfidf: any;
  private stemmer = natural.PorterStemmer;
  private tokenizer = new natural.WordTokenizer();

  constructor() {
    this.tfidf = new natural.TfIdf();
  }

  private preprocessText(text: string): string[] {
    try {
      const tokens = this.tokenizer.tokenize(text.toLowerCase()) || [];
      const filtered = tokens.filter((token) => token.length > 2);
      const withoutStopwords = removeStopwords(filtered, eng);
      return withoutStopwords.map((token) => this.stemmer.stem(token));
    } catch (error) {
      return text
        .toLowerCase()
        .replace(/[^\w\s]/g, " ")
        .split(/\s+/)
        .filter((token) => token.length > 2);
    }
  }

  private getSemanticFeatures(text: string) {
    try {
      const doc = nlp(text);
      return {
        nouns: doc.nouns().out("array"),
        verbs: doc.verbs().out("array"),
        adjectives: doc.adjectives().out("array"),
        topics: doc.topics().out("array"),
        phrases: this.extractPhrases(text),
        concepts: this.extractConcepts(text),
      };
    } catch (error) {
      return {
        nouns: this.extractSimpleNouns(text),
        verbs: [],
        adjectives: [],
        topics: [],
        phrases: this.extractPhrases(text),
        concepts: this.extractConcepts(text),
      };
    }
  }

  private extractSimpleNouns(text: string): string[] {
    const words = text.toLowerCase().split(/\s+/);
    return words.filter((word) => word.length > 3);
  }

  private extractPhrases(text: string): string[] {
    const words = text
      .toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/);
    const phrases: string[] = [];

    for (let i = 0; i < words.length - 1; i++) {
      if (words[i].length > 2 && words[i + 1].length > 2) {
        phrases.push(`${words[i]} ${words[i + 1]}`);
      }
    }

    for (let i = 0; i < words.length - 2; i++) {
      if (
        words[i].length > 2 &&
        words[i + 1].length > 2 &&
        words[i + 2].length > 2
      ) {
        phrases.push(`${words[i]} ${words[i + 1]} ${words[i + 2]}`);
      }
    }

    return phrases;
  }

  private extractConcepts(text: string): string[] {
    const concepts: string[] = [];
    const words = text.split(/\s+/);

    words.forEach((word) => {
      if (word.match(/^[A-Z][A-Z]+$/) || word.match(/^[A-Z][a-z]+$/)) {
        concepts.push(word.toLowerCase());
      }
      if (word.includes("-") && word.length > 5) {
        concepts.push(word.toLowerCase());
      }
    });

    return concepts;
  }

  private calculateTfIdfSimilarity(
    query: string,
    content: string,
    allContents: string[]
  ): number {
    try {
      this.tfidf = new natural.TfIdf();

      allContents.forEach((doc) => {
        this.tfidf.addDocument(this.preprocessText(doc).join(" "));
      });

      this.tfidf.addDocument(this.preprocessText(query).join(" "));

      const queryIndex = allContents.length;
      const contentIndex = allContents.findIndex((doc) => doc === content);

      if (contentIndex === -1) return 0;

      const queryTerms = this.tfidf.listTerms(queryIndex);
      const contentTerms = this.tfidf.listTerms(contentIndex);

      if (queryTerms.length === 0 || contentTerms.length === 0) return 0;

      const queryMap = new Map<string, number>();
      const contentMap = new Map<string, number>();

      queryTerms.forEach((term: any) => {
        queryMap.set(term.term, term.tfidf);
      });
      contentTerms.forEach((term: any) => {
        contentMap.set(term.term, term.tfidf);
      });

      let overlap = 0;
      let queryWeight = 0;

      queryMap.forEach((weight, term) => {
        queryWeight += weight;
        if (contentMap.has(term)) {
          overlap += Math.min(weight, contentMap.get(term) || 0);
        }
      });

      return queryWeight > 0 ? overlap / queryWeight : 0;
    } catch (error) {
      console.error("TF-IDF calculation error:", error);
      return 0;
    }
  }

  private calculateStringSimilarity(query: string, content: string): number {
    try {
      const diceScore = stringSimilarity.compareTwoStrings(
        query.toLowerCase(),
        content.toLowerCase()
      );

      const jaroScore = natural.JaroWinklerDistance(
        query.toLowerCase(),
        content.toLowerCase()
      );

      const queryTerms = new Set(query.toLowerCase().split(/\s+/));
      const contentTerms = new Set(content.toLowerCase().split(/\s+/));
      const intersection = new Set(
        [...queryTerms].filter((x) => contentTerms.has(x))
      );
      const overlapScore =
        queryTerms.size > 0 ? intersection.size / queryTerms.size : 0;

      // Weighted average with higher weight on term overlap
      return diceScore * 0.3 + jaroScore * 0.2 + overlapScore * 0.5;
    } catch (error) {
      console.error("String similarity calculation error:", error);
      return 0;
    }
  }

  private calculateSemanticSimilarity(query: string, content: string): number {
    try {
      const queryFeatures = this.getSemanticFeatures(query);
      const contentFeatures = this.getSemanticFeatures(content);

      let totalScore = 0;
      let weights = 0;

      const phraseScore = this.calculateOverlap(
        queryFeatures.phrases,
        contentFeatures.phrases
      );
      if (phraseScore > 0) {
        totalScore += phraseScore * 0.4;
        weights += 0.4;
      }

      const conceptScore = this.calculateOverlap(
        queryFeatures.concepts,
        contentFeatures.concepts
      );
      if (conceptScore > 0) {
        totalScore += conceptScore * 0.3;
        weights += 0.3;
      }

      const nounScore = this.calculateOverlap(
        queryFeatures.nouns,
        contentFeatures.nouns
      );
      if (nounScore > 0) {
        totalScore += nounScore * 0.2;
        weights += 0.2;
      }

      const topicScore = this.calculateOverlap(
        queryFeatures.topics,
        contentFeatures.topics
      );
      if (topicScore > 0) {
        totalScore += topicScore * 0.1;
        weights += 0.1;
      }

      return weights > 0 ? totalScore / weights : 0;
    } catch (error) {
      console.error("Semantic similarity calculation error:", error);
      return 0;
    }
  }

  private calculateOverlap(arr1: string[], arr2: string[]): number {
    if (arr1.length === 0) return 0;

    const set1 = new Set(arr1.map((item) => item.toLowerCase()));
    const set2 = new Set(arr2.map((item) => item.toLowerCase()));

    const intersection = new Set([...set1].filter((x) => set2.has(x)));
    return intersection.size / set1.size;
  }

  private calculateFuzzyScore(
    query: string,
    items: RerankItem[]
  ): Map<string, number> {
    try {
      const fuse = new Fuse(items, {
        keys: ["content"],
        includeScore: true,
        threshold: 0.6,
        distance: 100,
        ignoreLocation: true,
        findAllMatches: true,
      });

      const results = fuse.search(query);
      const scoreMap = new Map<string, number>();

      results.forEach((result) => {
        // Fuse.js score is 0 (perfect) to 1 (no match), so invert it
        const normalizedScore = result.score
          ? Math.max(0, 1 - result.score)
          : 0;
        scoreMap.set(result.item.id, normalizedScore);
      });

      return scoreMap;
    } catch (error) {
      console.error("Fuzzy search error:", error);
      return new Map();
    }
  }

  private calculateRecencyScore(content: string): number {
    const currentYear = new Date().getFullYear();
    const yearMatches = content.match(/\b(20[0-2][0-9])\b/g);

    if (!yearMatches) return 0.7;

    const years = yearMatches.map((y) => parseInt(y));
    const mostRecentYear = Math.max(...years);

    const yearDiff = currentYear - mostRecentYear;
    if (yearDiff <= 1) return 1.0;
    if (yearDiff <= 2) return 0.9;
    if (yearDiff <= 5) return 0.8;
    return Math.max(0.5, 1 - yearDiff / 10);
  }

  private calculateLengthScore(query: string, content: string): number {
    const queryLen = query.length;
    const contentLen = content.length;

    if (contentLen < 50) return 0.3;
    if (contentLen < queryLen * 0.5) return 0.5;
    if (contentLen > 1000) return 0.7;

    return 0.8;
  }

  private calculateQueryTermMatch(query: string, content: string): number {
    const queryTerms = query
      .toLowerCase()
      .split(/\s+/)
      .filter((term) => term.length > 2);
    const contentLower = content.toLowerCase();

    if (queryTerms.length === 0) return 0;

    const matchedTerms = queryTerms.filter((term) =>
      contentLower.includes(term)
    );
    return matchedTerms.length / queryTerms.length;
  }

  public rerank(
    query: string,
    items: RerankItem[],
    excludeFactors: ExcludeFactor[] = []
  ): RerankItem[] {
    const allContents = items.map((item) => item.content);
    const fuzzyScores = this.calculateFuzzyScore(query, items);

    return items
      .map((item) => {
        const scores = {
          vectorScore: excludeFactors.includes("vectorScore")
            ? 0
            : this.calculateTfIdfSimilarity(query, item.content, allContents),
          semanticScore: excludeFactors.includes("semanticScore")
            ? 0
            : this.calculateSemanticSimilarity(query, item.content),
          queryTermMatch: excludeFactors.includes("queryTermMatch")
            ? 0
            : this.calculateQueryTermMatch(query, item.content),
          recency: excludeFactors.includes("recency")
            ? 0
            : this.calculateRecencyScore(item.content),
          length: excludeFactors.includes("length")
            ? 0
            : this.calculateLengthScore(query, item.content),
        };

        const stringScore = this.calculateStringSimilarity(query, item.content);
        const fuzzyScore = fuzzyScores.get(item.id) || 0;

        const activeFactors = Object.values(scores).filter(
          (score) => score > 0
        ).length;
        const hasActiveFactors = activeFactors > 0;

        // Adjust weights based on active factors
        let finalScore = 0;
        if (hasActiveFactors) {
          finalScore =
            scores.vectorScore * 0.25 +
            scores.semanticScore * 0.25 +
            scores.queryTermMatch * 0.2 +
            stringScore * 0.15 +
            fuzzyScore * 0.1 +
            scores.recency * 0.03 +
            scores.length * 0.02;
        } else {
          finalScore = stringScore * 0.6 + fuzzyScore * 0.4;
        }

        return {
          ...item,
          finalScore,
          length: item.content.length,
          debug: {
            vectorScore: scores.vectorScore,
            semanticScore: scores.semanticScore,
            queryTermMatch: scores.queryTermMatch,
            stringScore,
            fuzzyScore,
            recency: scores.recency,
            lengthScore: scores.length,
            excludedFactors: excludeFactors,
          },
        };
      })
      .sort((a, b) => (b.finalScore || 0) - (a.finalScore || 0));
  }
}
