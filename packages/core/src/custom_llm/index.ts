/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
} from '@google/genai';
import OpenAI from 'openai';
import type { ContentGenerator } from '../core/contentGenerator.js';
import {
  extractToolFunctions,
  normalizeContents,
} from './util.js';
import type {
  CustomLLMContentGeneratorConfig,
  ToolCallMap,
} from './types.js';
import { ModelConverter } from './converter.js';

interface CustomLLMOptions {
  userAgent?: string;
}

export class CustomLLMContentGenerator implements ContentGenerator {
  private readonly client: OpenAI;
  private readonly apiKey: string;
  private readonly baseURL: string;
  private readonly modelName: string;
  private readonly temperature: number;
  private readonly maxTokens: number;
  private readonly topP: number;
  private readonly config: CustomLLMContentGeneratorConfig;

  constructor(options: CustomLLMOptions = {}) {
    this.apiKey = process.env['CUSTOM_LLM_API_KEY'] ?? '';
    this.baseURL = process.env['CUSTOM_LLM_ENDPOINT'] ?? '';
    this.modelName = process.env['CUSTOM_LLM_MODEL_NAME'] ?? '';
    this.temperature = Number(process.env['CUSTOM_LLM_TEMPERATURE'] ?? 0);
    this.maxTokens = Number(process.env['CUSTOM_LLM_MAX_TOKENS'] ?? 8192);
    this.topP = Number(process.env['CUSTOM_LLM_TOP_P'] ?? 1);

    this.config = {
      model: this.modelName,
      temperature: this.temperature,
      max_tokens: this.maxTokens,
      top_p: this.topP,
      stream_options: {
        include_usage: true,
      },
    };

    this.client = new OpenAI({
      apiKey: this.apiKey,
      baseURL: this.baseURL || undefined,
      defaultHeaders: options.userAgent
        ? { 'User-Agent': options.userAgent }
        : undefined,
    });
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const messages = ModelConverter.toOpenAIMessages(request);
    const completion = await this.client.chat.completions.create({
      messages,
      stream: false,
      ...this.config,
    });
    return ModelConverter.toGeminiResponse(completion);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = ModelConverter.toOpenAIMessages(request);
    const tools = extractToolFunctions(request.config) ?? [];
    const stream = await this.client.chat.completions.create({
      messages,
      stream: true,
      tools,
      ...this.config,
    });
    const toolCallMap: ToolCallMap = new Map();

    return (async function* (): AsyncGenerator<GenerateContentResponse> {
      for await (const chunk of stream) {
        const { response } = ModelConverter.processStreamChunk(
          chunk,
          toolCallMap,
        );
        if (response) {
          yield response;
        }
      }
    })();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    const contents = normalizeContents(request.contents ?? []);
    const text = contents
      .flatMap((content) =>
        content.parts
          ?.map((part) => {
            if (typeof part === 'string') {
              return part;
            }
            if (typeof part === 'object' && part !== null) {
              if ('text' in part && typeof part.text === 'string') {
                return part.text;
              }
              if ('inlineData' in part && part.inlineData?.data) {
                return part.inlineData.data;
              }
            }
            return '';
          })
          .filter(Boolean) ?? [],
      )
      .join(' ');

    const englishWords = (text.match(/[a-zA-Z]+[']?[a-zA-Z]*/g) ?? []).length;
    const chineseChars = (text.match(/[\u4e00-\u9fff]/g) ?? []).length;
    const numbers = (text.match(/\b\d+\b/g) ?? []).length;
    const punctuations =
      text.match(/[.,!?;:"'(){}[\]<>@#$%^&*\-_+=~`|\\/]/g)?.length ?? 0;
    const spaces = Math.ceil(((text.match(/\s+/g) ?? []).length) / 5);

    const totalTokens = Math.ceil(
      englishWords * 1.2 +
        chineseChars * 1 +
        numbers * 0.8 +
        punctuations * 0.5 +
        spaces,
    );

    return {
      totalTokens,
    };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error('Embedding is not supported for custom LLM providers.');
  }
}
